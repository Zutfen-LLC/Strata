#include "strata/worker/cpu_capture.hpp"
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
namespace strata::worker {
namespace {
constexpr uint32_t Version=1, Window=1, Infer=2, MaxRecords=16384, MaxWindows=128;
const uint8_t Magic[8]={'S','C','C','P','U','0','0','1'};
void set32(uint8_t* p,uint32_t v){for(int i=0;i<4;++i)p[i]=(uint8_t)(v>>(8*i));}
void set64(uint8_t* p,uint64_t v){for(int i=0;i<8;++i)p[i]=(uint8_t)(v>>(8*i));}
void digest(uint8_t* p,Digest128 d){set64(p,d.a);set64(p+8,d.b);}
uint32_t get32(const uint8_t* p){uint32_t v=0;for(int i=0;i<4;++i)v|=(uint32_t)p[i]<<(8*i);return v;}
uint64_t get64(const uint8_t* p){uint64_t v=0;for(int i=0;i<8;++i)v|=(uint64_t)p[i]<<(8*i);return v;}
Digest128 get_digest(const uint8_t* p){return {get64(p),get64(p+8)};}
bool geometry(const SetupManifest& m){
    return m.n_layers && m.n_layers<=kMaxLayers && m.n_expert && m.n_expert<=kMaxExperts &&
        m.n_embd && m.n_embd<=kMaxNEmbd && m.n_ff && m.n_ff<=kMaxNFf && m.gu_type==8 && m.d_type==8 &&
        m.partition.size()==(size_t)m.n_layers*((m.n_expert+7)/8) &&
        m.partition_id==blob_digest(m.partition).a && m.owned_count()>0;
}
bool window_ok(CpuCaptureWindow w,uint32_t prompt,uint64_t& next,bool& done){
    if(done || !w.T || w.T>kMaxT || w.pos!=next || w.pos>=prompt || w.T>prompt-w.pos)return false;
    if(w.phase==1){if(w.pos+w.T>=prompt)return false;}
    else if(w.phase==2){if(w.pos+1!=prompt || w.T!=1)return false;done=true;}
    else return false;
    next+=w.T;return true;
}
bool bits(const std::vector<float>& a,const std::vector<float>& b){
    return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*4);
}
bool finite(const std::vector<float>& v){return std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);});}
}
void CpuCapture::u32(uint32_t v){set32(data_.get()+used_,v);used_+=4;}
void CpuCapture::u64(uint64_t v){set64(data_.get()+used_,v);used_+=8;}
void CpuCapture::bytes(const void* p,size_t n){std::memcpy(data_.get()+used_,p,n);used_+=n;}
bool CpuCapture::room(size_t n){return !error_ && used_<=capacity_ && capacity_-used_>=16 && n<=capacity_-used_-16 ? true : fail("CPU capture capacity exceeded");}
bool CpuCapture::covered() const {
    if(!windows_)return true;
    for(size_t l=0;l<setup_.n_layers;++l)if(coverage_[l]!=(1u<<current_.T)-1u)return false;
    return true;
}
bool CpuCapture::prepare(const SetupManifest& m,Digest128 identity,Digest128 index,
                         const std::string& path,std::string& err,size_t limit){
#ifdef _WIN32
    err="CPU capture publishing requires POSIX";return false;
#else
    if(prepared_ || !geometry(m) || m.model_id!=identity.a || limit>kCpuCaptureLimit ||
       limit<kCpuCaptureHeader+16 || !std::filesystem::path(path).is_absolute() ||
       !std::filesystem::is_directory(std::filesystem::path(path).parent_path()) ||
       std::filesystem::exists(path) || std::filesystem::exists(path+".partial")){
        err="invalid CPU capture geometry/path/capacity or existing artifact";return false;
    }
    std::vector<uint8_t> setup;
    if(!encode_setup_begin(m,setup)){err="CPU capture setup encoding failed";return false;}
    if(kCpuCaptureHeader+setup.size()+16>limit){err="CPU capture header exceeds capacity";return false;}
    data_.reset(new uint8_t[limit]);capacity_=limit;setup_=m;path_=path;
    payload_.reserve(kInferPayloadMax);
    std::memset(data_.get(),0,kCpuCaptureHeader);std::memcpy(data_.get(),Magic,8);
    set32(data_.get()+8,Version);set32(data_.get()+16,(uint32_t)setup.size());
    digest(data_.get()+24,identity);digest(data_.get()+40,index);
    used_=kCpuCaptureHeader;bytes(setup.data(),setup.size());prepared_=true;return true;
#endif
}
bool CpuCapture::request(const std::string& line,const std::vector<int64_t>& tokens,uint64_t max_new){
    if(sealed_)return fail("CPU capture permits one request only");
    if(!active())return true;
    if(bound_ || tokens.empty() || tokens.size()>MaxWindows || !max_new || max_new>UINT32_MAX ||
       line.rfind("GEN ",0)!=0 || line.find('=')!=std::string::npos || line.size()>16384)
        return fail("CPU capture requires one plain bounded GEN request");
    // Worst case: every token is its own callback for every layer. Refuse before any GPU window.
    const size_t bound=tokens.size()*setup_.n_layers*(2u*setup_.n_embd*4u+kMaxK*7u+128u);
    if(!room(bound))return false;
    std::vector<uint8_t> ids;ids.reserve(tokens.size()*4);
    for(auto t:tokens){if(t<0 || t>INT32_MAX)return fail("CPU capture token outside bounds");
        for(int b=0;b<4;++b)ids.push_back((uint8_t)((uint32_t)t>>(8*b)));}
    prompt_=(uint32_t)tokens.size();bound_=true;
    set32(data_.get()+60,prompt_);set32(data_.get()+64,(uint32_t)max_new);set32(data_.get()+68,1);
    digest(data_.get()+72,blob_digest(line.data(),line.size()));digest(data_.get()+88,blob_digest(ids));return true;
}
bool CpuCapture::window(uint64_t pos,uint32_t T,uint32_t phase){
    if(!active())return true;
    CpuCaptureWindow w{windows_+1,phase,T,pos};
    if(!bound_ || pending_ || !covered() || windows_==MaxWindows || !window_ok(w,prompt_,next_pos_,decoded_) || !room(28))
        return fail("CPU capture window lifecycle/position mismatch");
    current_=w;coverage_.fill(0);++windows_;u32(Window);u32(28);u32(w.ordinal);u32(phase);u64(pos);u32(T);return true;
}
bool CpuCapture::begin(const InferRequest& r){
    if(!active())return true;
    if(error_ || !bound_ || !windows_ || pending_ || records_==MaxRecords ||
       token_begin_>current_.T || r.T>current_.T-token_begin_ || r.layer>=setup_.n_layers ||
       r.model_id!=setup_.model_id || r.partition_id!=setup_.partition_id ||
       !finite(r.activations) || !finite(r.weights) || !encode_infer(r,setup_.n_embd,payload_))
        return fail("CPU capture request lifecycle/shape mismatch");
    for(size_t i=0;i<r.mask.size();++i)if(r.mask[i]!=(setup_.owns(r.layer,r.ids[i])?1:0))
        return fail("CPU capture ownership mismatch");
    const auto span=(uint16_t)(((1u<<r.T)-1u)<<token_begin_);
    if(coverage_[r.layer]&span)return fail("CPU capture duplicate token/layer slice");
    const size_t values=(size_t)r.T*setup_.n_embd;
    const size_t n=24+payload_.size()+values*4;
    if(!room(n))return false;
    bool sent=std::find(r.mask.begin(),r.mask.end(),1)!=r.mask.end();
    u32(Infer);u32((uint32_t)n);u32(current_.ordinal);u32(token_begin_);u32(sent);u32((uint32_t)payload_.size());
    bytes(payload_.data(),payload_.size());reply_values_=(uint32_t)values;
    pending_layer_=r.layer;pending_span_=span;pending_=true;pending_sent_=sent;return true;
}
bool CpuCapture::finish(const float* partials){
    if(!active())return true;
    if(error_ || !pending_ || (pending_sent_ && !partials))return fail("CPU capture finish without reply");
    for(uint32_t i=0;i<reply_values_;++i){float v=partials?partials[i]:0.f;
        if(!std::isfinite(v))return fail("CPU capture reply nonfinite");
        uint32_t u;std::memcpy(&u,&v,4);u32(u);}
    coverage_[pending_layer_]|=pending_span_;pending_=false;++records_;return true;
}
bool CpuCapture::seal(const float* logits,size_t vocab){
    if(!active())return true;
    if(error_ || pending_ || !decoded_ || !covered() || !records_ || !logits || !vocab || vocab>(1u<<20))
        return fail("CPU capture incomplete at first logits");
    for(size_t i=0;i<vocab;++i)if(!std::isfinite(logits[i]))return fail("CPU capture logits nonfinite");
    set32(data_.get()+12,(uint32_t)(used_+16));set32(data_.get()+20,records_);set32(data_.get()+56,windows_);
    digest(data_.get()+104,blob_digest(logits,vocab*4));set32(data_.get()+120,(uint32_t)vocab);
    set64(data_.get()+128,current_.pos);set32(data_.get()+136,current_.T);
    const auto d=blob_digest(data_.get(),used_);u64(d.a);u64(d.b);
#ifndef _WIN32
    const auto temp=path_+".partial";
    int fd=::open(temp.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd<0)return fail("CPU capture exclusive file create failed");
    size_t at=0;bool ok=true;
    while(at<used_){const auto n=::write(fd,data_.get()+at,used_-at);
        if(n<0 && errno==EINTR)continue;if(n<=0){ok=false;break;}at+=(size_t)n;}
    if(::close(fd))ok=false;
    if(ok && ::link(temp.c_str(),path_.c_str()))ok=false; // exclusive atomic publication, no overwrite
    ::unlink(temp.c_str());
    if(!ok)return fail("CPU capture file write/publication failed");
    sealed_=true;return true;
#else
    return fail("CPU capture publishing requires POSIX");
#endif
}
bool decode_cpu_capture(const uint8_t* p,size_t n,CpuCaptureFile& out,std::string& err){
    auto refuse=[&](const char* s){err=s;return false;};CpuCaptureFile f;
    if(n<kCpuCaptureHeader+16 || n>kCpuCaptureLimit)return refuse("capture file size outside bounds");
    if(std::memcmp(p,Magic,8) || get32(p+8)!=Version || get32(p+12)!=n || get32(p+68)!=1 ||
       get32(p+124)!=0 || get32(p+140)!=0)return refuse("capture header invalid");
    if(blob_digest(p,n-16)!=get_digest(p+n-16))return refuse("capture integrity digest mismatch");
    const auto setup_bytes=get32(p+16),records=get32(p+20),windows=get32(p+56);
    if(!records || records>MaxRecords || !windows || windows>MaxWindows ||
       setup_bytes>kSetupFrameMax || setup_bytes>n-kCpuCaptureHeader-16 ||
       !decode_setup_begin(p+kCpuCaptureHeader,setup_bytes,f.setup) || !geometry(f.setup))
        return refuse("capture setup/count invalid");
    f.identity=get_digest(p+24);f.index=get_digest(p+40);f.prompt_tokens=get32(p+60);f.max_new=get32(p+64);
    f.line=get_digest(p+72);f.tokens=get_digest(p+88);f.logits=get_digest(p+104);f.vocab=get32(p+120);
    if(f.setup.model_id!=f.identity.a || !f.prompt_tokens || f.prompt_tokens>MaxWindows || !f.max_new ||
       !f.vocab || f.vocab>(1u<<20) || get64(p+128)!=f.prompt_tokens-1u || get32(p+136)!=1)
        return refuse("capture request/logit metadata invalid");
    size_t at=kCpuCaptureHeader+setup_bytes;uint64_t next=0;bool done=false;
    std::vector<uint16_t> covered(f.setup.n_layers,0);
    auto complete_window=[&]{return f.windows.empty() || std::all_of(covered.begin(),covered.end(),
        [&](uint16_t span){return span==(1u<<f.windows.back().T)-1u;});};
    while(at<n-16){
        if(n-16-at<8)return refuse("capture event truncated");
        const auto type=get32(p+at),bytes=get32(p+at+4);const auto* e=p+at;
        if(bytes<8 || bytes>n-16-at)return refuse("capture event length invalid");
        if(type==Window){
            if(bytes!=28)return refuse("capture window length invalid");
            CpuCaptureWindow w{get32(e+8),get32(e+12),get32(e+24),get64(e+16)};
            if(!complete_window() || w.ordinal!=f.windows.size()+1 || f.windows.size()>=windows ||
               !window_ok(w,f.prompt_tokens,next,done))return refuse("capture window sequence invalid");
            f.windows.push_back(w);std::fill(covered.begin(),covered.end(),0);
        } else if(type==Infer){
            if(bytes<24 || f.windows.empty() || f.records.size()>=records)return refuse("capture infer without window");
            CpuCaptureRecord r;r.window=get32(e+8);r.token_begin=get32(e+12);
            const auto sent=get32(e+16),payload=get32(e+20);
            if(sent>1 || payload>kInferPayloadMax || payload>bytes-24 ||
               !decode_infer(e+24,payload,f.setup.n_layers,f.setup.n_expert,f.setup.n_embd,r.request))
                return refuse("capture infer payload invalid");
            r.sent=sent;const auto& q=r.request;const auto& w=f.windows.back();
            if(r.window!=w.ordinal || r.token_begin>w.T || q.T>w.T-r.token_begin ||
               bytes!=24+payload+(size_t)q.T*f.setup.n_embd*4 ||
               q.model_id!=f.setup.model_id || q.partition_id!=f.setup.partition_id ||
               !finite(q.activations) || !finite(q.weights))return refuse("capture infer identity/shape/nonfinite");
            bool any=false;
            for(size_t i=0;i<q.mask.size();++i){
                if(q.mask[i]>1 || q.mask[i]!=(f.setup.owns(q.layer,q.ids[i])?1:0))
                    return refuse("capture CPU ownership mismatch");
                any=any || q.mask[i];}
            if(any!=r.sent)return refuse("capture sent flag mismatch");
            const uint16_t span=(uint16_t)(((1u<<q.T)-1u)<<r.token_begin);
            if(covered[q.layer]&span)return refuse("capture duplicate token/layer slice");
            covered[q.layer]|=span;
            r.partials.resize((size_t)q.T*f.setup.n_embd);
            for(size_t i=0;i<r.partials.size();++i){const auto u=get32(e+24+payload+4*i);std::memcpy(&r.partials[i],&u,4);}
            if(!finite(r.partials) || (!r.sent && std::any_of(r.partials.begin(),r.partials.end(),[](float x){return x!=0;})))
                return refuse("capture partial invalid");
            f.records.push_back(std::move(r));
        } else return refuse("capture unknown event");
        at+=bytes;
    }
    if(at!=n-16 || !done || !complete_window() || f.records.size()!=records || f.windows.size()!=windows)
        return refuse("capture incomplete");
    out=std::move(f);return true;
}
bool read_cpu_capture(const std::string& path,CpuCaptureFile& out,std::string& err){
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    const auto size=in.tellg();
    if(!in || size<0 || (uint64_t)size>kCpuCaptureLimit){err="capture unreadable/oversize";return false;}
    std::vector<uint8_t> bytes((size_t)size);in.seekg(0);
    if(!in.read((char*)bytes.data(),bytes.size())){err="capture short read";return false;}
    return decode_cpu_capture(bytes.data(),bytes.size(),out,err);
}
std::vector<int8_t> cpu_capture_owners(const SetupManifest& m){
    std::vector<int8_t> owners((size_t)m.n_layers*m.n_expert,-1);size_t ordinal=0;
    for(int l=0;l<m.n_layers;++l)for(int e=0;e<m.n_expert;++e)if(m.owns(l,e))
        owners[(size_t)l*m.n_expert+e]=(int8_t)(ordinal++%2);
    return owners;
}
bool replay_cpu_capture(const CpuCaptureFile& f,const CpuReplayExchange& exchange,CpuReplayReport& report,std::string& err){
    report={};const auto owners=cpu_capture_owners(f.setup);
    // Full validation pass first. No peer-masked request is issued before EVERY full reply matches.
    for(size_t i=0;i<f.records.size();++i){
        const auto& r=f.records[i];std::vector<float> got(r.partials.size(),0.f);
        if(r.sent && !exchange(r.request,got,err))return false;
        if(!finite(got) || !bits(got,r.partials)){err="full replay mismatch at record "+std::to_string(i);return false;}
        ++report.full_checked;
    }
    double delta2=0,norm2=0;
    for(size_t i=0;i<f.records.size();++i){
        const auto& r=f.records[i];std::vector<float> peer[2];bool active[2]{};
        for(int p=0;p<2;++p){
            auto q=r.request;
            for(size_t j=0;j<q.mask.size();++j)q.mask[j]=q.mask[j] && owners[(size_t)q.layer*f.setup.n_expert+q.ids[j]]==p;
            active[p]=std::find(q.mask.begin(),q.mask.end(),1)!=q.mask.end();peer[p].assign(r.partials.size(),0.f);
            if(active[p] && !exchange(q,peer[p],err))return false;
            if(peer[p].size()!=r.partials.size() || !finite(peer[p])){err="masked replay shape/nonfinite";return false;}
        }
        auto grouped=active[0]?peer[0]:peer[1];
        if(active[0] && active[1])for(size_t j=0;j<grouped.size();++j)grouped[j]=grouped[j]+peer[1][j];
        if(!finite(grouped)){err="grouped replay nonfinite";return false;}
        double record_max=0;
        for(size_t j=0;j<grouped.size();++j){
            const double d=std::abs((double)grouped[j]-r.partials[j]);
            record_max=std::max(record_max,d);
            report.max_abs=std::max(report.max_abs,d);
            report.max_rel=std::max(report.max_rel,d/std::max(1e-12,std::abs((double)r.partials[j])));
            delta2+=d*d;norm2+=(double)r.partials[j]*r.partials[j];
        }
        if(!bits(grouped,r.partials) && report.first_difference==UINT32_MAX)report.first_difference=(uint32_t)i;
        if(record_max>report.detail_max_abs){report.detail_max_abs=record_max;report.detail_record=(uint32_t)i;}
        ++report.grouped_checked;
    }
    // One bounded decomposition: choose the largest absolute discrepancy, rather than letting a
    // harmless early rounding bit consume the diagnostic before a later larger discrepancy.
    if(report.detail_record!=UINT32_MAX){
            const auto& r=f.records[report.detail_record];std::vector<float> peer[2];bool active[2]{};
            for(int p=0;p<2;++p){auto q=r.request;
                for(size_t j=0;j<q.mask.size();++j)q.mask[j]=q.mask[j] && owners[(size_t)q.layer*f.setup.n_expert+q.ids[j]]==p;
                peer[p].assign(r.partials.size(),0.f);
                active[p]=std::find(q.mask.begin(),q.mask.end(),1)!=q.mask.end();
                if(active[p] && !exchange(q,peer[p],err))return false;
                if(peer[p].size()!=r.partials.size() || !finite(peer[p])){err="detail peer invalid";return false;}
            }
            auto grouped=active[0]?peer[0]:peer[1];
            if(active[0] && active[1])for(size_t j=0;j<grouped.size();++j)grouped[j]+=peer[1][j];
            std::vector<float> entry_sum(grouped.size(),0.f),ordered_peer[2];
            ordered_peer[0]=ordered_peer[1]=entry_sum;std::vector<double> magnitude(grouped.size(),0);
            for(size_t j=0;j<r.request.mask.size();++j)if(r.request.mask[j]){
                auto q=r.request;std::fill(q.mask.begin(),q.mask.end(),0);q.mask[j]=1;std::vector<float> part;
                if(!exchange(q,part,err))return false;++report.contribution_calls;
                if(part.size()!=grouped.size() || !finite(part)){err="entry replay shape/nonfinite";return false;}
                const auto owner=owners[(size_t)q.layer*f.setup.n_expert+q.ids[j]];
                for(size_t k=0;k<part.size();++k){entry_sum[k]+=part[k];ordered_peer[owner][k]+=part[k];magnitude[k]+=std::abs((double)part[k]);}
            }
            report.entry_order_bits=bits(entry_sum,r.partials);
            report.peer_bits=bits(ordered_peer[0],peer[0]) && bits(ordered_peer[1],peer[1]);
            for(size_t j=0;j<grouped.size();++j){
                const double d=std::abs((double)grouped[j]-r.partials[j]);report.detail_max_abs=std::max(report.detail_max_abs,d);
                report.detail_envelope=report.detail_envelope && d<=1e-4+4e-6*magnitude[j];
            }
    }
    report.relative_l2=std::sqrt(delta2/std::max(1e-30,norm2));return true;
}
}
