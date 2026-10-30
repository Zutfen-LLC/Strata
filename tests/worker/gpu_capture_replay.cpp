// Bounded offline discriminator, not a benchmark or a model acceptance test.
#include "local_fixture.hpp"
#include "strata/worker/cpu_capture.hpp"
#include "strata/worker/gpu_group.hpp"
#include "strata/worker/gpu_worker.hpp"
#include "mixed_rows.hpp"
#include "cpu_capture_rows.hpp"
#include "ggml-cpu.h"
#include "../../src/core/native_q8_gu_layout.hpp"
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#ifdef STRATA_CAPTURE_CUDA
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#endif

namespace cpu = strata::kernels::cpu;
namespace {
struct Selection { size_t record, entry, plan; GpuAsset asset; };
Selection select(const CpuCaptureFile& f,const std::vector<GpuPlan>& plans) {
    require(f.setup.gu_type==8 && f.setup.d_type==8 && f.setup.n_embd==2560 && f.setup.n_ff==640,
            "requires plain Q8_0 2560x640 capture");
    for(const auto& p:plans) {
        std::string err;require(validate_gpu_plan(p,err),err);
        require(p.model_id==f.setup.model_id && p.n_layers==f.setup.n_layers && p.n_expert==f.setup.n_expert &&
                p.n_embd==f.setup.n_embd && p.n_ff==f.setup.n_ff,"capture/assignment model geometry mismatch");
    }
    for(size_t r=0;r<f.records.size();++r) {
        const auto& q=f.records[r].request;
        if(!f.records[r].sent)continue;
        require(q.T>=1 && q.T<=2 && q.K>=1 && q.K<=10 && q.mask.size()==size_t(q.T*q.K) &&
                q.ids.size()==q.mask.size() && q.weights.size()==q.mask.size() &&
                q.activations.size()==size_t(q.T)*2560,"capture record bound");
        for(size_t i=0;i<q.mask.size();++i)if(q.mask[i])for(size_t p=0;p<plans.size();++p)
            if(const auto* a=plans[p].find(q.layer,q.ids[i]))return {r,i,p,*a};
    }
    throw std::runtime_error("no active captured entry intersects supplied B assignments");
}
std::string hex(uint64_t v,int width=16) {std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(width)<<v;return s.str();}
std::string key(const GpuPlan& p,const GpuAsset& a) {
    return "sgpu1-"+hex(p.model_id)+"-"+hex(p.layout_id.a)+hex(p.layout_id.b)+"-"+
        hex(a.layer,4)+"-"+hex(a.expert,4)+"-"+hex(a.bytes)+"-"+hex(a.digest.a)+hex(a.digest.b)+".blob";
}
struct SurveyAsset {size_t plan;GpuAsset asset;std::vector<size_t> records;};
struct Inventory {
    std::vector<SurveyAsset> assets;
    std::vector<size_t> records;
    std::vector<std::vector<uint8_t>> masks;
    size_t active=0,selected=0,missing=0;uint64_t bytes=0;
};
Inventory inventory(const CpuCaptureFile& f,const std::vector<GpuPlan>& plans,const fs::path& blobs,bool print,const std::vector<size_t>& chosen={}) {
    (void)select(f,plans); // existing model/geometry/plan checks; also require a real intersection
    require(plans.size()<=4,"survey limited to four B plans");
    std::map<std::pair<int,int>,size_t> owners;
    for(size_t p=0;p<plans.size();++p) {
        require(plans[p].layout_id==plans[0].layout_id,"B plan layout identities differ");
        for(const auto& a:plans[p].assets)require(owners.emplace(std::make_pair(a.layer,a.expert),p).second,"overlapping B assignments");
    }
    Inventory result;result.masks.resize(f.records.size());
    result.records=chosen.empty()?diagnostic::capture_row_samples(f,"SURVEY_SAMPLE"):chosen;
    require(result.records.size()<=16,"record selection bound");
    for(size_t r:result.records)require(r<f.records.size(),"selected record absent");
    std::map<std::pair<int,int>,size_t> selected;
    for(size_t r:result.records) {
        const auto& record=f.records[r];const auto& q=record.request;
        require(q.T>=1 && q.T<=2 && q.K>=1 && q.K<=10 && q.mask.size()==size_t(q.T*q.K) &&
            q.ids.size()==q.mask.size() && q.weights.size()==q.mask.size() && q.activations.size()==size_t(q.T)*2560,"survey record bounds");
        auto& mask=result.masks[r];mask.assign(q.mask.size(),0);
        if(!record.sent)continue;
        for(size_t i=0;i<q.mask.size();++i)if(q.mask[i]) {
            ++result.active;const auto id=std::make_pair(int(q.layer),int(q.ids[i]));const auto owner=owners.find(id);
            if(owner==owners.end())continue;
            const auto p=owner->second;const auto& a=*plans[p].find(q.layer,q.ids[i]);
            auto it=selected.find(id);
            if(it==selected.end()) {
                const size_t index=result.assets.size();selected.emplace(id,index);result.assets.push_back({p,a,{}});it=selected.find(id);
                result.bytes+=a.bytes;const auto file=blobs/key(plans[p],a);
                const bool present=fs::is_regular_file(file) && fs::file_size(file)==a.bytes;result.missing+=!present;
                if(print)std::printf("ASSET index=%zu plan=%zu layer=%u expert=%u bytes=%llu available=%d cache_key=%s\n",
                    index,p,a.layer,a.expert,(unsigned long long)a.bytes,present,key(plans[p],a).c_str());
                if(print && !present)std::printf("NEEDED plan=%zu cache_key=%s bytes=%llu\n",p,key(plans[p],a).c_str(),(unsigned long long)a.bytes);
            }
            auto& records=result.assets[it->second].records;
            if(records.empty() || records.back()!=r)records.push_back(r);
            mask[i]=1;++result.selected;
            if(print)std::printf("ENTRY record=%zu entry=%zu layer=%u token=%zu expert=%u plan=%zu asset=%zu\n",r,i,q.layer,i/q.K,a.expert,p,it->second);
        }
        if(print)std::printf("RECORD record=%zu selected=%zu excluded=%zu\n",r,std::count(mask.begin(),mask.end(),1),
            std::count(q.mask.begin(),q.mask.end(),1)-std::count(mask.begin(),mask.end(),1));
    }
    require(!result.assets.empty() && result.active<=225 && result.selected<=225 && result.assets.size()<=225,"survey missing sample intersection or exceeds retained 225-entry bound");
    if(print)std::printf("INVENTORY records=%zu capture_records=%zu active_entries=%zu selected_entries=%zu unique_blobs=%zu total_blob_bytes=%llu missing_blobs=%zu "
        "weight_residency_experts=1 max_extra_private_cache_bytes=5222400 deadline_seconds=300\n",
        result.records.size(),f.records.size(),result.active,result.selected,result.assets.size(),(unsigned long long)result.bytes,result.missing);
    return result;
}
struct StageSummary {
    size_t observations=0,different=0,scales=0,codes=0;double maxabs=0,maxrel=0,maxscale=0;int maxcode=0;
    std::pair<size_t,size_t> earliest_order{SIZE_MAX,SIZE_MAX};std::string earliest,max_at,maxrel_at,maxscale_at,maxcode_at;
};
std::map<std::string,StageSummary> summaries;
bool surveying=false;std::string context;std::pair<size_t,size_t> context_order;
// Restrict the existing validated plan codec to one asset, retaining model/layout identity.
GpuPlan subset_plan(const GpuPlan& p,const std::vector<GpuAsset>& assets) {
    std::vector<uint8_t> wire;std::string err;require(encode_gpu_plan(p,wire,err),err);
    require(!assets.empty() && assets.size()<=20,"bounded nonempty expert subset");
    std::vector<uint8_t> one(wire.begin(),wire.begin()+48);
    one[44]=assets.size();one[45]=one[46]=one[47]=0;
    for(const auto& a:assets) {
        const auto* found=p.find(a.layer,a.expert);require(found && found->digest==a.digest,"subset asset binding");
        const size_t at=48+28*(found-p.assets.data());one.insert(one.end(),wire.begin()+at,wire.begin()+at+28);
    }
    const auto d=blob_digest(one);put<uint64_t>(one,d.a);put<uint64_t>(one,d.b);
    GpuPlan out;require(decode_gpu_plan(one.data(),one.size(),out,err),err);return out;
}
GpuPlan one_asset(const GpuPlan& p,const GpuAsset& a){return subset_plan(p,{a});}
std::vector<uint8_t> read_blob(const std::string& path,const GpuAsset& a) {
    require(a.bytes==5222400,"single Q8_0 expert size bound");
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    require(in && in.tellg()==std::streamoff(a.bytes),"selected blob size mismatch");
    std::vector<uint8_t> b(a.bytes);in.seekg(0);require(bool(in.read((char*)b.data(),b.size())),"selected blob short read");
    require(blob_digest(b)==a.digest,"selected assignment/blob digest mismatch");return b;
}
void stage_cache(GpuCache& cache,const fs::path& root,const GpuPlan& p,const std::vector<uint8_t>& b) {
    cache.close();std::string err;uint64_t removed=0;std::vector<uint8_t> missing;
    require(cache.open(root.string(),16ull<<20,err)&&cache.cleanup_unassigned(p,removed,err)&&cache.prepare(p,missing,err),err);
    require(missing.size()==1,"private diagnostic cache requires exactly one asset");
    if(missing[0])require(cache.publish(p.assets[0],b.data(),b.size(),err),err);
}
bool metrics(const char* label,const std::vector<float>& a,const std::vector<float>& b) {
    require(a.size()==b.size(),"metric shape");double sum=0,den=0,maxabs=0;size_t first=a.size(),max_index=0;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite diagnostic result");
        const double d=double(a[i])-b[i];sum+=d*d;den+=double(a[i])*a[i];
        if(std::fabs(d)>maxabs){maxabs=std::fabs(d);max_index=i;}
        if(first==a.size() && std::memcmp(&a[i],&b[i],4))first=i;
    }
    std::printf("COMPARE %s bits=%d first=%zu maxabs=%.9g relL2=%.9g\n",label,first==a.size(),first,maxabs,std::sqrt(sum/(den+1e-300)));
    if(surveying) {
        auto& s=summaries[label];++s.observations;const bool differs=first!=a.size();s.different+=differs;
        if(differs && context_order<s.earliest_order){s.earliest_order=context_order;s.earliest=context+" first_float="+std::to_string(first);}
        if(maxabs>s.maxabs){s.maxabs=maxabs;s.max_at=context+" max_float="+std::to_string(max_index);}
        const double relative=std::sqrt(sum/(den+1e-300));
        if(relative>s.maxrel){s.maxrel=relative;s.maxrel_at=context;}
    }
    return first==a.size();
}
#include "first_boundary_replay.hpp"
struct CpuRows {std::vector<float> rows,gate,up,hidden;std::vector<uint8_t> act,hq;};
CpuRows cpu_rows(const InferRequest& q,const std::vector<uint8_t>& b,const cpu::NativeFmt& f) {
    CpuRows c;c.act.resize(q.T*cpu::kNativeActBytes);c.rows.resize(q.T*f.n_embd);
    c.hidden.resize(q.T*f.n_ff);c.gate.resize(c.hidden.size());c.up.resize(c.hidden.size());c.hq.resize(q.T*f.h_bytes);
    const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
    for(int t=0;t<q.T;++t) {
        void* a=c.act.data()+t*cpu::kNativeActBytes;cpu::native_quant_act(f,q.activations.data()+t*f.n_embd,a);
        const void* av=a;float* h=c.hidden.data()+t*f.n_ff;
        cpu::native_gu_rows(f,b.data(),&av,1,&h,0,f.n_ff);
        for(int j=0;j<f.n_ff;++j) {
            dot(f.n_embd,&c.gate[t*f.n_ff+j],0,b.data()+j*f.gu_row,0,av,0,1);
            dot(f.n_embd,&c.up[t*f.n_ff+j],0,b.data()+f.up_off+j*f.gu_row,0,av,0,1);
        }
        void* hq=c.hq.data()+t*f.h_bytes;cpu::native_quant_h(f,h,hq);
        const void* hv=hq;float* out=c.rows.data()+t*f.n_embd;cpu::native_down_rows(f,b.data(),&hv,1,&out,0,f.n_embd);
    }
    return c;
}
void cpu_info() {
    ggml_cpu_init();std::printf("CPU_BUILD compiler=%s ggml_avx=%d ggml_avx2=%d ggml_avx512=%d ggml_fma=%d ggml_f16c=%d\n",
        __VERSION__,ggml_cpu_has_avx(),ggml_cpu_has_avx2(),ggml_cpu_has_avx512(),ggml_cpu_has_fma(),ggml_cpu_has_f16c());
    for(const char* name:{"STRATA_FORCE_ISA","STRATA_FORCE_AVX2","STRATA_CPU_IQ_ALL_EXACT1"})
        std::printf("ENV %s=%s\n",name,std::getenv(name)?std::getenv(name):"<unset>");
}
void export_cpu_row(const fs::path& prefix,const CpuRows& c,const cpu::NativeFmt& fmt,size_t t) {
    auto write=[&](const char* suffix,const void* data,size_t bytes){std::ofstream out(prefix.string()+suffix,std::ios::binary);
        require(bool(out.write((const char*)data,bytes)),"CPU control output failed");};
    write(".input.q8_0",c.act.data()+t*cpu::kNativeActBytes,fmt.act_bytes);
    write(".gate.f32",c.gate.data()+t*640,640*4);write(".up.f32",c.up.data()+t*640,640*4);
    write(".hidden.f32",c.hidden.data()+t*640,640*4);write(".hidden.q8_0",c.hq.data()+t*fmt.h_bytes,fmt.h_bytes);
    write(".row.f32",c.rows.data()+t*2560,2560*4);
}
void cpu_control(const CpuCaptureFile& f,const std::vector<GpuPlan>& plans,const fs::path& blobs,const fs::path& root) {
    (void)select(f,plans);require(f.records.size()>828,"CPU control requires original record IDs 0 and 828");
    require(fs::create_directory(root),"CPU control output must be a new directory");cpu_info();
    cpu::NativeFmt fmt;std::string err;require(cpu::native_fmt(8,8,2560,640,fmt,err),err);
    for(const auto [r,i]:{std::pair<size_t,size_t>{0,2},{828,13}}) {
        const auto& q=f.records[r].request;require(q.T<=2 && q.K==10 && i<q.mask.size() && q.mask[i] &&
            q.ids[i]==(r==0?77:390) && q.layer==(r==0?0:12),"CPU control retained entry binding mismatch");
        const GpuAsset* asset=nullptr;const GpuPlan* plan=nullptr;
        for(const auto& p:plans)if(const auto* a=p.find(q.layer,q.ids[i])){require(!asset,"overlapping control assignment");asset=a;plan=&p;}
        require(asset,"CPU control entry absent from actual B plans");const auto b=read_blob((blobs/key(*plan,*asset)).string(),*asset);
        const auto c=cpu_rows(q,b,fmt);const size_t t=i/q.K;
        const auto prefix=root/("record-"+std::to_string(r)+"-entry-"+std::to_string(i));
        export_cpu_row(prefix,c,fmt,t);
        std::printf("CPU_CONTROL record=%zu entry=%zu layer=%u token=%zu expert=%u asset_digest=%s%s input_digest=%s%s bytes=21320\n",
            r,i,q.layer,t,q.ids[i],hex(asset->digest.a).c_str(),hex(asset->digest.b).c_str(),
            hex(blob_digest(q.activations.data()+t*2560,2560*4).a).c_str(),hex(blob_digest(q.activations.data()+t*2560,2560*4).b).c_str());
    }
}
void trace_check(const InferRequest& q,const GpuPlan& p,const GpuExecutionTrace& tr) {
    auto print=[](const char* name,const auto& values){std::printf("META %s=",name);for(auto v:values)std::printf("%d,",int(v));std::puts("");};
    print("experts",tr.experts);print("asset_slots",tr.asset_slots);print("grp_start",tr.start);
    print("ent_dst",tr.dst);print("ent_tok",tr.tok);print("original_entries",tr.original);
    std::printf("META cap_groups=%zu cap_entries=%zu grid_groups=0\n",tr.experts.size(),tr.dst.size());
    require(tr.start.size()==tr.experts.size()+1 && !tr.start.empty() && tr.start.front()==0 &&
        tr.asset_slots.size()==tr.experts.size() && tr.dst.size()==tr.tok.size() && tr.dst.size()==tr.original.size() &&
        tr.start.back()==int(tr.dst.size()),"executor diagnostic metadata shape");
    std::vector<uint8_t> seen(q.mask.size());
    for(size_t g=0;g<tr.experts.size();++g) {
        const auto* asset=p.find(q.layer,tr.experts[g]);require(asset && tr.asset_slots[g]==asset-p.assets.data(),"executor asset slot identity");
        require(tr.start[g]>=0 && tr.start[g]<tr.start[g+1],"executor group start bounds");
        for(int j=tr.start[g];j<tr.start[g+1];++j) {
            const int i=tr.original[j];require(i>=0 && size_t(i)<seen.size() && q.mask[i] && !seen[i] &&
                q.ids[i]==tr.experts[g] && tr.tok[j]==i/q.K && tr.dst[j]==j,"executor actual group/index mapping");seen[i]=1;
        }
    }
    require(seen==q.mask,"executor trace entry coverage");

}
// Freeze one row matrix to isolate the REAL mixed reducer. This fixture computes no experts.
struct Frozen final:GpuExecutor {
    std::vector<float> rows;
    explicit Frozen(std::vector<float> r):rows(std::move(r)){}
    bool prepare(const GpuPlan& p,uint64_t,GpuCache& cache,GpuCounters& c,std::string& err)override {
        std::vector<uint8_t> b;for(const auto& a:p.assets)if(!cache.read(a,b,err))return false;
        c.gpu_weight_bytes=p.assigned_bytes();c.gpu_allocated_bytes=p.assigned_bytes();return true;
    }
    bool supports_rows()const override{return true;}
    bool infer(const GpuPlan&,const InferRequest&,InferReply&,std::string&)override{return false;}
    bool infer_rows(const GpuPlan& p,const InferRequest& q,EntryRowsReply& out,std::string&)override {
        for(size_t i=0;i<q.mask.size();++i)if(q.mask[i]) {
            out.indices.push_back(i);out.rows.insert(out.rows.end(),rows.begin()+i*p.n_embd,rows.begin()+(i+1)*p.n_embd);
        }return true;
    }
    void release(GpuCounters& c)override{c.gpu_allocated_bytes=c.gpu_weight_bytes=0;}
};
void fixed_reduce(const GpuPlan& p,const InferRequest& original,const std::vector<uint8_t>& blob,
                  const std::vector<float>& rows,const fs::path& root) {
    GpuWorkerOptions o;o.plan=p;o.cache_root=root.string();o.cache_budget=16ull<<20;o.deadline_ms=3000;
    GpuWorkerService service(o,std::make_unique<Frozen>(rows));uint16_t port=0;std::string err;
    require(service.listen(port,err),err);
    std::thread server([&]{service.run();});
    struct Join {GpuWorkerService& s;std::thread& t;~Join(){s.stop();if(t.joinable())t.join();}} join{service,server};
    GpuWorkerTier::Open open;open.plan=p;open.port=port;open.entry_rows=true;open.deadline_ms=3000;open.setup_ms=10000;
    open.fetch=[&](int64_t l,int64_t e,uint8_t* out,uint32_t n){
        if(!p.owns(l,e)||n!=blob.size())return false;
        std::memcpy(out,blob.data(),n);return true;};
    GpuWorkerGroup group;require(group.open({open},err),err);
    auto gpu=original.mask,cpu_mask=original.mask;
    std::vector<int32_t> ids(original.ids.begin(),original.ids.end());
    for(size_t i=0;i<gpu.size();++i) {
        gpu[i]=original.mask[i] && p.owns(original.layer,ids[i]);cpu_mask[i]=original.mask[i] && !gpu[i];
    }
    require(group.post(original.layer,original.activations.data(),ids.data(),original.weights.data(),gpu.data(),original.T,original.K,err)&&group.finish(err),err);
    std::vector<float> ref(original.T*p.n_embd),mixed;
    ExpertWorkerService::accumulate_partial(original.T,original.K,original.mask.data(),original.weights.data(),rows.data(),p.n_embd,ref.data());
    require(group.reduce_rows(rows,cpu_mask,original.mask,original.weights.data(),original.T,original.K,mixed,err),err);
    require(metrics("fixed_rows_mixed_vs_unsplit",ref,mixed),"identical fixed rows regrouped or misindexed");
    auto overlap=cpu_mask;const size_t first=std::find(gpu.begin(),gpu.end(),1)-gpu.begin();
    require(first<gpu.size(),"fixed reducer requires a GPU row");overlap[first]=1;
    require(!group.reduce_rows(rows,overlap,original.mask,original.weights.data(),original.T,original.K,mixed,err),"mixed overlap not refused");
    group.close();
}
void record_reduce(const InferRequest& q,const std::vector<uint8_t>& expected,const std::vector<GpuPlan>& plans,
                   const std::vector<float>& cpu_rows,const std::vector<float>& gpu_rows) {
    std::vector<float> cpu_sum(q.T*2560),gpu_sum(cpu_sum.size()),fixed(cpu_rows),reference(cpu_sum.size()),mixed;
    ExpertWorkerService::accumulate_partial(q.T,q.K,expected.data(),q.weights.data(),cpu_rows.data(),2560,cpu_sum.data());
    ExpertWorkerService::accumulate_partial(q.T,q.K,expected.data(),q.weights.data(),gpu_rows.data(),2560,gpu_sum.data());
    metrics("record_CPU_vs_CUDA_weighted_B_subset",cpu_sum,gpu_sum);
    for(bool hybrid:{false,true}) {
        std::vector<EntryRowsReply> replies(plans.size());std::vector<std::vector<uint8_t>> masks(plans.size(),std::vector<uint8_t>(expected.size()));
        auto cpu_mask=expected;fixed=cpu_rows;
        for(size_t i=0;i<expected.size();++i)if(expected[i] && (!hybrid || ((q.ids[i]+q.layer)&1))) {
            cpu_mask[i]=0;std::copy_n(gpu_rows.data()+i*2560,2560,fixed.data()+i*2560);
            for(size_t p=0;p<plans.size();++p)if(plans[p].owns(q.layer,q.ids[i])) {
                masks[p][i]=1;replies[p].indices.push_back(i);
                replies[p].rows.insert(replies[p].rows.end(),gpu_rows.begin()+i*2560,gpu_rows.begin()+(i+1)*2560);
            }
        }
        std::vector<MixedRowsPeer> peers;for(size_t p=0;p<plans.size();++p)peers.push_back({replies[p],masks[p]});
        std::string err;ExpertWorkerService::accumulate_partial(q.T,q.K,expected.data(),q.weights.data(),fixed.data(),2560,reference.data());
        require(reduce_mixed_rows(2560,q.T,q.K,cpu_rows,cpu_mask,expected,q.weights.data(),peers,mixed,err),err);
        require(metrics(hybrid?"record_real_CPU_CUDA_rows_mixed_vs_unsplit":"record_all_CUDA_rows_mixed_vs_unsplit",reference,mixed),"real fixed row reducer mismatch");
    }
}
#ifdef STRATA_CAPTURE_CUDA
void cu(cudaError_t s){require(s==cudaSuccess,cudaGetErrorString(s));}
void cuda_info() {
    int device=0,runtime=0,driver=0;cudaDeviceProp p{};cu(cudaGetDevice(&device));cu(cudaGetDeviceProperties(&p,device));
    cu(cudaRuntimeGetVersion(&runtime));cu(cudaDriverGetVersion(&driver));
    std::printf("CUDA_BUILD device=%d name=%s cc=%d.%d runtime=%d driver=%d compiled_cudart=%d compiler=%s\n",
        device,p.name,p.major,p.minor,runtime,driver,CUDART_VERSION,__VERSION__);
    for(const char* name:{"CUDA_VISIBLE_DEVICES","STRATA_KQ_SPLIT","STRATA_GROUPED_V1","STRATA_OLD_IQ_MMVQ","STRATA_NO_SUB16_GU",
        "STRATA_GU_SWIGLU","STRATA_EXP_MODE","STRATA_SM75_EXP","STRATA_SM75_PERF","STRATA_SM60_TABLE","STRATA_SM61_TABLE",
        "STRATA_SM70_TABLE","STRATA_EXPERT_V2","STRATA_EXPERT_V2K"})
        std::printf("ENV %s=%s\n",name,std::getenv(name)?std::getenv(name):"<unset>");
}
struct Device {void* p=nullptr;explicit Device(size_t n){cu(cudaMalloc(&p,n));}~Device(){if(p)cudaFree(p);}operator void*(){return p;}};
void packed_compare(const char* tag,const uint8_t* a,const uint8_t* b,int blocks) {
    size_t scales=0,codes=0;int first=-1,maxcode=0,maxcode_index=-1,maxscale_block=-1;double maxscale=0;
    for(int k=0;k<blocks;++k) {
        const bool scale=std::memcmp(a+34*k,b+36*k,2)!=0;scales+=scale;
        uint16_t as,bs;std::memcpy(&as,a+34*k,2);std::memcpy(&bs,b+36*k,2);
        const double scale_delta=std::fabs(double(ggml_fp16_to_fp32(as))-ggml_fp16_to_fp32(bs));
        if(scale_delta>maxscale){maxscale=scale_delta;maxscale_block=k;}
        for(int j=0;j<32;++j)if(a[34*k+2+j]!=b[36*k+4+j]) {
            ++codes;if(first<0)first=32*k+j;
            const int delta=std::abs(int((int8_t)a[34*k+2+j])-int((int8_t)b[36*k+4+j]));
            if(delta>maxcode){maxcode=delta;maxcode_index=32*k+j;}
        }
    }
    std::printf("PACKED %s blocks=%d scale_mismatches=%zu code_mismatches=%zu first_code=%d max_scale_delta=%.9g max_code_delta=%d unused_sum_ignored=1 actual_cuda=1\n",tag,blocks,scales,codes,first,maxscale,maxcode);
    if(surveying) {
        auto& s=summaries[std::string(tag).rfind("input_token_",0)==0?"input_quant":"hidden_quant"];
        ++s.observations;s.scales+=scales;s.codes+=codes;s.different+=bool(scales||codes);
        if(maxscale>s.maxscale){s.maxscale=maxscale;s.maxscale_at=context+" "+tag+" block="+std::to_string(maxscale_block);}
        if(maxcode>s.maxcode){s.maxcode=maxcode;s.maxcode_at=context+" "+tag+" code="+std::to_string(maxcode_index);}
        if((scales||codes) && context_order<s.earliest_order){s.earliest_order=context_order;s.earliest=context+" "+tag+" first_code="+std::to_string(first);}
    }
}
std::vector<float> cuda_rows(const InferRequest& q,const std::vector<uint8_t>& b,const cpu::NativeFmt& f,const CpuRows& c,bool compare_cpu=true) {
    using namespace strata::kernels;
    std::vector<int32_t> tok,original;
    for(size_t i=0;i<q.mask.size();++i)if(q.mask[i]){tok.push_back(i/q.K);original.push_back(i);}
    const int count=tok.size();require(count>0 && count<=2,"selected entry count bound");
    std::vector<int32_t> dst(count);for(int j=0;j<count;++j)dst[j]=j;
    const int32_t start[]={0,count},groups=1;
    auto L=native_expert_layout(8,8,f.n_embd,f.n_ff);
    Device weights(b.size()),x(q.activations.size()*4),xq(q.T*(f.n_embd/32)*36),ptr(8),st(8),ng(4),dd(count*4),tt(count*4),
        scratch(native_expert_scratch_bytes(count,f.n_ff)),out(count*f.n_embd*4);
    const unsigned long long wp=(unsigned long long)weights.p;
    auto upload=[](void* d,const void* h,size_t n){cu(cudaMemcpy(d,h,n,cudaMemcpyHostToDevice));};
    upload(weights,b.data(),b.size());upload(x,q.activations.data(),q.activations.size()*4);
    upload(ptr,&wp,8);upload(st,start,8);upload(ng,&groups,4);upload(dd,dst.data(),count*4);upload(tt,tok.data(),count*4);
    quantize_q8_1_rows((float*)x.p,q.T,f.n_embd,xq,nullptr);cu(cudaGetLastError());
    std::vector<uint8_t> packed(q.T*(f.n_embd/32)*36);cu(cudaMemcpy(packed.data(),xq,packed.size(),cudaMemcpyDeviceToHost));
    for(int t=0;compare_cpu && t<q.T;++t) {
        if(surveying && std::find(q.mask.begin()+t*q.K,q.mask.begin()+(t+1)*q.K,1)==q.mask.begin()+(t+1)*q.K)continue;
        packed_compare(("input_token_"+std::to_string(t)).c_str(),c.act.data()+t*cpu::kNativeActBytes,
                       packed.data()+t*(f.n_embd/32)*36,f.n_embd/32);
    }
    // Mark unwritten stage scratch as NaN. Some fused kernels do not materialize gate/up.
    cu(cudaMemset(scratch,0xff,native_expert_scratch_bytes(count,f.n_ff)));
    native_expert_grouped(L,(unsigned long long*)ptr.p,(int32_t*)st.p,(int32_t*)ng.p,(int32_t*)dd.p,(int32_t*)tt.p,
                          1,count,xq,scratch,(float*)out.p,nullptr);cu(cudaGetLastError());
    std::vector<float> result(count*f.n_embd),expected(result.size());cu(cudaMemcpy(result.data(),out,result.size()*4,cudaMemcpyDeviceToHost));
    for(int j=0;j<count;++j)std::copy_n(c.rows.data()+tok[j]*f.n_embd,f.n_embd,expected.data()+j*f.n_embd);
    const bool row_bits=!compare_cpu || metrics("CPU_vs_direct_CUDA_unweighted",expected,result);
    if(compare_cpu && (!row_bits || surveying)) {
        const size_t fa=(count*f.n_ff*4+255)&~size_t(255);
        std::vector<float> gate(count*f.n_ff),up(gate.size()),cg(gate.size()),cpu_up(gate.size());
        std::vector<uint8_t> hq(count*(f.n_ff/32)*36);
        cu(cudaMemcpy(gate.data(),scratch,gate.size()*4,cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(up.data(),(uint8_t*)scratch.p+fa,up.size()*4,cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(hq.data(),(uint8_t*)scratch.p+3*fa,hq.size(),cudaMemcpyDeviceToHost));
        for(int j=0;j<count;++j) {
            std::copy_n(c.gate.data()+tok[j]*f.n_ff,f.n_ff,cg.data()+j*f.n_ff);
            std::copy_n(c.up.data()+tok[j]*f.n_ff,f.n_ff,cpu_up.data()+j*f.n_ff);
            packed_compare(("hidden_entry_"+std::to_string(original[j])).c_str(),c.hq.data()+tok[j]*f.h_bytes,hq.data()+j*(f.n_ff/32)*36,f.n_ff/32);
        }
        const auto finite=[](float v){return std::isfinite(v);};
        if(!std::all_of(gate.begin(),gate.end(),finite)||!std::all_of(up.begin(),up.end(),finite)) {
            std::puts("STAGE gate_up_unavailable_or_nonfinite=1 hidden_F32_not_read=1");return result;
        }
        metrics("CPU_vs_CUDA_gate",cg,gate);metrics("CPU_vs_CUDA_up",cpu_up,up);
        // Feed the actual CUDA scale/codes to CPU dot, removing input quantization as a variable.
        std::vector<uint8_t> common(q.T*cpu::kNativeActBytes);
        for(int t=0;t<q.T;++t)for(int k=0;k<f.n_embd/32;++k) {
            auto* a=common.data()+t*cpu::kNativeActBytes+k*34;
            const auto* g=packed.data()+(t*(f.n_embd/32)+k)*36;
            std::memcpy(a,g,2);std::memcpy(a+2,g+4,32);
        }
        const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
        for(int j=0;j<count;++j)for(int k=0;k<f.n_ff;++k) {
            const auto* a=common.data()+tok[j]*cpu::kNativeActBytes;
            dot(f.n_embd,&cg[j*f.n_ff+k],0,b.data()+k*f.gu_row,0,a,0,1);
            dot(f.n_embd,&cpu_up[j*f.n_ff+k],0,b.data()+f.up_off+k*f.gu_row,0,a,0,1);
        }
        metrics("CPU_common_CUDA_input_vs_CUDA_gate",cg,gate);
        metrics("CPU_common_CUDA_input_vs_CUDA_up",cpu_up,up);
        std::vector<float> same_hidden(result.size());std::vector<uint8_t> common_h(f.h_bytes);
        for(int j=0;j<count;++j) {
            for(int k=0;k<f.n_ff/32;++k) {
                const auto* g=hq.data()+(j*(f.n_ff/32)+k)*36;
                std::memcpy(common_h.data()+k*34,g,2);std::memcpy(common_h.data()+k*34+2,g+4,32);
            }
            const void* h=common_h.data();float* outrow=same_hidden.data()+j*f.n_embd;
            cpu::native_down_rows(f,b.data(),&h,1,&outrow,0,f.n_embd);
        }
        metrics("CPU_common_CUDA_hidden_vs_CUDA_down",same_hidden,result);
        // The fused SwiGLU/quantizer need not materialize h. Do not inspect unwritten scratch.
        std::puts("STAGE hidden_F32_not_read_fused_kernel_may_not_materialize_it=1");
    }
    return result;
}
int run(const CpuCaptureFile& f,const GpuPlan& assigned,const Selection& s,const std::string& path,const fs::path& root) {
    auto p=one_asset(assigned,s.asset);auto b=read_blob(path,s.asset);std::string err;
    require(fs::create_directory(root),"scratch must be a new directory; existing cache/residents refused");
    const auto cache_dir=root/"one-expert-cache";require(fs::create_directory(cache_dir),"create isolated cache");
    cpu::NativeFmt fmt;require(cpu::native_fmt(8,8,2560,640,fmt,err),err);
    auto q=f.records[s.record].request;const auto full_mask=q.mask;
    for(size_t i=0;i<q.mask.size();++i)q.mask[i]=full_mask[i] && q.ids[i]==s.asset.expert;
    const auto c=cpu_rows(q,b,fmt);cu(cudaSetDevice(0));
    const auto direct=cuda_rows(q,b,fmt,c);
    GpuCache cache;std::vector<uint8_t> missing;require(cache.open(cache_dir.string(),16ull<<20,err)&&cache.prepare(p,missing,err)&&
        cache.publish(s.asset,b.data(),b.size(),err),err);
    auto executor=create_cuda_gpu_executor(0);GpuCounters counters;
    require(executor->prepare(p,256ull<<20,cache,counters,err),err);
    q.session=1;q.sequence=1;q.partition_id=p.placement_id.a;EntryRowsReply reply;
    require(executor->infer_rows(p,q,reply,err),err);
    std::vector<uint16_t> indices;for(size_t i=0;i<q.mask.size();++i)if(q.mask[i])indices.push_back(i);
    require(reply.indices==indices,"executor original-entry association mismatch");
    const bool executor_bits=metrics("direct_CUDA_vs_GPU_executor_unweighted",direct,reply.rows);
    std::printf("EXECUTOR gpu_weight_bytes=%llu selected_experts=1 entries=%zu actual_cuda=1\n",(unsigned long long)counters.gpu_weight_bytes,indices.size());
    executor->release(counters);cache.close();
    // Other entries are surrogate rows derived from this one real expert. This isolates reduction,
    // and does NOT claim to reproduce full captured partials, B routing, or full-model logits.
    auto original=f.records[s.record].request;std::vector<float> fixed(original.mask.size()*2560);
    for(size_t i=0;i<original.mask.size();++i)std::copy_n(c.rows.data()+(i/original.K)*2560,2560,fixed.data()+i*2560);
    for(size_t j=0;j<indices.size();++j)std::copy_n(reply.rows.data()+j*2560,2560,fixed.data()+indices[j]*2560);
    std::puts("REDUCTION fixed_real_selected_rows=1 surrogate_other_rows=1 loopback_fixture_only=1");
    fixed_reduce(p,original,b,fixed,cache_dir);
    return executor_bits ? 0 : 1; // CPU/CUDA differences are reported, never silently accepted as model parity.
}
int survey(const CpuCaptureFile& f,const std::vector<GpuPlan>& plans,const fs::path& blobs,const fs::path& root) {
    const auto inv=inventory(f,plans,blobs,true);
    require(!inv.missing,"selected blobs missing; stage only the NEEDED basenames before replay");
    require(fs::create_directory(root),"survey scratch must be a new directory");
    const auto cache_dir=root/"one-expert-cache";require(fs::create_directory(cache_dir),"create survey private cache");
    std::string err;GpuCache cache;
    cpu::NativeFmt fmt;require(cpu::native_fmt(8,8,2560,640,fmt,err),err);cu(cudaSetDevice(0));
    surveying=true;summaries.clear();const auto start=std::chrono::steady_clock::now();
    auto deadline=[&]{require(std::chrono::steady_clock::now()-start<std::chrono::seconds(300),"survey 300-second bound reached");};
    std::vector<std::vector<float>> cpu_dense(f.records.size()),gpu_dense(f.records.size());
    for(size_t r:inv.records){cpu_dense[r].resize(f.records[r].request.mask.size()*2560);gpu_dense[r].resize(cpu_dense[r].size());}
    bool executor_bits=true;size_t calls=0;
    for(size_t ai=0;ai<inv.assets.size();++ai) {
        deadline();const auto& selected=inv.assets[ai];const auto& assigned=plans[selected.plan];
        const auto p=one_asset(assigned,selected.asset);const auto b=read_blob((blobs/key(assigned,selected.asset)).string(),selected.asset);
        auto request=[&](size_t r) {
            auto q=f.records[r].request;
            for(size_t i=0;i<q.mask.size();++i)q.mask[i]=inv.masks[r][i] && q.ids[i]==selected.asset.expert;
            return q;
        };
        for(size_t r:selected.records) {
            deadline();auto q=request(r);const auto first=std::find(q.mask.begin(),q.mask.end(),1)-q.mask.begin();
            context_order={r,size_t(first)};context="record="+std::to_string(r)+" layer="+std::to_string(q.layer)+" expert="+std::to_string(selected.asset.expert);
            std::printf("CONTEXT %s asset=%zu plan=%zu phase=CPU_direct_CUDA\n",context.c_str(),ai,selected.plan);
            const auto c=cpu_rows(q,b,fmt);const auto direct=cuda_rows(q,b,fmt,c);size_t j=0;
            for(size_t i=0;i<q.mask.size();++i)if(q.mask[i]) {
                std::copy_n(c.rows.data()+(i/q.K)*2560,2560,cpu_dense[r].data()+i*2560);
                std::copy_n(direct.data()+j++*2560,2560,gpu_dense[r].data()+i*2560);
            }
            ++calls;
        }
        // Only the newly created private diagnostic cache is rotated. Staged/fleet caches are read-only inputs.
        stage_cache(cache,cache_dir,p,b);
        auto executor=create_cuda_gpu_executor(0);GpuCounters counters;require(executor->prepare(p,256ull<<20,cache,counters,err),err);
        uint64_t sequence=0;
        for(size_t r:selected.records) {
            deadline();auto q=request(r);q.session=1;q.sequence=++sequence;q.partition_id=p.placement_id.a;
            const auto first=std::find(q.mask.begin(),q.mask.end(),1)-q.mask.begin();context_order={r,size_t(first)};
            context="record="+std::to_string(r)+" layer="+std::to_string(q.layer)+" expert="+std::to_string(selected.asset.expert);
            EntryRowsReply reply;require(executor->infer_rows(p,q,reply,err),err);
            std::vector<uint16_t> indices;std::vector<float> direct;
            for(size_t i=0;i<q.mask.size();++i)if(q.mask[i]){indices.push_back(i);direct.insert(direct.end(),gpu_dense[r].begin()+i*2560,gpu_dense[r].begin()+(i+1)*2560);}
            require(indices==reply.indices,"survey executor original-entry association mismatch");
            std::printf("CONTEXT %s phase=GPU_executor\n",context.c_str());
            executor_bits=metrics("direct_CUDA_vs_GPU_executor_unweighted",direct,reply.rows)&&executor_bits;
            for(size_t j=0;j<indices.size();++j)std::copy_n(reply.rows.data()+j*2560,2560,gpu_dense[r].data()+indices[j]*2560);
        }
        executor->release(counters);cache.close();
    }
    cache.close();
    for(size_t r:inv.records)if(std::find(inv.masks[r].begin(),inv.masks[r].end(),1)!=inv.masks[r].end()) {
        deadline();context_order={r,0};context="record="+std::to_string(r);std::printf("CONTEXT %s phase=weighted_subset_reduction surrogate_rows=0\n",context.c_str());
        record_reduce(f.records[r].request,inv.masks[r],plans,cpu_dense[r],gpu_dense[r]);
    }
    for(const auto& [label,s]:summaries)std::printf("SUMMARY stage=%s observations=%zu differing=%zu maxabs=%.9g max_relL2=%.9g "
        "scale_mismatches=%zu code_mismatches=%zu max_scale_delta=%.9g max_code_delta=%d earliest=[%s] maxabs_at=[%s] max_relL2_at=[%s] max_scale_at=[%s] max_code_at=[%s]\n",
        label.c_str(),s.observations,s.different,s.maxabs,s.maxrel,s.scales,s.codes,s.maxscale,s.maxcode,s.earliest.c_str(),s.max_at.c_str(),s.maxrel_at.c_str(),s.maxscale_at.c_str(),s.maxcode_at.c_str());
    std::printf("SURVEY records=%zu selected_entries=%zu unique_blobs=%zu expert_record_calls=%zu total_blob_bytes=%llu "
        "seconds=%.6f executor_bits=%d surrogate_rows=0 no_listeners=1 full_capture_partials_checked=0 model_parity=unestablished\n",
        inv.records.size(),inv.selected,inv.assets.size(),calls,(unsigned long long)inv.bytes,
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),executor_bits);
    surveying=false;return executor_bits?0:1;
}
int grouped_control(const CpuCaptureFile& f,const std::vector<GpuPlan>& plans,const fs::path& blobs,const fs::path& root) {
    require(std::getenv("STRATA_KQ_SPLIT") && std::string(std::getenv("STRATA_KQ_SPLIT"))=="0","grouped control requires explicit STRATA_KQ_SPLIT=0 matching the audited network worker");
    const auto inv=inventory(f,plans,blobs,true,{0,828});
    require(!inv.missing && inv.active<=40,"grouped control requires the two records' staged blobs");
    require(f.records[0].request.layer==0 && f.records[828].request.layer==12,"grouped control record/layer identity");
    require(fs::create_directory(root),"grouped scratch must be a new directory");
    const auto cache_dir=root/"batch-cache";require(fs::create_directory(cache_dir),"create bounded batch cache");
    cu(cudaSetDevice(0));cuda_info();cpu::NativeFmt fmt;std::string err;require(cpu::native_fmt(8,8,2560,640,fmt,err),err);
    const auto start=std::chrono::steady_clock::now();auto deadline=[&]{require(std::chrono::steady_clock::now()-start<std::chrono::seconds(60),"grouped control 60-second bound");};
    size_t batches=0,calls=0;GpuCache cache;
    for(size_t r:inv.records)for(size_t peer=0;peer<plans.size();++peer) {
        deadline();auto q=f.records[r].request;std::vector<GpuAsset> assets;
        for(size_t i=0;i<q.mask.size();++i)q.mask[i]=inv.masks[r][i] && plans[peer].owns(q.layer,q.ids[i]);
        for(const auto& a:plans[peer].assets)if(a.layer==q.layer) {
            bool used=false;for(size_t i=0;i<q.mask.size();++i)used|=q.mask[i] && q.ids[i]==a.expert;
            if(used)assets.push_back(a);
        }
        if(assets.empty())continue;
        const auto p=subset_plan(plans[peer],assets);std::vector<float> standalone(q.mask.size()*2560);
        for(const auto& a:assets) {
            deadline();auto one=q;for(size_t i=0;i<one.mask.size();++i)one.mask[i]=q.mask[i] && q.ids[i]==a.expert;
            const auto b=read_blob((blobs/key(plans[peer],a)).string(),a);const auto c=cpu_rows(one,b,fmt);
            const auto rows=cuda_rows(one,b,fmt,c,false);size_t j=0;
            for(size_t i=0;i<one.mask.size();++i)if(one.mask[i])std::copy_n(rows.data()+j++*2560,2560,standalone.data()+i*2560);
        }
        // Bounded batch weights are read/published one blob at a time. No full plan or resident is loaded.
        cache.close();uint64_t removed=0;std::vector<uint8_t> missing;
        require(cache.open(cache_dir.string(),128ull<<20,err)&&cache.cleanup_unassigned(p,removed,err)&&cache.prepare(p,missing,err),err);
        for(size_t i=0;i<assets.size();++i)if(missing[i]) {
            const auto b=read_blob((blobs/key(plans[peer],assets[i])).string(),assets[i]);require(cache.publish(assets[i],b.data(),b.size(),err),err);
        }
        auto executor=create_cuda_gpu_executor(0);GpuCounters counters;require(executor->prepare(p,256ull<<20,cache,counters,err),err);
        std::printf("BATCH record=%zu layer=%u peer=%zu original_placement=%s%s experts=%zu gpu_weight_bytes=%llu\n",r,q.layer,peer,
            hex(plans[peer].placement_id.a).c_str(),hex(plans[peer].placement_id.b).c_str(),assets.size(),(unsigned long long)counters.gpu_weight_bytes);
        for(const auto& a:assets)std::printf("BATCH_ASSET expert=%u original_B_slot=%zu digest=%s%s\n",a.expert,
            size_t(plans[peer].find(a.layer,a.expert)-plans[peer].assets.data()),hex(a.digest.a).c_str(),hex(a.digest.b).c_str());
        GpuExecutionTrace trace;int last_expert=-1;
        for(int phase=0;phase<3;++phase) {
            deadline();auto test=q;test.session=1;test.sequence=phase+1;test.partition_id=p.placement_id.a;
            if(phase==1)for(size_t i=0;i<test.mask.size();++i)test.mask[i]=q.mask[i] && q.ids[i]==last_expert;
            EntryRowsReply reply;require(executor->infer_rows_diagnostic(p,test,reply,trace,err),err);
            std::printf("BATCH_PHASE record=%zu peer=%zu phase=%d kind=%s\n",r,peer,phase,phase==1?"one_real_expert_scratch_perturbation":"full_B_peer_batch");
            trace_check(test,p,trace);last_expert=trace.experts.back();std::vector<uint16_t> indices;std::vector<float> reference;
            for(size_t i=0;i<test.mask.size();++i)if(test.mask[i]){indices.push_back(i);reference.insert(reference.end(),standalone.begin()+i*2560,standalone.begin()+(i+1)*2560);}
            std::printf("META restored_indices=");for(auto i:reply.indices)std::printf("%u,",unsigned(i));std::puts("");
            require(reply.indices==indices,"grouped restored entry identity mismatch");++calls;
            if(!metrics("standalone_vs_B_peer_batch",reference,reply.rows)) {
                for(size_t j=0;j<reference.size();++j)if(std::memcmp(&reference[j],&reply.rows[j],4)) {
                    uint32_t a,b;std::memcpy(&a,&reference[j],4);std::memcpy(&b,&reply.rows[j],4);
                    std::printf("GROUPED_STOP record=%zu entry=%u token=%u channel=%zu peer=%zu phase=%d standalone=%.9g grouped=%.9g standalone_bits=%08x grouped_bits=%08x\n",
                        r,indices[j/2560],indices[j/2560]/q.K,j%2560,peer,phase,reference[j],reply.rows[j],a,b);break;
                }
                return 1;
            }
        }
        executor->release(counters);cache.close();++batches;
    }
    std::printf("GROUPED_DONE records=0,828 batches=%zu executor_calls=%zu seconds=%.6f bitexact=1 max_resident_experts=20 no_listeners=1\n",
        batches,calls,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());return 0;
}
int first_boundary(const CpuCaptureFile& f,const FirstBoundary& b,const std::vector<GpuPlan>& plans,
                   const fs::path& blobs,const fs::path& root) {
    // Every historical/SRA1 gate precedes CUDA initialization or weight reads.
    const char* reference_env=std::getenv("STRATA_CUDA_Q8_CPU_REFERENCE");
    const bool cpu_reference=reference_env && reference_env[0]=='1';
    std::printf("BOUNDARY_MODE %s\n",cpu_reference?"CPU_REFERENCE_CONTROL":"RETAINED_B_RECONSTRUCTION");
    const auto a0=boundary_cpu_gate(f,b);
    const auto inv=inventory(f,plans,blobs,true,{0});
    require(plans.size()==4 && inv.selected==13 && inv.assets.size()==13 && inv.bytes==67891200,
        "BOUNDARY B assignment count/bytes mismatch");
    const auto& original=f.records[0].request;
    for(size_t peer=0;peer<4;++peer)for(size_t i=0;i<20;++i)
        require(bool(b.peers[peer][i])==plans[peer].owns(0,original.ids[i]),"BOUNDARY B plan/trace peer ownership mismatch");
    require(!inv.missing,"BOUNDARY selected blobs missing; no full-model fallback");
    require(fs::create_directory(root),"BOUNDARY output must be a new directory");
    const auto cache_dir=root/"one-expert-cache";require(fs::create_directory(cache_dir),"BOUNDARY private cache create failed");
    std::string err;GpuCache cache;auto rows=b.rows;const auto start=std::chrono::steady_clock::now();
    cu(cudaSetDevice(0));cpu_info();cuda_info();size_t calls=0;
    for(const auto& selected:inv.assets) {
        require(std::chrono::steady_clock::now()-start<std::chrono::seconds(60),"BOUNDARY 60-second CUDA budget exceeded");
        const auto& plan=plans[selected.plan];const auto p=one_asset(plan,selected.asset);
        const auto blob=read_blob((blobs/key(plan,selected.asset)).string(),selected.asset);
        stage_cache(cache,cache_dir,p,blob);
        auto executor=create_cuda_gpu_executor(0);GpuCounters counters;
        require(executor->prepare(p,256ull<<20,cache,counters,err),err);
        auto q=original;for(size_t i=0;i<20;++i)q.mask[i]=b.peers[selected.plan][i] && q.ids[i]==selected.asset.expert;
        q.session=1;q.sequence=1;q.partition_id=p.placement_id.a;
        EntryRowsReply reply;require(executor->infer_rows(p,q,reply,err),err);
        std::vector<uint16_t> expected;for(size_t i=0;i<20;++i)if(q.mask[i])expected.push_back((uint16_t)i);
        require(expected.size()==1 && reply.indices==expected && reply.rows.size()==2560,"BOUNDARY CUDA original-entry association mismatch");
        const size_t i=expected[0];std::vector<float> cpu_row(b.rows.begin()+i*2560,b.rows.begin()+(i+1)*2560);
        std::printf("BOUNDARY_ROW entry=%zu token=%zu expert=%u peer=%zu ",i,i/10,q.ids[i],selected.plan);
        const bool row_bits=metrics("CPU_vs_CUDA_unweighted",cpu_row,reply.rows);
        if(cpu_reference)require(row_bits,"CPU reference CUDA row mismatch; model trace not qualified");
        std::copy(reply.rows.begin(),reply.rows.end(),rows.begin()+i*2560);
        executor->release(counters);cache.close();++calls;
    }
    auto mixed=boundary_reduce(b,rows,"actual_mixed_rows");
    auto save=[&](const char* name,const std::vector<float>& data){std::ofstream out(root/name,std::ios::binary);
        require(bool(out.write((const char*)data.data(),data.size()*4)),"BOUNDARY result write failed");};
    save("CPU-original-entry-rows.f32",b.rows);save("mixed-original-entry-rows.f32",rows);
    save("reconstructed-A0-union.f32",a0);save("reconstructed-B-union.f32",mixed);
    const bool recorded_b_bits=metrics("actual_rows_reconstruct_recorded_B_union",b.b,mixed);
    const bool exact=cpu_reference?metrics("CPU_reference_union_vs_recorded_A0",b.a0,mixed):recorded_b_bits;
    metrics("recorded_A0_vs_recorded_B_union",b.a0,b.b);
    std::printf("BOUNDARY_DONE A0_bits=1 B_bits=%d recorded_B_bits=%d target_bits=%d cpu_reference=%d identical_row_controls=1 CUDA_rows=%zu CUDA_blobs=%zu "
        "blob_bytes=67891200 seconds=%.6f no_CPU_recompute=1 no_listeners=1 model_parity=unestablished\n",
        recorded_b_bits,recorded_b_bits,exact,cpu_reference,calls,inv.assets.size(),std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    return exact?0:1;
}
#endif

void first_boundary_check() {
    FirstBoundary b;b.prompt.resize(38*4);for(size_t i=0;i<38;++i)b.prompt[4*i]=(uint8_t)i;
    b.x.assign(5120,.1f);b.weights.assign(20,1.f);b.expected.assign(20,1);b.cpu_mask.assign(20,1);b.peers.assign(4,std::vector<uint8_t>(20));
    b.rows.assign(20*2560,0.f);b.ids.resize(20);
    for(size_t i=0;i<20;++i){b.ids[i]=(uint32_t)i;if(i<13){b.cpu_mask[i]=0;b.peers[i%4][i]=1;}}
    // Cancelling values make a regrouped sum differ from original-entry order.
    for(int t=0;t<2;++t)for(int h=0;h<2560;++h) {
        b.rows[(t*10)*2560+h]=1e20f;b.rows[(t*10+1)*2560+h]=-1e20f;b.rows[(t*10+2)*2560+h]=1.f;
    }
    b.a0=boundary_reference(b,b.rows);b.b=b.a0;b.sra_partial=b.a0;
    CpuCaptureFile f;f.setup.n_embd=2560;f.setup.n_ff=640;f.setup.gu_type=f.setup.d_type=8;
    f.prompt_tokens=38;f.max_new=256;f.tokens=blob_digest(b.prompt);f.windows.push_back({1,1,2,0});
    CpuCaptureRecord r;r.window=1;r.sent=true;r.request.layer=0;r.request.T=2;r.request.K=10;
    r.request.activations=b.x;r.request.weights=b.weights;r.request.mask=b.expected;
    for(auto id:b.ids)r.request.ids.push_back((uint16_t)id);
    r.partials=b.a0;f.records.push_back(r);
    (void)boundary_cpu_gate(f,b);
    auto changed=b.rows;changed[2*2560]+=.125f;b.b=boundary_reference(b,changed);
    require(metrics("synthetic_boundary_B",b.b,boundary_reduce(b,changed,"synthetic_mixed")),"synthetic B reconstruction");
    auto refuse=[&](auto change){auto bad=f;change(bad);bool stopped=false;try{boundary_historic(bad,b);}catch(...){stopped=true;}
        require(stopped,"boundary identity mismatch not refused");};
    refuse([](auto& bad){bad.records[0].request.activations[0]+=.1f;});
    refuse([](auto& bad){bad.records[0].request.ids[0]++;});
    refuse([](auto& bad){bad.records[0].request.weights[0]+=.1f;});
    refuse([](auto& bad){bad.records[0].request.mask[0]=0;});
    refuse([](auto& bad){bad.records[0].token_begin=1;});
    refuse([](auto& bad){bad.records[0].partials[0]+=.1f;});
    auto bad_rows=b;bad_rows.rows[2*2560]+=.125f;bool stopped=false;
    try{(void)boundary_cpu_gate(f,bad_rows);}catch(...){stopped=true;}
    require(stopped,"SRA1 row mismatch not refused");
    b.sra_partial[0]+=.1f;stopped=false;try{boundary_historic(f,b);}catch(...){stopped=true;}
    require(stopped,"SRA1 partial mismatch not refused");b.sra_partial=b.a0;
    // Exercise the exact bounded parser with a synthetic packet, then corrupt its header.
    std::vector<uint8_t> packet{'S','F','B','1'};
    for(uint32_t v:{2u,10u,2560u,38u,256u,0u,0u,0u})put(packet,v);
    packet.insert(packet.end(),FirstBoundarySraSha,FirstBoundarySraSha+32);
    auto append=[&](const auto& v){const auto* p=(const uint8_t*)v.data();packet.insert(packet.end(),p,p+v.size()*sizeof(v[0]));};
    append(b.prompt);append(b.x);append(b.ids);append(b.weights);append(b.expected);append(b.cpu_mask);
    for(const auto& mask:b.peers)append(mask);
    append(b.a0);append(b.b);append(b.rows);append(b.sra_partial);
    require(packet.size()==287220,"synthetic boundary packet size");
    char path[]="/tmp/strata-first-boundary-XXXXXX";const int fd=mkstemp(path);require(fd>=0,"boundary fixture create");close(fd);
    struct Cleanup {const char* path;~Cleanup(){std::remove(path);}} cleanup{path};
    auto write=[&]{std::ofstream out(path,std::ios::binary);require(bool(out.write((char*)packet.data(),packet.size())),"boundary fixture write");};
    write();const auto decoded=read_boundary(path);(void)boundary_cpu_gate(f,decoded);
    packet[36]^=1;write();stopped=false;try{(void)read_boundary(path);}catch(...){stopped=true;}
    require(stopped,"bad packet SRA1 binding not refused");
    std::puts("BOUNDARY_CHECK raw_identity_refusals=6 SRA1_partial_and_row_refusals=1 packet_refusal=1 ordered_reconstruction=1 identical_row_controls=1 actual_CUDA=0");
}
}
int main(int argc,char** argv) {
    try {
        if((argc==2 && std::string(argv[1])=="--stage-check") || ((argc==2 || argc==3) && std::string(argv[1])=="--stage-check80")) {
            const int T=std::string(argv[1])=="--stage-check80"?8:1,K=T==8?10:2;
            Fixture fixture;const auto setup=fixture.setup();const auto path=(fixture.root/"capture.bin").string();std::string err;CpuCapture capture;
            require(capture.prepare(setup,fixture.m.identity,fixture.m.index_digest,path,err,1u<<20),err);
            require(capture.request("GEN 1 1",std::vector<int64_t>(T==8?9:1,1),1),"stage fixture capture request");
            for(int window=0;window<(T==8?2:1);++window) {
              const int rows=window?1:T;
              require(capture.window(window?T:0,rows,T==8 && !window?1:2),"stage fixture capture window");
              for(int layer=0;layer<2;++layer){InferRequest q;q.session=1;q.sequence=layer+1;q.model_id=setup.model_id;q.partition_id=setup.partition_id;q.layer=layer;q.T=rows;q.K=K;
                q.ids.resize(rows*K);for(int i=0;i<rows*K;++i)q.ids[i]=1-i%2;
                q.mask.assign(rows*K,1);q.weights.assign(rows*K,.5f);q.activations.assign(rows*2560,.1f);
                for(int t=0;t<rows;++t)std::fill_n(q.activations.begin()+t*2560,2560,.1f+float(t)*.01f);
                std::vector<float> partial(rows*2560,.25f);
                require(capture.begin(q) && capture.finish(partial.data()),"stage fixture capture record");}
            }
            const float logits[]={1.f,2.f};require(capture.seal(logits,2),"stage fixture capture seal");
            const auto output=(fixture.root/"staged").string();
            auto invoke=[&]{const pid_t child=fork();require(child>=0,"stage fixture fork");if(!child){execl(argv[0],argv[0],"--stage-record",path.c_str(),"0",(fixture.root/"local.manifest").c_str(),fixture.root.c_str(),output.c_str(),static_cast<char*>(nullptr));_exit(127);}int status=0;require(waitpid(child,&status,0)==child && WIFEXITED(status),"stage fixture wait");return WEXITSTATUS(status);};
            require(invoke()==0,"stage CLI failed");GpuPlan plan;require(read_gpu_plan(output+"/one-layer.plan",plan,err),err);
            require(plan.assets.size()==2 && plan.assets[0].expert==0 && plan.assets[1].expert==1,"full record staging lost an expert");
            GpuCache cache;require(cache.open(output+"/blobs",plan.assigned_bytes(),err),err);std::vector<uint8_t> missing,blob;
            require(cache.prepare(plan,missing,err) && missing==std::vector<uint8_t>({0,0}),"staged blobs failed reuse validation");
            for(const auto& a:plan.assets){require(cache.read(a,blob,err) && blob==fixture.blobs[a.expert],"staged expert bytes differ");}cache.close();
            if(T==8) {
                // Optional private export reuses this existing synthetic capture,
                // verified staged plan and blobs. No model/backend is started.
                const fs::path retained=argc==3?fs::path(argv[2]):fixture.root/"export-check";
                require(retained.is_absolute() && fs::create_directory(retained),"stage export must be a new absolute directory");
                fs::permissions(retained,fs::perms::owner_all,fs::perm_options::replace);
                fs::copy_file(path,retained/"capture.bin");
                fs::copy_file(output+"/one-layer.plan",retained/"one-layer.plan");
                fs::copy_file(output+"/assignments.txt",retained/"assignments.txt");
                fs::create_directory(retained/"blobs");
                fs::permissions(retained/"blobs",fs::perms::owner_all,fs::perm_options::replace);
                for(const auto& asset:plan.assets)fs::copy_file(fs::path(output)/"blobs"/key(plan,asset),retained/"blobs"/key(plan,asset));
                CpuCaptureFile exported;GpuPlan exported_plan;
                require(read_cpu_capture((retained/"capture.bin").string(),exported,err),err);
                require(read_gpu_plan((retained/"one-layer.plan").string(),exported_plan,err),err);
                require(exported.records.front().request.T==8 && exported.records.front().request.K==10 &&
                    exported.records.front().request.mask==std::vector<uint8_t>(80,1) && exported_plan.assigned_bytes()==plan.assigned_bytes() && exported_plan.assets.size()==2,
                    "exported 80-entry capture/plan changed");
                if(argc==3)std::printf("STAGE_EXPORT synthetic=true GPU=0 record=0 entries=80 path=%s\n",retained.c_str());
            }
            require(invoke()!=0,"stage overwrote existing output");std::puts("STAGE_CHECK full_record=1 existing_manifest_source_cache=1 exact_blobs=1 no_overwrite=1 GPU=0");return 0;
        }
        if(argc==7 && std::string(argv[1])=="--stage-record") {
            // Offline only: preserve every active entry, reuse verified manifest
            // ranges and the existing one-blob source/cache publication path.
            CpuCaptureFile f;LocalManifest m;std::string err;
            require(read_cpu_capture(argv[2],f,err),err);
            const std::string ordinal=argv[3];require(!ordinal.empty() && ordinal.find_first_not_of("0123456789")==std::string::npos,"invalid record index");
            const size_t record=std::stoull(ordinal);require(record<f.records.size() && f.records[record].sent,"record absent/not sent");
            const auto& q=f.records[record].request;
            require(read_local_manifest(argv[4],m,err) && m.matches(f.setup) && m.identity==f.identity && m.index_digest==f.index,err.empty()?"capture/source manifest mismatch":err);
            std::set<std::pair<uint16_t,uint16_t>> unique;size_t entries=0;
            for(size_t i=0;i<q.mask.size();++i)if(q.mask[i]){unique.emplace(q.layer,q.ids[i]);++entries;}
            require(!unique.empty() && unique.size()<=24 && entries<=strata::core::portable_detail::q8_entry_capacity,"full record exceeds provider bounds; not truncated");
            GpuPlan plan;require(make_gpu_plan(m,{unique.begin(),unique.end()},plan,err),err);
            const fs::path output=argv[6];require(fs::create_directory(output),"stage output must be a new directory");
            fs::permissions(output,fs::perms::owner_all,fs::perm_options::replace);
            require(write_gpu_plan((output/"one-layer.plan").string(),plan,err),err);
            std::ofstream assignments(output/"assignments.txt");for(auto [layer,expert]:unique)assignments<<layer<<' '<<expert<<'\n';require(bool(assignments),"assignment write failed");
            for(size_t i=0;i<q.mask.size();++i)if(q.mask[i])std::printf("FULL_ENTRY record=%zu layer=%u entry=%zu token=%zu expert=%u\n",record,q.layer,i,i/q.K,q.ids[i]);
            GpuAssetSource source;GpuCache cache;std::vector<uint8_t> missing,blob;
            require(source.open(argv[5],m,plan,err),err);
            require(fs::create_directory(output/"blobs"),"stage blob directory create failed");
            require(cache.open((output/"blobs").string(),plan.assigned_bytes(),err) && cache.prepare(plan,missing,err),err);
            for(const auto& asset:plan.assets){require(source.read(asset,blob,err),err);require(cache.publish(asset,blob.data(),blob.size(),err),err);}
            cache.close();
            std::printf("FULL_RECORD record=%zu entries=%zu groups=%zu bytes=%llu source_staging_peak=%llu source_weight_bytes=%llu no_remask=1\n",record,entries,unique.size(),(unsigned long long)plan.assigned_bytes(),(unsigned long long)source.stats().host_staging_peak,(unsigned long long)source.stats().source_read_bytes);
            return 0;
        }
        if(argc==2 && std::string(argv[1])=="--boundary-check") {first_boundary_check();return 0;}
        if(argc>=2 && std::string(argv[1])=="--boundary") {
            require(argc==10,"--boundary requires CAPTURE PACKET BLOB_DIR NEW_OUTPUT and four B plans in original peer order");
            CpuCaptureFile f;std::string err;require(read_cpu_capture(argv[2],f,err),err);
            const auto b=read_boundary(argv[3]);
#ifdef STRATA_CAPTURE_CUDA
            std::vector<GpuPlan> plans;for(int i=6;i<argc;++i){GpuPlan p;require(read_gpu_plan(argv[i],p,err),err);plans.push_back(p);}
            return first_boundary(f,b,plans,argv[4],argv[5]);
#else
            (void)boundary_cpu_gate(f,b);
            throw std::runtime_error("--boundary needs CUDA target gpu_capture_replay; CPU reuse gates passed, CUDA not run");
#endif
        }
        if(argc==2 && std::string(argv[1])=="--self-check") {
            // The real fixture writes GGUF slices, so test digest refusal separately below.
            Fixture fixture;GpuPlan p;std::string err;require(make_gpu_plan(fixture.m,{{0,1}},p,err),err);
            CpuCaptureFile f;f.setup=fixture.setup();CpuCaptureRecord r;r.sent=true;r.request.layer=0;r.request.T=2;
            r.request.K=3;r.request.ids={0,1,0,1,0,1};r.request.mask={1,1,1,1,1,1};r.request.weights.assign(6,1.f);r.request.activations.assign(5120,.1f);f.records.push_back(r);
            const auto s=select(f,{p});require(s.entry==1 && s.asset.expert==1,"first B intersection");
            auto one=one_asset(p,s.asset);const auto c=cpu_rows(r.request,fixture.blobs[1],fixture.fmt);
            require(c.rows.size()==5120,"native CPU row shape");
            std::vector<float> rows(6*2560);const float adversarial[]={1e20f,-1e20f,1.f,1.f,1e20f,-1e20f};
            for(int i=0;i<6;++i)std::fill_n(rows.data()+i*2560,2560,adversarial[i]);
            const auto root=fixture.root/"fixed";fs::create_directory(root);fixed_reduce(one,r.request,fixture.blobs[1],rows,root);
            auto none=f;none.records[0].request.mask={1,0,1,0,1,0};bool refused=false;
            try{select(none,{p});}catch(...){refused=true;}require(refused,"missing intersection refused");
            auto foreign=p;foreign.model_id^=1;refused=false;try{select(f,{foreign});}catch(...){refused=true;}require(refused,"foreign model refused");
            const auto file=fixture.root/"single.blob";{std::ofstream o(file,std::ios::binary);o.write((char*)fixture.blobs[1].data(),fixture.blobs[1].size());}
            require(read_blob(file.string(),s.asset)==fixture.blobs[1],"bound blob read");
            auto bad=s.asset;bad.digest.a^=1;refused=false;try{read_blob(file.string(),bad);}catch(...){refused=true;}require(refused,"bad digest refused");
            GpuPlan other;require(make_gpu_plan(fixture.m,{{0,0}},other,err),err);
            auto sample=f;auto& sq=sample.records[0].request;sq.K=10;sq.ids.resize(20);sq.mask.assign(20,1);sq.weights.assign(20,.1f);
            for(int i=0;i<20;++i)sq.ids[i]=i%2;
            auto inv=inventory(sample,{p,other},fixture.root,false);
            require(inv.records.size()==1 && inv.assets.size()==2 && inv.selected==20 && inv.missing==2,"survey exact bounded inventory");
            for(const auto& selected:inv.assets) {
                const std::vector<GpuPlan> plans{p,other};const auto out=fixture.root/key(plans[selected.plan],selected.asset);
                const auto& b=fixture.blobs[selected.asset.expert];std::ofstream o(out,std::ios::binary);o.write((char*)b.data(),b.size());
            }
            inv=inventory(sample,{p,other},fixture.root,false);require(inv.missing==0,"survey staged inventory");
            GpuCache rotating;const auto private_cache=fixture.root/"rotating";fs::create_directory(private_cache);
            stage_cache(rotating,private_cache,p,fixture.blobs[1]);
            stage_cache(rotating,private_cache,other,fixture.blobs[0]);rotating.close();
            require(fs::exists(private_cache/key(other,other.assets[0])) && !fs::exists(private_cache/key(p,p.assets[0])),"one-expert private cache rotation");
            std::vector<float> real(20*2560);for(int i=0;i<20;++i)std::copy_n(c.rows.data()+(i/10)*2560,2560,real.data()+i*2560);
            record_reduce(sq,inv.masks[0],{p,other},real,real);
            auto tq=r.request;for(size_t i=0;i<tq.mask.size();++i)tq.mask[i]=tq.ids[i]==1;
            GpuExecutionTrace tr;tr.experts={1};tr.asset_slots={0};tr.start={0,3};tr.dst={0,1,2};tr.tok={0,1,1};tr.original={1,3,5};
            trace_check(tq,p,tr);tr.tok[1]=0;refused=false;try{trace_check(tq,p,tr);}catch(...){refused=true;}require(refused,"bad actual trace mapping refused");
            GpuPlan full;require(make_gpu_plan(fixture.m,{{0,0},{0,1}},full,err),err);
            require(subset_plan(full,full.assets).placement_id==full.placement_id,"multi-expert subset preserves metadata");
            const auto out=fixture.root/"cpu-export";export_cpu_row(out,c,fixture.fmt,0);size_t exported=0;
            for(const char* suffix:{".input.q8_0",".gate.f32",".up.f32",".hidden.f32",".hidden.q8_0",".row.f32"})exported+=fs::file_size(out.string()+suffix);
            require(exported==21320,"one-entry CPU export size bound");
            std::puts("SELF_CHECK selection=1 digest=1 repeated_expert_T2=1 mixed_reducer=1 survey_inventory=1 subset_reducer=1 batch_metadata=1 cpu_export=1 actual_cuda=0");return 0;
        }
        const bool selecting=argc>=4 && std::string(argv[1])=="--select";
        const bool running=argc>=6 && std::string(argv[1])=="--run";
        const bool listing=argc>=5 && std::string(argv[1])=="--inventory";
        const bool survey_mode=argc>=6 && std::string(argv[1])=="--survey";
        const bool grouped=argc>=6 && std::string(argv[1])=="--grouped";
        const bool cpu_mode=argc>=6 && std::string(argv[1])=="--cpu-control";
        if(!selecting && !running && !listing && !survey_mode && !grouped && !cpu_mode)throw std::runtime_error("usage: --stage-check80 [NEW_PRIVATE_EXPORT] | --stage-record CAPTURE RECORD LOCAL_MANIFEST MODEL_ROOT NEW_OUTPUT | --select CAPTURE B_PLAN... | --run CAPTURE SELECTED_BLOB NEW_SCRATCH B_PLAN... | --inventory CAPTURE BLOB_DIR B_PLAN... | --survey/--grouped/--cpu-control CAPTURE BLOB_DIR NEW_OUTPUT B_PLAN... | --self-check | --boundary CAPTURE PACKET BLOB_DIR NEW_OUTPUT B_PLAN... | --boundary-check");
        CpuCaptureFile f;std::string err;require(read_cpu_capture(argv[2],f,err),err);
        std::vector<GpuPlan> plans;const int begin=selecting?3:listing?4:5;
        for(int i=begin;i<argc;++i){GpuPlan p;require(read_gpu_plan(argv[i],p,err),err);plans.push_back(p);}
        if(listing)return inventory(f,plans,argv[3],true).missing?3:0;
        if(cpu_mode){cpu_control(f,plans,argv[3],argv[4]);return 0;}
        if(grouped) {
#ifdef STRATA_CAPTURE_CUDA
            return grouped_control(f,plans,argv[3],argv[4]);
#else
            throw std::runtime_error("--grouped requires CUDA target gpu_capture_replay");
#endif
        }
        if(survey_mode) {
#ifdef STRATA_CAPTURE_CUDA
            return survey(f,plans,argv[3],argv[4]);
#else
            throw std::runtime_error("--survey requires CUDA target gpu_capture_replay");
#endif
        }
        const auto s=select(f,plans);const auto& q=f.records[s.record].request;const auto& p=plans[s.plan];
        std::printf("SELECT record=%zu entry=%zu layer=%u token=%zu expert=%u plan=%s original_placement=%s%s bytes=%llu digest=%s%s cache_key=%s\n",
            s.record,s.entry,q.layer,s.entry/q.K,s.asset.expert,argv[begin+s.plan],hex(p.placement_id.a).c_str(),hex(p.placement_id.b).c_str(),
            (unsigned long long)s.asset.bytes,hex(s.asset.digest.a).c_str(),hex(s.asset.digest.b).c_str(),key(p,s.asset).c_str());
        if(selecting)return 0;
#ifdef STRATA_CAPTURE_CUDA
        return run(f,p,s,argv[3],argv[4]);
#else
        throw std::runtime_error("--run requires CUDA target gpu_capture_replay; CPU target only selects/tests");
#endif
    }catch(const std::exception& e){std::fprintf(stderr,"REFUSED %s\n",e.what());return 2;}
}
