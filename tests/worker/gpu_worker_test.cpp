// Sources only on Geekom. Remote fixtures exercise real SGW1 sockets and optionally real CUDA.
#include "local_fixture.hpp"
#include "strata/worker/gpu_client.hpp"
#include "strata/worker/gpu_worker.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef STRATA_TEST_TLS
#include "tls_fixture.hpp"
#endif

namespace {
struct FakeExecutor final : GpuExecutor {
    bool capacity=false,loss=false;
    int prepare_delay_ms=0,prepare_calls=0;
    bool prepare(const GpuPlan& p,uint64_t,GpuCache& cache,GpuCounters& c,std::string& err)override{
        ++prepare_calls;
        if(prepare_delay_ms)std::this_thread::sleep_for(std::chrono::milliseconds(prepare_delay_ms));
        if(capacity){err="synthetic available VRAM below required allocation/reserve";return false;}
        std::vector<uint8_t> b;for(const auto& a:p.assets)if(!cache.read(a,b,err))return false;
        c.gpu_weight_bytes=p.assigned_bytes();c.gpu_scratch_bytes=1024;c.gpu_allocated_bytes=p.assigned_bytes()+1024;
        c.gpu_peak_bytes=std::max(c.gpu_peak_bytes,c.gpu_allocated_bytes);++c.gpu_allocation_count;return true;
    }
    bool infer(const GpuPlan& p,const InferRequest& r,InferReply& out,std::string& err)override{
        if(loss){err="synthetic device disconnected";return false;}
        out.partials.assign((size_t)r.T*p.n_embd,0.125f);return true;
    }
    void release(GpuCounters& c)override{c.gpu_allocated_bytes=c.gpu_weight_bytes=c.gpu_scratch_bytes=0;}
};
struct GpuService {
    GpuWorkerService service;uint16_t port=0;std::thread thread;
    GpuService(GpuWorkerOptions options,std::unique_ptr<GpuExecutor> executor):service(std::move(options),std::move(executor)){
        std::string err;require(service.listen(port,err),err);thread=std::thread([&]{service.run();});
    }
    void stop(){service.stop();if(thread.joinable())thread.join();}
    ~GpuService(){stop();}
};
GpuPlan plan_for(const Fixture& f,bool addition=false){
    GpuPlan p;std::string err;
    std::vector<std::pair<uint16_t,uint16_t>> assigned{{0,0},{0,1}};if(addition)assigned.push_back({1,0});
    require(make_gpu_plan(f.m,assigned,p,err),err);return p;
}
GpuWorkerOptions options_for(const Fixture& f,const GpuPlan& plan,const std::string& suffix){
    GpuWorkerOptions o;o.plan=plan;o.cache_root=(f.root/suffix).string();fs::create_directory(o.cache_root);
    o.cache_budget=64ull<<20;o.runtime_reserve=256ull<<20;o.deadline_ms=2000;o.setup_ms=30000;return o;
}
GpuWorkerTier::Open open_for(const Fixture& f,const GpuPlan& plan,uint16_t port,int& fetches){
    GpuWorkerTier::Open o;o.plan=plan;o.port=port;o.deadline_ms=5000;o.setup_ms=30000;
    o.fetch=[&f,&fetches](int64_t l,int64_t e,uint8_t* dst,uint32_t n){++fetches;
        const auto& b=f.blobs[(size_t)l*2+e];if(n!=b.size())return false;std::memcpy(dst,b.data(),n);return true;};return o;
}
void populate_cache(const Fixture& f,const GpuWorkerOptions& options){
    GpuCache cache;std::vector<uint8_t> missing;std::string err;
    require(cache.open(options.cache_root,options.cache_budget,err)&&cache.prepare(options.plan,missing,err),err);
    for(const auto& a:options.plan.assets){const auto& bytes=f.blobs[(size_t)a.layer*2+a.expert];
        require(cache.publish(a,bytes.data(),bytes.size(),err),err);}
}
InferRequest request_for(const GpuPlan& p,uint64_t session=123){
    InferRequest r;r.session=session;r.model_id=p.model_id;r.partition_id=p.placement_id.a;r.sequence=1;
    r.layer=0;r.T=2;r.K=2;r.mask={1,1,1,1};r.ids={1,0,0,1};r.weights={0.3f,0.7f,0.6f,0.4f};
    r.activations.resize((size_t)r.T*p.n_embd);
    for(size_t i=0;i<r.activations.size();++i)r.activations[i]=0.1f*(int(i%11)-5);return r;
}
std::vector<float> run_tier(GpuWorkerTier& tier,const GpuPlan& plan){
    auto r=request_for(plan);std::vector<int32_t> ids(r.ids.begin(),r.ids.end());std::string err;
    require(tier.post(r.layer,r.activations.data(),ids.data(),r.weights.data(),r.mask.data(),r.T,r.K,err),err);
    require(tier.posted()&&tier.finish(err)&&tier.served(),err);return {tier.partials(),tier.partials()+(size_t)r.T*plan.n_embd};
}
std::vector<float> float_reference(const Fixture& f,const InferRequest& r){
    const auto* traits=ggml_get_type_traits(GGML_TYPE_Q8_0);const size_t H=f.fmt.n_embd,F=f.fmt.n_ff;
    std::vector<float> out((size_t)r.T*H,0.f),row(H),down(F),hidden(F),expert(H);
    for(size_t entry=0;entry<r.mask.size();++entry)if(r.mask[entry]){
        const auto& b=f.blobs[(size_t)r.layer*2+r.ids[entry]];const float* x=r.activations.data()+(entry/r.K)*H;
        for(size_t j=0;j<F;++j){
            traits->to_float(b.data()+j*f.fmt.gu_row,row.data(),H);double g=0,u=0;
            for(size_t h=0;h<H;++h)g+=(double)row[h]*x[h];
            traits->to_float(b.data()+f.fmt.up_off+j*f.fmt.gu_row,row.data(),H);
            for(size_t h=0;h<H;++h)u+=(double)row[h]*x[h];
            hidden[j]=(float)(g/(1.0+std::exp(-g))*u);
        }
        for(size_t h=0;h<H;++h){traits->to_float(b.data()+f.fmt.down_off+h*f.fmt.d_row,down.data(),F);
            double v=0;for(size_t j=0;j<F;++j)v+=(double)down[j]*hidden[j];expert[h]=(float)v;}
        float* target=out.data()+(entry/r.K)*H;
        for(size_t h=0;h<H;++h)target[h]=target[h]+r.weights[entry]*expert[h];
    }return out;
}
void compare(const std::vector<float>& got,const std::vector<float>& ref,double tolerance,const char* label,double max_tolerance=std::numeric_limits<double>::infinity()){
    require(got.size()==ref.size(),"numerical result shape");double sum=0,denom=0,maxabs=0,square=0;
    for(size_t i=0;i<got.size();++i){require(std::isfinite(got[i])&&std::isfinite(ref[i]),"numerical finite");
        const double d=std::fabs((double)got[i]-ref[i]);sum+=d;square+=d*d;denom+=std::fabs((double)ref[i]);maxabs=std::max(maxabs,d);}
    const double rel=sum/(denom+1e-30);std::printf("NUMERICS %s rel_L1=%.9g max_abs=%.9g RMSE=%.9g tolerance=%.9g max_tolerance=%.9g\n",label,rel,maxabs,std::sqrt(square/got.size()),tolerance,max_tolerance);
    check(rel<=tolerance&&maxabs<=max_tolerance,label);
}
void codec_checks(const GpuPlan& p){
    auto r=request_for(p);std::vector<uint8_t> encoded;require(gpu_codec::encode_request(p,r,encoded),"encode request");
    InferRequest decoded;check(gpu_codec::decode_request(p,encoded,decoded),"full model/layout/placement request codec");
    auto bad=encoded;bad[16]^=1;check(!gpu_codec::decode_request(p,bad,decoded),"stale layout rejected");
    InferReply reply;reply.session=r.session;reply.sequence=r.sequence;reply.partials.assign((size_t)r.T*p.n_embd,1.f);
    const auto digest=blob_digest(encoded);auto wire=gpu_codec::encode_reply(p,digest,reply);InferReply output;
    check(gpu_codec::decode_reply(p,digest,r,wire,output),"bound exact reply accepted");
    bad=wire;bad.pop_back();check(!gpu_codec::decode_reply(p,digest,r,bad,output),"missing result bytes rejected");
    bad=wire;bad.insert(bad.end(),wire.begin(),wire.end());check(!gpu_codec::decode_reply(p,digest,r,bad,output),"duplicate results rejected");
    ++reply.sequence;bad=gpu_codec::encode_reply(p,digest,reply);check(!gpu_codec::decode_reply(p,digest,r,bad,output),"stale sequence reply rejected");
    reply.sequence=r.sequence;reply.partials[0]=std::numeric_limits<float>::quiet_NaN();bad=gpu_codec::encode_reply(p,digest,reply);
    check(!gpu_codec::decode_reply(p,digest,r,bad,output),"nonfinite output rejected");
}
#ifndef _WIN32
void transport_diagnostics() {
    auto run=[](const std::vector<uint8_t>& bytes,bool close_peer,bool cancelled,int deadline,const char* stage,TransportOutcome outcome,size_t count) {
        int fd[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"socketpair");
        if(!bytes.empty())require(::send(fd[0],bytes.data(),bytes.size(),0)==(ssize_t)bytes.size(),"test send");
        if(close_peer)::shutdown(fd[0],SHUT_WR);
        WireFrame frame;std::string err;gpu_codec::ReceiveDiagnostic d;
        check(!gpu_codec::receive(fd[1],frame,kGpuFrameMax,deadline,[&]{return cancelled;},err,0,nullptr,&d) &&
              std::string(d.stage)==stage && d.transport.outcome==outcome && d.transport.bytes==count,"bounded receive diagnostic");
        ::close(fd[0]);::close(fd[1]);
    };
    run({},true,false,100,"header",TransportOutcome::eof,0);
    run({1,2,3},true,false,100,"header",TransportOutcome::eof,3);
    std::vector<uint8_t> partial;gpu_codec::put(partial,kGpuWireMagic,4);gpu_codec::put(partial,GPU_HELLO,2);
    gpu_codec::put(partial,ST_OK,2);gpu_codec::put(partial,4);partial.push_back(1);
    run(partial,true,false,100,"payload",TransportOutcome::eof,1);
    run({},false,false,20,"header",TransportOutcome::timeout,0);
    run({},false,true,100,"header",TransportOutcome::cancelled,0);
    {
        int fd[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"one-shot socketpair");
        int calls=0;WireFrame frame;std::string why;gpu_codec::ReceiveDiagnostic detail;
        check(!gpu_codec::receive(fd[1],frame,kGpuFrameMax,500,[&]{return ++calls==2;},why,0,nullptr,&detail) &&
              detail.transport.outcome==TransportOutcome::cancelled && calls==2,"one-shot cancellation during wait is not re-polled");
        ::close(fd[0]);::close(fd[1]);
    }
    {
        // No peer bytes: the second cancellation poll occurs inside the first-byte idle read.
        int fd[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"idle cancellation socketpair");
        int calls=0;WireFrame frame;std::string err;gpu_codec::ReceiveDiagnostic d;
        check(!gpu_codec::receive(fd[1],frame,kGpuFrameMax,5000,[&]{return ++calls>=2;},err,60000,nullptr,&d)&&
              calls>=2&&std::string(d.stage)=="header"&&d.transport.outcome==TransportOutcome::cancelled&&
              d.transport.bytes==0&&err.find("first byte")!=std::string::npos,
              "cancellation interrupts the quiet first-byte idle receive");
        ::close(fd[0]);::close(fd[1]);
    }
    std::vector<uint8_t> malformed(16,0);run(malformed,false,false,100,"malformed_header",TransportOutcome::none,0);
    {
        // The opt-in first-byte (idle) budget is distinct from the frame deadline.
        int fd[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"idle socketpair");
        WireFrame frame;std::string err;gpu_codec::ReceiveDiagnostic d;
        const auto began=std::chrono::steady_clock::now();
        check(!gpu_codec::receive(fd[1],frame,kGpuFrameMax,5000,{},err,30,nullptr,&d)&&
              d.transport.outcome==TransportOutcome::timeout&&d.transport.bytes==0&&
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count()<1000,
              "quiet idle wait ends by its own short budget, not the frame deadline");
        ::close(fd[0]);::close(fd[1]);
    }
    {
        int fd[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"ready socketpair");
        std::string err;std::vector<uint8_t> payload;
        gpu_codec::put(payload,kGpuWireVersion,4);gpu_codec::put(payload,kGpuCapabilityCuda,4);
        gpu_codec::put(payload,kGpuNumericsQ8_1,4);
        require(gpu_codec::send(fd[0],GPU_HELLO,ST_OK,payload,1000,{},err),"preload one frame");
        WireFrame frame;gpu_codec::ReceiveDiagnostic d;
        check(gpu_codec::receive(fd[1],frame,kGpuFrameMax,1000,{},err,1,nullptr,&d)&&
              frame.msg==GPU_HELLO&&frame.payload==payload,
              "already available frame needs no idle wait");
        ::close(fd[0]);::close(fd[1]);
    }
    {
        int fd[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"shared budget socketpair");
        std::vector<uint8_t> wire;gpu_codec::put(wire,kGpuWireMagic,4);gpu_codec::put(wire,GPU_HELLO,2);
        gpu_codec::put(wire,ST_OK,2);gpu_codec::put(wire,(uint64_t)4);
        // Header completes well into the frame budget; the complete body arrives
        // after that budget, but comfortably before a renewed body budget expires.
        wire.insert(wire.end(),{1,2,3,4});
        std::thread writer([&]{::send(fd[0],wire.data(),1,MSG_NOSIGNAL);
            sock::sleep_ms(700);::send(fd[0],wire.data()+1,kHeaderBytes-1,MSG_NOSIGNAL);
            sock::sleep_ms(500);::send(fd[0],wire.data()+kHeaderBytes,4,MSG_NOSIGNAL);});
        WireFrame frame;std::string err;gpu_codec::ReceiveDiagnostic d;
        const bool ok=gpu_codec::receive(fd[1],frame,kGpuFrameMax,1000,{},err,3000,nullptr,&d);
        writer.join();
        check(!ok&&std::string(d.stage)=="payload"&&d.transport.outcome==TransportOutcome::timeout,
              "header and complete body share one frame deadline");
        ::close(fd[0]);::close(fd[1]);
    }
    int fd[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"send socketpair");
    TransportRead send_detail;std::string err;
    check(!gpu_codec::send(fd[0],GPU_HELLO,ST_OK,{},100,[]{return true;},err,nullptr,&send_detail) &&
          send_detail.outcome==TransportOutcome::cancelled && send_detail.bytes==0,"send cancellation diagnostic");
    ::close(fd[0]);::close(fd[1]);
    TransportRead early;check(!gpu_codec::send(0,GPU_HELLO,ST_OK,{},0,{},err,nullptr,&early)&&early.outcome==TransportOutcome::timeout,"invalid send deadline classified");
    WireFrame frame;gpu_codec::ReceiveDiagnostic receive_early;
    check(!gpu_codec::receive(0,frame,kGpuFrameMax,0,{},err,0,nullptr,&receive_early)&&receive_early.transport.outcome==TransportOutcome::timeout,"invalid receive deadline classified");
}
#endif
uint64_t raw_setup(sock_fd fd,const GpuPlan& p){
    // local_fixture.hpp also defines put<T>(buffer, value), whose width is sizeof(T).
    // SGW1 fields use explicit codec widths, independent of literal or host integer types.
    std::vector<uint8_t> payload;
    gpu_codec::put(payload,kGpuWireVersion,4);
    gpu_codec::put(payload,kGpuCapabilityCuda,4);
    gpu_codec::put(payload,kGpuNumericsQ8_1,4);
    const std::vector<uint8_t> hello_bytes{1,0,0,0,1,0,0,0,1,0,0,0};
    require(payload==hello_bytes,"SGW1 HELLO is three little-endian u32 fields at offsets 0,4,8");
    gpu_codec::Cursor hello{payload};
    require(hello.get(4)==kGpuWireVersion&&hello.get(4)==kGpuCapabilityCuda&&
            hello.get(4)==kGpuNumericsQ8_1&&hello.done(),"SGW1 HELLO decodes exactly 12 bytes");
    std::string err;WireFrame out;
    require(gpu_codec::send(fd,GPU_HELLO,ST_OK,payload,2000,{},err)&&
            gpu_codec::receive(fd,out,kGpuFrameMax,2000,{},err)&&out.status==ST_OK,err);
    payload.clear();gpu_codec::put(payload,171,8);
    const std::vector<uint8_t> nonce_bytes{0xab,0,0,0,0,0,0,0};
    require(payload==nonce_bytes,"SGW1 SETUP nonce occupies eight little-endian bytes even for an int literal");
    std::vector<uint8_t> plan;require(encode_gpu_plan(p,plan,err),err);
    payload.insert(payload.end(),plan.begin(),plan.end());
    require(payload.size()==8+plan.size()&&std::memcmp(payload.data()+8,plan.data(),plan.size())==0,
            "SGW1 SETUP plan begins at offset 8 and has no padding or trailing bytes");
    gpu_codec::Cursor setup{payload};
    require(setup.get(8)==171&&setup.i==8&&setup.ok,"SGW1 SETUP nonce decoder consumes exactly eight bytes");
    GpuPlan decoded;
    require(decode_gpu_plan(payload.data()+8,payload.size()-8,decoded,err)&&
            decoded.placement_id==p.placement_id,"SGW1 SETUP plan decodes from the wire offset");
    auto short_nonce=nonce_bytes;short_nonce.resize(4);
    short_nonce.insert(short_nonce.end(),plan.begin(),plan.end());
    require(!decode_gpu_plan(short_nonce.data()+8,short_nonce.size()-8,decoded,err),
            "a four-byte nonce cannot masquerade as the eight-byte SETUP prefix");
    err.clear();
    require(gpu_codec::send(fd,GPU_SETUP,ST_OK,payload,2000,{},err)&&
            gpu_codec::receive(fd,out,kGpuFrameMax,2000,{},err)&&out.status==ST_OK,err);
    gpu_codec::Cursor c{out.payload};return c.get(8);
}
void raw_ready(sock_fd fd,const GpuPlan& p){
    const auto session=raw_setup(fd,p);
    std::string err; WireFrame reply;
    require(gpu_codec::send(fd,GPU_READY,ST_OK,gpu_codec::binding(session,p),2000,{},err)&&
            gpu_codec::receive(fd,reply,kGpuFrameMax,2000,{},err)&&reply.status==ST_OK,err);
}
// A deliberately dishonest peer sends invalid successful inference replies through the
// real transport/client. Codec unit checks alone do not exercise failure latching.
struct MalformedPeer {
    enum Mode { truncated,duplicated,wrong_binding,nonfinite,wrong_sequence,wrong_index,missing_index,oversized,unsupported };
    sock_fd listener=kBadFd;uint16_t port=0;std::thread thread;
    bool completed=false,retry_frame=false;std::string failure;
    MalformedPeer(const GpuPlan& plan,Mode mode,bool rows=false){
        std::string err;require(sock::listen_loopback(port,listener,err),err);
        thread=std::thread([&,plan,mode,rows]{
            sock_fd peer=kBadFd;
            try{
                require(sock::accept_one(listener,peer,2000),"malformed peer accept deadline");
                constexpr uint64_t session=919;
                const uint16_t expected[]={GPU_HELLO,GPU_SETUP,GPU_READY,GPU_STATUS,(uint16_t)(rows?GPU_INFER_ROWS:GPU_INFER)};
                for(uint16_t msg:expected){
                    WireFrame frame;std::string why;
                    require(gpu_codec::receive(peer,frame,kGpuFrameMax,2000,{},why)&&frame.msg==msg,why);
                    std::vector<uint8_t> response;
                    if(msg==GPU_HELLO){
                        gpu_codec::put(response,kGpuWireVersion,4);gpu_codec::put(response,kGpuCapabilityCuda|(rows&&mode!=unsupported?kGpuCapabilityEntryRows:0),4);
                        gpu_codec::put(response,kGpuNumericsQ8_1,4);gpu_codec::put(response,kUploadChunkMax,4);
                    }else if(msg==GPU_SETUP){
                        response=gpu_codec::binding(session,plan);gpu_codec::put(response,plan.assets.size(),4);
                        response.insert(response.end(),plan.assets.size(),0);
                    }else if(msg==GPU_READY)response=gpu_codec::binding(session,plan);
                    else if(msg==GPU_STATUS){
                        GpuCounters counters;counters.assigned_bytes=counters.cached_bytes=counters.gpu_weight_bytes=plan.assigned_bytes();
                        counters.gpu_allocated_bytes=plan.assigned_bytes()+1024;
                        response=gpu_codec::binding(session,plan);gpu_codec::put(response,1,4);
                        const auto values=gpu_codec::counters(counters);response.insert(response.end(),values.begin(),values.end());
                    }else{
                        InferRequest request;require(gpu_codec::decode_request(plan,frame.payload,request),"malformed peer request decode");
                        InferReply reply;reply.session=session;reply.sequence=request.sequence;
                        reply.partials.assign((size_t)request.T*plan.n_embd,0.5f);
                        if(mode==nonfinite)reply.partials[0]=std::numeric_limits<float>::quiet_NaN();
                        if(mode==wrong_sequence)++reply.sequence;
                        response=gpu_codec::encode_reply(plan,blob_digest(frame.payload),reply);
                        if(rows) {
                            EntryRowsReply entry;entry.session=session;entry.sequence=request.sequence;
                            for(size_t i=0;i<request.mask.size();++i)if(request.mask[i])entry.indices.push_back((uint16_t)i);
                            entry.rows.assign(entry.indices.size()*plan.n_embd,.5f);
                            require(gpu_codec::encode_rows_reply(plan,blob_digest(frame.payload),request,entry,response),"dishonest row encode");
                            if(mode==nonfinite) {uint32_t nan=0x7fc00000;std::memcpy(response.data()+86,&nan,4);}
                            if(mode==wrong_sequence)response[72]^=1;
                            if(mode==wrong_index)response[84]=1; // duplicate first/second index
                            if(mode==missing_index)response[82]-=1;
                            if(mode==oversized)response.resize(65+kEntryRowsPayloadMax);
                        }
                        if(mode==truncated)response.pop_back();
                        if(mode==duplicated){const auto copy=response;response.insert(response.end(),copy.begin(),copy.end());}
                        if(mode==wrong_binding)response[16]^=1;
                    }
                    const bool sent=gpu_codec::send(peer,msg+1,ST_OK,response,2000,{},why);
                    // An oversized duplicate is rejected from its header; the client may
                    // close before the dishonest peer finishes sending that payload.
                    if(!sent&&(msg==GPU_INFER || msg==GPU_INFER_ROWS)&&(mode==duplicated || mode==oversized)){completed=true;sock::close_fd(peer);return;}
                    require(sent,why);
                    if(mode==unsupported)break;
                }
                // The client must close immediately; a second post must not send another frame.
                WireFrame extra;std::string why;
                retry_frame=gpu_codec::receive(peer,extra,kGpuFrameMax,500,{},why);
                completed=true;
            }catch(const std::exception& e){failure=e.what();}
            sock::close_fd(peer);
        });
    }
    void join(){if(thread.joinable())thread.join();sock::close_fd(listener);}
    ~MalformedPeer(){join();}
};
void malformed_peer_checks(const Fixture& f,const GpuPlan& plan,bool rows=false){
    for(auto mode:{MalformedPeer::truncated,MalformedPeer::duplicated,MalformedPeer::wrong_binding,
                  MalformedPeer::nonfinite,MalformedPeer::wrong_sequence,MalformedPeer::wrong_index,MalformedPeer::missing_index,MalformedPeer::oversized}){
        if(!rows && mode>=MalformedPeer::wrong_index)continue;
        MalformedPeer peer(plan,mode,rows);GpuWorkerTier tier;int fetches=0;std::string err;
        auto option=open_for(f,plan,peer.port,fetches);option.entry_rows=rows;
        require(tier.open(option,err),err);
        const auto r=request_for(plan);std::vector<int32_t> ids(r.ids.begin(),r.ids.end());
        check(!tier.post(r.layer,r.activations.data(),ids.data(),r.weights.data(),r.mask.data(),r.T,r.K,err)&&
            tier.dead()&&!tier.served()&&!tier.posted()&&tier.partial_count()==0&&tier.entry_rows().rows.empty(),
            "malformed network reply kills tier and clears partials");
        check(!tier.post(r.layer,r.activations.data(),ids.data(),r.weights.data(),r.mask.data(),r.T,r.K,err),
            "dead CUDA tier refuses repeat post");
        peer.join();
        check(peer.completed&&peer.failure.empty()&&!peer.retry_frame,"failed CUDA request sends no invisible retry frame");
        check(fetches==0,"malformed peer fixture transfers no expert weights");
    }
}
void unsupported_peer_check(const Fixture& f,const GpuPlan& plan) {
    MalformedPeer peer(plan,MalformedPeer::unsupported,true);GpuWorkerTier tier;int fetches=0;std::string err;
    auto option=open_for(f,plan,peer.port,fetches);option.entry_rows=true;
    check(!tier.open(option,err)&&tier.dead()&&!tier.ready(),"weighted-only peer cannot downgrade negotiated rows");
    peer.join();check(peer.completed&&peer.failure.empty()&&!peer.retry_frame&&fetches==0,"unsupported peer receives no setup, weights or retry");
}
void setup_budget_checks(const Fixture& f,const GpuPlan& plan){
    for(int mode=0;mode<3;++mode){
        auto options=options_for(f,plan,"cache-budget-"+std::to_string(mode));populate_cache(f,options);
        if(mode==1)options.setup_ms=50;
        auto executor=std::make_unique<FakeExecutor>();executor->prepare_delay_ms=200;
        const auto* observed=executor.get();GpuService service(options,std::move(executor));
        GpuWorkerTier tier;int fetches=0;std::string err;auto open=open_for(f,plan,service.port,fetches);
        open.deadline_ms=100;open.setup_ms=mode==2?50:2000;
        const bool opened=tier.open(open,err);
        if(mode==0)check(opened&&tier.ready(),"READY wait may exceed frame deadline within total setup budget");
        else check(!opened&&tier.dead()&&!tier.ready(),"server/client whole setup deadline never exposes READY");
        tier.close();service.stop();
        check(fetches==0,"delayed setup uses verified cached fixtures");
        if(mode==0)check(observed->prepare_calls==1,"delayed READY executor reached before setup result");
        check(service.service.counters().gpu_allocated_bytes==0,"setup stop releases GPU allocations before counter read");
    }
}
// The opt-in idle budget lets a session wait for its next request frame past the
// short per-frame deadline; started frames, expiry, stop and the default stay bounded.
void idle_budget_checks(const Fixture& f,const GpuPlan& p){
    const auto eof_within=[&](sock_fd fd,int least_ms,int most_ms,const char* label){
        // connect_numeric_controlled returns a nonblocking socket; wait explicitly
        // so EAGAIN cannot masquerade as an early close or hang the suite.
        char byte=0;const auto began=std::chrono::steady_clock::now();
        pollfd ready{fd,POLLIN,0};
        const int polled=::poll(&ready,1,most_ms);
        const auto n=polled>0?::recv(fd,&byte,1,0):-1;
        const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count();
        check(n==0&&elapsed>=least_ms&&elapsed<=most_ms,label);
    };
    {
        auto options=options_for(f,p,"cache-idle-gap");populate_cache(f,options);
        options.deadline_ms=250;options.idle_ms=3000;
        GpuService service(options,std::make_unique<FakeExecutor>());
        GpuWorkerTier tier;int fetches=0;std::string err;
        require(tier.open(open_for(f,p,service.port,fetches),err),err);
        std::this_thread::sleep_for(std::chrono::milliseconds(700)); // past deadline_ms, inside idle_ms
        run_tier(tier,p);
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
        run_tier(tier,p);
        check(fetches==0&&service.service.counters().failures==0,
            "valid request after an allowed idle gap serves verified cache without failure");
        tier.close();service.stop();
    }
    {
        auto options=options_for(f,p,"cache-idle-default");populate_cache(f,options);
        options.deadline_ms=250; // idle_ms stays 0: unchanged default behavior
        GpuService service(options,std::make_unique<FakeExecutor>());
        GpuWorkerTier tier;int fetches=0;std::string err;
        require(tier.open(open_for(f,p,service.port,fetches),err),err);
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
        auto r=request_for(p);std::vector<int32_t> ids(r.ids.begin(),r.ids.end());
        check(!tier.post(r.layer,r.activations.data(),ids.data(),r.weights.data(),r.mask.data(),r.T,r.K,err)&&tier.dead(),
            "without the idle option the frame deadline still closes an idle session");
        service.stop();
        check(service.service.counters().failures==1,"default idle close counted once");
    }
    {
        auto options=options_for(f,p,"cache-idle-pre-ready");
        options.deadline_ms=250;options.idle_ms=3000;
        GpuService service(options,std::make_unique<FakeExecutor>());
        sock_fd fd=kBadFd;std::string err;
        require(sock::connect_numeric_controlled("127.0.0.1",service.port,2000,{},fd,err),err);
        eof_within(fd,120,1500,"pre-READY idle retains frame deadline");
        sock::close_fd(fd);service.stop();
    }
    for(int mode=0;mode<2;++mode){ // stalled partial header, then stalled partial body
        auto options=options_for(f,p,"cache-idle-stall-"+std::to_string(mode));populate_cache(f,options);
        options.deadline_ms=300;options.idle_ms=20000;
        GpuService service(options,std::make_unique<FakeExecutor>());
        sock_fd fd=kBadFd;std::string err;
        require(sock::connect_numeric_controlled("127.0.0.1",service.port,2000,{},fd,err),err);
        raw_ready(fd,p);
        std::vector<uint8_t> stalled;
        gpu_codec::put(stalled,kGpuWireMagic,4);gpu_codec::put(stalled,GPU_STATUS,2);
        gpu_codec::put(stalled,ST_OK,2);gpu_codec::put(stalled,(uint64_t)48);
        stalled.resize(mode?kHeaderBytes+8:6);
        require(::send(fd,stalled.data(),stalled.size(),0)==(ssize_t)stalled.size(),"stalled frame send");
        eof_within(fd,150,2000,mode?"stalled partial body still fails under the active frame deadline":
                                   "stalled partial header still fails under the active frame deadline");
        sock::close_fd(fd);service.stop();
        check(service.service.counters().failures==1,"stalled frame counted once");
    }
    {
        auto options=options_for(f,p,"cache-idle-expiry");populate_cache(f,options);
        options.deadline_ms=5000;options.idle_ms=500;
        GpuService service(options,std::make_unique<FakeExecutor>());
        sock_fd fd=kBadFd;std::string err;
        require(sock::connect_numeric_controlled("127.0.0.1",service.port,2000,{},fd,err),err);
        raw_ready(fd,p);
        eof_within(fd,300,2000,"idle budget alone bounds a quiet session");
        sock::close_fd(fd);service.stop();
        check(service.service.counters().failures==1,"idle expiry counted once");
    }
    {
        auto options=options_for(f,p,"cache-idle-stop");populate_cache(f,options);
        options.deadline_ms=5000;options.idle_ms=3600000;
        GpuService service(options,std::make_unique<FakeExecutor>());
        sock_fd fd=kBadFd;std::string err;
        require(sock::connect_numeric_controlled("127.0.0.1",service.port,2000,{},fd,err),err);
        raw_ready(fd,p);
        const auto began=std::chrono::steady_clock::now();
        service.stop();
        const auto stopped=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count();
        eof_within(fd,0,2000,"stopped idle wait closes the connection");
        check(stopped<=3000,"stop stays responsive during a long idle wait");
        sock::close_fd(fd);
    }
    for(int idle:{-1,kGpuWorkerIdleMaxMs+1}){
        auto options=options_for(f,p,"cache-idle-invalid");options.idle_ms=idle;
        GpuWorkerService service(options,std::make_unique<FakeExecutor>());
        uint16_t port=0;std::string err;
        check(!service.listen(port,err)&&err.find("idle")!=std::string::npos&&port==0,
            "idle budget bounded and validated before any listener");
    }
}
// Exercise the real listener/session path with synthetic execution only.
void listener_checks(const Fixture& f,const GpuPlan& p){
    auto probe=[&](const std::string& bind,const std::vector<std::string>& peers,bool valid){
        GpuWorkerOptions o;o.bind_ip=bind;o.allow_peer_ips=peers;o.plan=p;
        // Empty cache root is a deterministic sentinel after listener validation, before bind.
        GpuWorkerService service(o,std::make_unique<FakeExecutor>());uint16_t port=0;std::string err;
        check(!service.listen(port,err),"policy probe never opens a socket");
        check((err=="CUDA cache root must already be an operator-created directory")==valid,
              "CUDA listener policy accepts only explicit private settings or loopback defaults");
    };
    probe("127.0.0.1",{},true);
    for(const auto* ip:{"10.1.2.3","172.16.0.1","172.31.255.254","192.168.1.2"}){
        probe(ip,{"10.1.2.4","127.0.0.1"},true);probe(ip,{},false);
    }
    const std::vector<std::string> bad{"","0.0.0.0","0.1.2.3","224.0.0.1","255.255.255.255",
        "localhost","::1","10.1.2.3/24","10.1.2.999",std::string("10.1.2.3\0extra",14)};
    for(const auto& ip:bad){probe(ip,{"10.1.2.4"},false);probe("10.1.2.3",{"10.1.2.4",ip},false);}
    for(const auto* ip:{"8.8.8.8","192.0.2.1","169.254.1.2","172.15.255.255","172.32.0.1","192.169.1.1"})
        probe(ip,{"10.1.2.4"},false);
    for(int deadline:{0,-1}){
        auto o=options_for(f,p,"cache-listener-deadline-"+std::to_string(deadline));
        o.allow_peer_ips={"127.0.0.1"};o.deadline_ms=deadline;
        GpuWorkerService service(o,std::make_unique<FakeExecutor>());uint16_t port=0;std::string err;
        check(!service.listen(port,err),"filtered listener retains bounded frame deadline requirement");
    }
    for(int mode=0;mode<3;++mode){
        const char* bind=mode==2?"127.0.0.4":"127.0.0.1";
        auto o=options_for(f,p,"cache-listener-"+std::to_string(mode));o.bind_ip=bind;
        if(mode)o.allow_peer_ips={"127.0.0.1","127.0.0.3","127.0.0.1"};
        o.deadline_ms=5000;
        GpuService service(o,std::make_unique<FakeExecutor>());
        auto connect_source=[&](const char* source){
            sock_fd fd=::socket(AF_INET,SOCK_STREAM,0);require(fd!=kBadFd,"source socket");
            sockaddr_in from{},to{};from.sin_family=to.sin_family=AF_INET;to.sin_port=htons(service.port);
            require(::inet_pton(AF_INET,source,&from.sin_addr)==1&&
                    ::inet_pton(AF_INET,bind,&to.sin_addr)==1,"numeric fixture addresses");
            if(::bind(fd,(sockaddr*)&from,sizeof from)!=0||::connect(fd,(sockaddr*)&to,sizeof to)!=0){
                sock::close_fd(fd);require(false,"source connect");}
            return fd;
        };
        auto denied=connect_source("127.0.0.2");
        if(mode){
            pollfd ready{denied,POLLIN,0};check(::poll(&ready,1,1500)>0,"denied source closes before frame deadline");
            char byte;const auto n=::recv(denied,&byte,1,MSG_DONTWAIT);
            check(n==0||(n<0&&errno==ECONNRESET),"denied source receives no protocol bytes");
        }else raw_setup(denied,p); // unchanged default admits any source that can reach loopback
        sock::close_fd(denied);
        for(const auto* source:{"127.0.0.1","127.0.0.3"}){
            auto fd=connect_source(source);raw_setup(fd,p);sock::close_fd(fd);
        }
        int fetches=0;GpuWorkerTier tier;auto open=open_for(f,p,service.port,fetches);open.host=bind;std::string err;
        require(tier.open(open,err),err);run_tier(tier,p);tier.close();
        service.stop();check(service.service.counters().requests==1,"allowed inference survives rejected peer");
    }
}
void lifecycle(const Fixture& f,const GpuPlan& p){
    unsupported_peer_check(f,p);codec_checks(p);malformed_peer_checks(f,p);malformed_peer_checks(f,p,true);setup_budget_checks(f,p);idle_budget_checks(f,p);std::string err;
    {
        auto executor=std::make_unique<FakeExecutor>();GpuService service(options_for(f,p,"cache-no-rows"),std::move(executor));
        int fetches=0;GpuWorkerTier tier;auto option=open_for(f,p,service.port,fetches);option.entry_rows=true;
        check(!tier.open(option,err)&&tier.dead()&&fetches==0,"unsupported entry-row executor refuses negotiation without weighted fallback");
    }
    {
        auto executor=std::make_unique<FakeExecutor>();GpuService service(options_for(f,p,"cache-main"),std::move(executor));
        int fetches=0;GpuWorkerTier cold;require(cold.open(open_for(f,p,service.port,fetches),err),err);
        check(fetches==2&&cold.stats().setup_network_bytes==p.assigned_bytes(),"only assigned cold blobs transferred");run_tier(cold,p);cold.close();
        GpuWorkerTier warm;int missing=0;require(warm.open(open_for(f,p,service.port,missing),err),err);
        check(missing==0&&warm.stats().setup_network_bytes==0&&warm.stats().cache_reused_bytes==p.assigned_bytes(),"verified disk cache reuse across new session");
        run_tier(warm,p);warm.close();
    }
    {
        auto executor=std::make_unique<FakeExecutor>();GpuService service(options_for(f,p,"cache-early"),std::move(executor));
        sock_fd fd=kBadFd;require(sock::connect_numeric_controlled("127.0.0.1",service.port,2000,{},fd,err),err);
        const auto session=raw_setup(fd,p);WireFrame reply;
        require(gpu_codec::send(fd,GPU_READY,ST_OK,gpu_codec::binding(session,p),2000,{},err)&&gpu_codec::receive(fd,reply,1024,2000,{},err),err);
        check(reply.status==ST_INCOMPLETE,"READY rejected while any assigned blob missing");sock::close_fd(fd);
    }
    {
        auto executor=std::make_unique<FakeExecutor>();executor->capacity=true;GpuService service(options_for(f,p,"cache-capacity"),std::move(executor));
        GpuWorkerTier tier;int n=0;check(!tier.open(open_for(f,p,service.port,n),err)&&tier.dead()&&!tier.ready(),"capacity refusal never exposes READY");
    }
    {
        auto executor=std::make_unique<FakeExecutor>();executor->loss=true;GpuService service(options_for(f,p,"cache-loss"),std::move(executor));
        GpuWorkerTier tier;int n=0;require(tier.open(open_for(f,p,service.port,n),err),err);auto r=request_for(p);std::vector<int32_t> ids(r.ids.begin(),r.ids.end());
        check(!tier.post(0,r.activations.data(),ids.data(),r.weights.data(),r.mask.data(),r.T,r.K,err)&&tier.dead()&&!tier.served(),"device loss latches failure without partials or CPU retry");
        check(tier.holds(0,0),"failed GPU retains frozen execution ownership");
    }
    {
        Service old(f.options());GpuWorkerTier tier;int n=0;check(!tier.open(open_for(f,p,old.port,n),err)&&tier.dead(),"CUDA client clearly refuses old CPU protocol");
        ExpertWorkerClient cpu;connect(cpu,old);setup(cpu,f);std::vector<std::pair<uint16_t,uint16_t>> slots{{0,0},{0,1},{1,0},{1,1}};
        std::vector<uint8_t> hits;require(cpu.load_local(slots,hits,err)&&cpu.mark_ready(err),err);infer(cpu,f,0,1,1);
    }
    {
        auto executor=std::make_unique<FakeExecutor>();GpuService service(options_for(f,p,"cache-stale"),std::move(executor));
        auto wrong=p;wrong.model_id^=1; // Cannot manufacture a valid new identity by editing one field.
        check(!validate_gpu_plan(wrong,err),"modified placement rejected before connect");
        sock_fd fd=kBadFd;require(sock::connect_numeric_controlled("127.0.0.1",service.port,2000,{},fd,err),err);
        const auto session=raw_setup(fd,p);WireFrame response;auto stale=gpu_codec::binding(session+1,p);
        require(gpu_codec::send(fd,GPU_READY,ST_OK,stale,2000,{},err)&&gpu_codec::receive(fd,response,1024,2000,{},err),err);
        check(response.status==ST_STALE_SESSION,"stale session refused by service");sock::close_fd(fd);
    }
}
#ifndef _WIN32
void failure_log_checks(const Fixture& f,const GpuPlan& p) {
    { GpuWorkerTier empty;std::string err;check(!empty.close_orderly(err),"unopened tier cannot orderly-close"); }
    FILE* capture=std::tmpfile();require(capture!=nullptr,"diagnostic capture");
    const int saved=::dup(STDERR_FILENO);require(saved>=0 && ::dup2(::fileno(capture),STDERR_FILENO)>=0,"redirect diagnostic");
    std::string err;
    {
        GpuService service(options_for(f,p,"cache-diagnostic-close"),std::make_unique<FakeExecutor>());
        int fetches=0;GpuWorkerTier first;require(first.open(open_for(f,p,service.port,fetches),err),err);
        auto request=request_for(p);std::vector<int32_t> ids(request.ids.begin(),request.ids.end());
        require(first.post(request.layer,request.activations.data(),ids.data(),request.weights.data(),request.mask.data(),request.T,request.K,err),err);
        check(!first.close_orderly(err)&&first.ready(),"pending reply cannot be orderly-closed");
        require(first.finish(err)&&first.served()&&first.close_orderly(err),err);
        GpuWorkerTier second;require(second.open(open_for(f,p,service.port,fetches),err),err);run_tier(second,p);second.close();
        service.stop();check(service.service.counters().requests==2,"ordinary close permits another client");
    }
    {
        auto executor=std::make_unique<FakeExecutor>();executor->loss=true;
        GpuService service(options_for(f,p,"cache-diagnostic-executor"),std::move(executor));
        int fetches=0;GpuWorkerTier tier;require(tier.open(open_for(f,p,service.port,fetches),err),err);
        auto r=request_for(p);std::vector<int32_t> ids(r.ids.begin(),r.ids.end());
        check(!tier.post(r.layer,r.activations.data(),ids.data(),r.weights.data(),r.mask.data(),r.T,r.K,err),"executor failure refused");
        service.stop();check(service.service.counters().failures>=1,"executor failure counted");
    }
    std::fflush(stderr);require(::dup2(saved,STDERR_FILENO)>=0,"restore diagnostic");::close(saved);
    std::rewind(capture);std::string log;char buf[512];while(size_t n=std::fread(buf,1,sizeof(buf),capture))log.append(buf,n);std::fclose(capture);
    check(log.find("site=receive stage=header outcome=eof bytes=0")!=std::string::npos,"clean close recorded as header EOF");
    check(log.find("site=dispatch msg=9")!=std::string::npos,"executor failure recorded at dispatch");
}
#endif
#ifdef STRATA_TEST_TLS
void tls_pending_frame() {
    TlsFixture cert;int fd[2];require(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"TLS socketpair");
    bool sent=false;std::thread sender([&]{TlsSession tls;std::string err;
        if(!tls.handshake(fd[0],cert.config("coordinator","worker.test"),false,2000,{},err))return;
        std::vector<uint8_t> bytes;
        for(int i=0;i<2;++i){gpu_codec::put(bytes,kGpuWireMagic,4);gpu_codec::put(bytes,GPU_STATUS,2);
            gpu_codec::put(bytes,ST_OK,2);gpu_codec::put(bytes,(uint64_t)0);}
        sent=tls.send_all(bytes.data(),bytes.size(),2000,{});sock::sleep_ms(300);
    });
    TlsSession tls;std::string err;require(tls.handshake(fd[1],cert.config("worker","coordinator.test"),true,2000,{},err),err);
    WireFrame frame;const bool first=gpu_codec::receive(fd[1],frame,kGpuFrameMax,500,{},err,500,&tls);
    const bool second=gpu_codec::receive(fd[1],frame,kGpuFrameMax,500,{},err,50,&tls);
    sender.join();check(sent&&first&&second&&frame.msg==GPU_STATUS,"TLS buffered second frame bypasses socket readiness wait");
    ::close(fd[0]);::close(fd[1]);
}
void tls_lifecycle(const Fixture& f,const GpuPlan& p,bool nodelay) {
    TlsFixture certificates;
    auto options=options_for(f,p,"cache-tls-"+std::to_string(nodelay));options.accepted_tcp_nodelay=nodelay;options.tls=certificates.config("worker","coordinator.test");
    GpuService service(options,std::make_unique<FakeExecutor>());
    std::string err;int fetches=0;auto open=open_for(f,p,service.port,fetches);
    open.tls=certificates.config("coordinator","worker.test");
    GpuWorkerTier tier;require(tier.open(open,err),err);const auto session=tier.stats().session;
    auto first=run_tier(tier,p),second=run_tier(tier,p);GpuCounters counters;
    require(tier.status(counters,err),err);
    check(first==second && counters.requests==2 && tier.stats().session==session,
          "mTLS actual GPU setup/upload/ready/status and two requests retain one session");
    check(fetches==2 && tier.stats().setup_network_bytes==p.assigned_bytes(),"mTLS upload/cache accounting unchanged");
    require(tier.close_orderly(err),err);GpuWorkerTier warm;fetches=0;require(warm.open(open,err),err);
    check(fetches==0 && warm.stats().cache_reused_bytes==p.assigned_bytes(),"mTLS reconnect reuses verified cache");
    require(warm.status(counters,err),err);check(counters.failures==1,"orderly prior disconnect is still counted and service remains available");
    warm.close();GpuWorkerTier wrong;open.tls.peer_dns="wrong.test";
    check(!wrong.open(open,err)&&wrong.dead()&&!wrong.ready(),"GPU mTLS identity failure latches dead without plaintext retry");
    auto plain=open;plain.tls={};plain.deadline_ms=200;
    GpuWorkerTier plaintext;check(!plaintext.open(plain,err)&&plaintext.dead(),"plaintext client refused by mTLS GPU service");
    GpuService plain_service(options_for(f,p,"cache-plain-refusal-"+std::to_string(nodelay)),std::make_unique<FakeExecutor>());
    auto tls_to_plain=open;tls_to_plain.port=plain_service.port;tls_to_plain.tls.peer_dns="worker.test";
    GpuWorkerTier encrypted;check(!encrypted.open(tls_to_plain,err)&&encrypted.dead(),"mTLS client refuses plaintext GPU service without fallback");
}
#endif
#ifdef STRATA_GPU_TEST_CUDA
void cuda_checks(const Fixture& f,const GpuPlan& p){
    std::vector<float> network;EntryRowsReply network_rows;std::string err;
    {
        GpuService service(options_for(f,p,"cache-cuda"),create_cuda_gpu_executor(0));
        int n=0;GpuWorkerTier tier;require(tier.open(open_for(f,p,service.port,n),err),err);network=run_tier(tier,p);
        GpuCounters counters;require(tier.status(counters,err),err);check(counters.entries==4&&counters.requests==1,"actual CUDA worker executed all assigned entries");
        std::string report;tier.report(report);std::printf("%s\n",report.c_str());tier.close();
    }
    {
        // The successful real CUDA case above establishes a usable device first. This
        // impossible reserve reaches the actual allocator's available-VRAM refusal.
        auto options=options_for(f,p,"cache-cuda-overcapacity");options.runtime_reserve=UINT64_MAX;
        GpuService service(options,create_cuda_gpu_executor(0));GpuWorkerTier tier;int n=0;
        check(!tier.open(open_for(f,p,service.port,n),err)&&tier.dead()&&!tier.ready(),
            "actual CUDA available-VRAM refusal never exposes READY");
        service.stop();const auto& counters=service.service.counters();
        check(counters.gpu_allocation_count==0&&counters.gpu_weight_bytes==0&&counters.gpu_allocated_bytes==0,
            "actual overcapacity refuses before weight/workspace allocation");
    }
    {
        GpuService service(options_for(f,p,"cache-cuda-rows"),create_cuda_gpu_executor(0));
        int n=0;GpuWorkerTier tier;auto option=open_for(f,p,service.port,n);option.entry_rows=true;
        require(tier.open(option,err),err);auto r=request_for(p);std::vector<int32_t> ids(r.ids.begin(),r.ids.end());
        require(tier.post(0,r.activations.data(),ids.data(),r.weights.data(),r.mask.data(),r.T,r.K,err)&&tier.finish(err),err);
        network_rows=tier.entry_rows();tier.close();
    }
    // Same device/kernel path as reference, independently loaded and called without wire transport.
    const auto options=options_for(f,p,"cache-direct");GpuCache cache;std::vector<uint8_t> missing;
    require(cache.open(options.cache_root,options.cache_budget,err)&&cache.prepare(p,missing,err),err);
    for(const auto& a:p.assets){const auto& b=f.blobs[(size_t)a.layer*2+a.expert];require(cache.publish(a,b.data(),b.size(),err),err);}
    auto executor=create_cuda_gpu_executor(0);GpuCounters counters;require(executor->prepare(p,256ull<<20,cache,counters,err),err);
    auto request=request_for(p);InferReply reply;reply.session=request.session;reply.sequence=request.sequence;
    require(executor->infer(p,request,reply,err),err);
    EntryRowsReply rows;require(executor->infer_rows(p,request,rows,err),err);executor->release(counters);
    check(rows.indices==network_rows.indices && rows.rows.size()==network_rows.rows.size() &&
          !std::memcmp(rows.rows.data(),network_rows.rows.data(),rows.rows.size()*4),"actual CUDA row bytes survive transport exactly");
    std::vector<float> reference(reply.partials.size(),0.f);
    for(size_t j=0;j<rows.indices.size();++j) {
        const size_t i=rows.indices[j];
        for(size_t h=0;h<p.n_embd;++h) {
            const float product=request.weights[i]*rows.rows[j*p.n_embd+h];
            reference[(i/request.K)*p.n_embd+h]+=product;
        }
    }
    check(!std::memcmp(reference.data(),reply.partials.data(),reference.size()*4),"direct CUDA weighted result equals independent ordered row reference");
    compare(network,reply.partials,1e-6,"network CUDA versus direct grouped CUDA",1e-5);
    compare(network,float_reference(f,request),3e-2,"network CUDA versus dequantized float reference");
}
#endif
int peer(const Fixture& f,const GpuPlan& p,const std::string& host,uint16_t port){
    GpuAssetSource source;std::string err;require(source.open(f.root.string(),f.m,p,err),err);
    GpuWorkerTier::Open o;o.plan=p;o.host=host;o.port=port;o.deadline_ms=5000;o.setup_ms=60000;
    o.fetch=[&](int64_t layer,int64_t expert,uint8_t* dst,uint32_t bytes){
        const auto* a=p.find((int)layer,(int)expert);
        return a&&source.read_into(*a,dst,bytes,err);};
    GpuWorkerTier tier;require(tier.open(o,err),err);const auto output=run_tier(tier,p);
    compare(output,float_reference(f,request_for(p)),3e-2,"remote CUDA versus dequantized float reference");
    GpuCounters c;require(tier.status(c,err),err);check(c.entries>=4&&c.requests>=1,"remote worker execution counters positive");
    std::string report;tier.report(report);std::printf("%s source_payload_read=%llu\n",report.c_str(),(unsigned long long)source.stats().source_read_bytes);
    check(source.stats().source_read_bytes==tier.stats().setup_network_bytes,"source expert payload reads equal missing-only transfer bytes");return failures?1:0;
}
}
int main(int argc,char** argv){
    try{
        Fixture fixture;const bool addition=argc==4&&std::string(argv[1])=="--emit-plan"&&std::string(argv[3])=="--addition";
        auto plan=plan_for(fixture,addition);std::string err;
        if(argc>=3&&std::string(argv[1])=="--emit-plan"){
            require(argc==3||addition,"--emit-plan FILE [--addition]");require(write_gpu_plan(argv[2],plan,err),err);
            std::printf("synthetic plan=%s assigned=%llu experts=%zu\n",argv[2],(unsigned long long)plan.assigned_bytes(),plan.assets.size());return 0;}
        if(argc==5&&std::string(argv[1])=="--peer"){
            char* end=nullptr;const auto port=std::strtoul(argv[3],&end,10);require(*argv[3]&&!*end&&port>0&&port<=65535,"peer numeric port");
            require(read_gpu_plan(argv[4],plan,err),err);return peer(fixture,plan,argv[2],(uint16_t)port);}
#ifdef STRATA_GPU_TEST_CUDA
        if(argc==2&&std::string(argv[1])=="--cuda"){cuda_checks(fixture,plan);return failures?1:0;}
#endif
        require(argc==1,"usage: gpu_worker_test [--emit-plan FILE [--addition] | --peer HOST PORT FILE | --cuda]");
#ifdef STRATA_TEST_TLS
        tls_pending_frame();tls_lifecycle(fixture,plan,false);tls_lifecycle(fixture,plan,true);
#endif
#ifndef STRATA_TEST_TLS
        TlsConfig disabled;disabled.ca_file="unused-test-path";
        check(!disabled.validate(err),"TLS request fails closed in TLS-disabled build");
        TlsSession unavailable;TransportRead detail;char byte=0;
        check(!unavailable.recv_all(&byte,1,100,{},&detail)&&detail.outcome==TransportOutcome::tls_error,"disabled TLS outcome initialized");
        check(!unavailable.recv_all(&byte,1,0,{},&detail)&&detail.outcome==TransportOutcome::timeout,"disabled TLS deadline outcome initialized");
        check(!unavailable.send_close_notify(100,{},err),"TLS-disabled close notification refuses");
#endif
        listener_checks(fixture,plan);lifecycle(fixture,plan);
#ifndef _WIN32
        transport_diagnostics();failure_log_checks(fixture,plan);
#endif
        return failures?1:0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL gpu fixture: %s\n",e.what());return 1;}
}
