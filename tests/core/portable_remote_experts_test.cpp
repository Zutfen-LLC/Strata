#include "portable_remote_experts.hpp"
#include "portable_model_preflight.hpp"
#include "local_fixture.hpp"
#include "strata/worker/gpu_worker.hpp"
#include <atomic>
#ifdef STRATA_TEST_TLS
#include "tls_fixture.hpp"
#endif
using strata::core::PortableRoute;
using strata::core::PortableExpertRows;
using strata::core::portable_detail::RemoteExperts;
namespace {
struct Fake final : GpuExecutor {
    int mode=0;std::atomic<int>* prepares;
    explicit Fake(std::atomic<int>& n,int m=0):mode(m),prepares(&n){}
    bool native_q8()const override{return true;}
    bool supports_rows()const override{return true;}
    bool prepare(const GpuPlan& p,uint64_t,GpuCache& cache,GpuCounters& c,std::string& e)override{
        ++*prepares;std::vector<uint8_t> bytes;for(const auto& asset:p.assets)if(!cache.read(asset,bytes,e))return false;
        c.gpu_weight_bytes=p.assigned_bytes();c.gpu_allocated_bytes=p.assigned_bytes();c.gpu_allocation_count=1;return true;
    }
    bool infer(const GpuPlan&,const InferRequest&,InferReply&,std::string&)override{return false;}
    bool infer_rows(const GpuPlan& p,const InferRequest& r,EntryRowsReply& out,std::string&)override{
        for(size_t i=0;i<r.mask.size();++i)if(r.mask[i]){
            out.indices.push_back(uint16_t(i));out.rows.insert(out.rows.end(),p.n_embd,float(r.ids[i]+10));
        }
        if(mode==1 && !out.indices.empty()){out.indices.pop_back();out.rows.resize(out.indices.size()*p.n_embd);}
        if(mode==2 && out.indices.size()>1)out.indices[1]=out.indices[0];
        if(mode==3 && !out.indices.empty())out.indices[0]=0;
        if(mode==4 && !out.rows.empty())out.rows[0]=std::numeric_limits<float>::quiet_NaN();
        if(mode==5)return false;
        if(mode==6)std::this_thread::sleep_for(std::chrono::milliseconds(400));
        return true;
    }
    void release(GpuCounters& c)override{c.gpu_allocated_bytes=c.gpu_weight_bytes=c.gpu_allocation_count=0;}
};
struct RemoteService {
    GpuWorkerService service;uint16_t port=0;std::thread thread;
    RemoteService(GpuWorkerOptions o,std::unique_ptr<GpuExecutor> e):service(std::move(o),std::move(e)){
        std::string err;require(service.listen(port,err),err);thread=std::thread([this]{service.run();});
    }
    ~RemoteService(){service.stop();if(thread.joinable())thread.join();}
};
// Exercise the real FFN graph around the real worker adapter. The independent
// callback returns the same unweighted rows directly in original slot order.
void check_ffn_dispatch(RemoteExperts& provider) {
    using namespace strata::core;
    PortableFfnGeometry geometry{2560,4,1,32,12,10,1e-6f};
    const char* names[]={"hc_ffn_norm.weight","hc_ffn_down.weight","hc_ffn_up.weight","hc_ffn_inject.weight",
        "ffn_gate_inp.weight","ffn_gate_inp_shexp.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
    const std::vector<std::vector<int64_t>> shapes={{10240},{10240,1},{1,10240},{10240,4},{2560,12},{2560},{2560,32},{2560,32},{32,2560}};
    std::vector<std::vector<float>> data(9);std::vector<PortableTensorView> views;
    for(size_t i=0;i<9;++i){size_t n=1;for(auto v:shapes[i])n*=size_t(v);data[i].assign(n,0);}
    std::fill(data[0].begin(),data[0].end(),1.f);
    for(size_t e=0;e<12;++e)data[4][e*2560]=float(11-e)*0.125f;
    data[6][0]=data[7][0]=1.f;for(size_t h=0;h<2560;++h)data[8][h*32]=1.f;
    for(size_t i=0;i<9;++i)views.push_back({std::string("blk.0.")+names[i],shapes[i],0,data[i].data(),data[i].size()*sizeof(float)});
    std::string error; PortableFfnOptions options;
    auto actual=PortableFfn::create(geometry,0,views,options,error);require(bool(actual),error);
    auto expected=PortableFfn::create(geometry,0,views,options,error);require(bool(expected),error);
    std::vector<float> initial(10240,0.5f);
    for(int attempt=0;attempt<2;++attempt) {
        require(actual->reset(initial,error)&&expected->reset(initial,error),error);
        PortableRoute observed,reference;
        require(actual->step([&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e){return provider.run(l,r,rows,e);},observed,error),error);
        require(expected->step([](int64_t,const PortableRoute& r,std::vector<float>& rows,std::string&){
            rows.clear();for(auto id:r.ids)rows.insert(rows.end(),2560,float(id<2?id+10:id+100));return true;
        },reference,error),error);
        check(observed.ids==reference.ids && observed.ids[0]==0 && observed.ids[1]==1 && observed.weights==reference.weights &&
              observed.weights[0]!=observed.weights[1],"actual FFN routes retain nontrivial weights and slots");
        check(actual->residual()==expected->residual(),"actual FFN combines unweighted remote rows exactly once");
    }
}
void run(){
    Fixture f(12);std::string error;GpuPlan plan;require(make_gpu_plan(f.m,{{0,0},{0,1}},plan,error),error);
    GpuWorkerOptions opts;opts.plan=plan;opts.vulkan=opts.native_q8=true;opts.runtime_reserve=0;
    opts.cache_root=(f.root/"remote-adapter-cache").string();fs::create_directory(opts.cache_root);opts.cache_budget=64ull<<20;
#ifdef STRATA_TEST_TLS
    TlsFixture tls;opts.tls=tls.config("worker","coordinator.test");
#endif
    std::atomic<int> prepares{0};RemoteService server(opts,std::make_unique<Fake>(prepares));
    GpuWorkerTier::Open open;open.port=server.port;open.plan=plan;open.vulkan=open.native_q8=open.entry_rows=true;
#ifdef STRATA_TEST_TLS
    open.tls=tls.config("coordinator","worker.test");
#endif
    open.fetch=[&](int64_t l,int64_t e,uint8_t* dst,uint32_t n){const auto& b=f.blobs[size_t(l)*f.n_expert+size_t(e)];if(n!=b.size())return false;std::memcpy(dst,b.data(),n);return true;};
    int complement_calls=0,complement_entries=0;bool complement_fail=false;
    auto complement=[&](int64_t,const std::vector<PortableRoute>& routes,const std::vector<uint8_t>& mask,std::vector<float>& out,std::string&){
        ++complement_calls;if(complement_fail)return false;const auto K=routes[0].ids.size();out.assign(mask.size()*plan.n_embd,0);
        for(size_t i=0;i<mask.size();++i)if(mask[i]) {
            ++complement_entries;std::fill_n(out.begin()+i*plan.n_embd,plan.n_embd,float(routes[i/K].ids[i%K]+100));
        }
        return true;};
    bool source_ok=true;int guards=0,fail_guard=-1;auto guard=[&](std::string& e){++guards;const bool ok=source_ok&&guards!=fail_guard;if(!ok)e="source changed";return ok;};
    RemoteExperts bad;auto invalid=open;invalid.native_q8=false;check(!bad.open(invalid,plan,{},10,complement,guard,error)&&prepares==0,"hybrid numerics refused before setup");
    RemoteExperts overlap;check(!overlap.open(open,plan,{plan},10,complement,guard,error)&&prepares==0,"overlapping ownership refused before setup");
    GpuPlan other;require(make_gpu_plan(f.m,{{0,0}},other,error),error);
    RemoteExperts wrong_source;check(!wrong_source.open(open,other,{},10,complement,guard,error)&&prepares==0,"plan/source mismatch refused before setup");
#ifdef STRATA_TEST_TLS
    auto wrong_san=open;wrong_san.tls.peer_dns="wrong.test";RemoteExperts wrong_peer;
    check(!wrong_peer.open(wrong_san,plan,{},10,complement,guard,error),"wrong SAN refuses without fallback");
#endif
    source_ok=false;RemoteExperts stale;check(!stale.open(open,plan,{},10,complement,guard,error)&&prepares==0,"changed source refused before setup");source_ok=true;
    RemoteExperts provider;require(provider.open(open,plan,{},10,complement,guard,error),error);
    check(provider.usable(),"new remote provider is usable");
    PortableRoute route;route.activation.assign(plan.n_embd,0.125f);route.ids={9,0,1,2,3,4,5,6,7,8};route.weights={0.03f,0.17f,0.11f,0.09f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f};
    PortableExpertRows dispatch=[&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e){return provider.run(l,r,rows,e);};
    std::vector<float> rows;require(dispatch(0,route,rows,error),error);
    check(rows.size()==10*plan.n_embd&&rows[0]==109.f&&rows[plan.n_embd]==10.f&&rows[2*plan.n_embd]==11.f,"original slots and unweighted rows preserved");
    check(complement_calls==1&&complement_entries==8,"only unowned entry computed by complement");
    route.ids={2,3,4,5,6,7,8,9,10,11};route.weights.assign(10,0.1f);require(dispatch(0,route,rows,error),error);
    check(complement_calls==2&&complement_entries==18&&rows[0]==102.f,"zero remote hit uses complement");
    route.ids={2};route.weights={0.3f};rows.assign(1,9);
    check(!dispatch(0,route,rows,error)&&rows.empty(),"wrong route width refused");
    route.ids={0,0,2,3,4,5,6,7,8,9};route.weights.assign(10,0.1f);rows.assign(1,9);
    check(!dispatch(0,route,rows,error)&&rows.empty(),"duplicate input identity refused");
    route.ids={2,3,4,5,6,7,8,9,10,11};route.weights.assign(10,0.1f);route.activation[0]=std::numeric_limits<float>::infinity();rows.assign(1,9);
    check(!dispatch(0,route,rows,error)&&rows.empty(),"nonfinite activation refused");route.activation[0]=0.125f;
    route.weights[0]=std::numeric_limits<float>::quiet_NaN();rows.assign(1,9);
    check(!dispatch(0,route,rows,error)&&rows.empty(),"nonfinite input fails atomically");
    GpuCounters counters;require(provider.status(counters,error),error);
    check(provider.stats().requests==1 && provider.stats().entries==2 && provider.complement_entries()==18 && counters.requests==1 && counters.entries==2,
          "remote and complement counters retained");
    route.ids={0,1,2,3,4,5,6,7,8,9};route.weights.assign(10,0.1f);fail_guard=guards+2;rows.assign(1,9);
    check(!dispatch(0,route,rows,error)&&rows.empty()&&error=="source changed","source change after remote reply prevents publication");
    fail_guard=-1;route.ids={2,0,1,3,4,5,6,7,8,9};route.weights.assign(10,0.1f);complement_fail=true;rows.assign(1,9);
    check(!dispatch(0,route,rows,error)&&rows.empty()&&!error.empty(),"complement failure does not publish partial rows");
    complement_fail=false;
    check_ffn_dispatch(provider);
    require(provider.close_orderly(error),error);
    check(!provider.usable(),"explicitly closed provider is unusable");
    RemoteExperts next;require(next.open(open,plan,{},10,complement,guard,error),error);
    require(next.close_orderly(error),error);
    check(true,"orderly close leaves shared worker available");
    {
        std::vector<std::pair<uint16_t,uint16_t>> owned;
        for(uint16_t i=0;i<10;++i)owned.emplace_back(0,i);
        GpuPlan all_plan;require(make_gpu_plan(f.m,owned,all_plan,error),error);
        auto all_options=opts;all_options.plan=all_plan;all_options.cache_root=(f.root/"all-remote").string();
        fs::create_directory(all_options.cache_root);std::atomic<int> all_prepares{0};
        RemoteService all_server(all_options,std::make_unique<Fake>(all_prepares));
        auto all_open=open;all_open.port=all_server.port;all_open.plan=all_plan;
        RemoteExperts all;require(all.open(all_open,all_plan,{},10,complement,guard,error),error);
        route.ids={9,0,8,1,7,2,6,3,5,4};route.weights.assign(10,0.1f);
        const int before=complement_calls;
        for(int i=0;i<2;++i) {
            require(all.run(0,route,rows,error),error);
            for(size_t k=0;k<10;++k)check(rows[k*plan.n_embd]==float(route.ids[k]+10),"all-remote original slot row");
        }
        check(complement_calls==before && all.stats().requests==2 && all.stats().entries==20,
              "repeated all-remote requests never execute CPU complement");
        require(all.close_orderly(error),error);
    }
    for(int mode=1;mode<=6;++mode){
        auto options=opts;options.cache_root=(f.root/("remote-fault-"+std::to_string(mode))).string();fs::create_directory(options.cache_root);
        std::atomic<int> observed{0};RemoteService faulty(options,std::make_unique<Fake>(observed,mode));
        auto request=open;request.port=faulty.port;if(mode==6)request.deadline_ms=100;
        RemoteExperts candidate;require(candidate.open(request,plan,{},10,complement,guard,error),error);
        route.ids={9,0,1,2,3,4,5,6,7,8};route.weights={0.03f,0.17f,0.11f,0.09f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f};rows.assign(1,99);const int before=complement_calls;
        check(!candidate.run(0,route,rows,error)&&rows.empty()&&complement_calls==before,"remote failure is atomic and never falls back");
        check(!candidate.usable(),"failed remote provider is unusable");
        rows.assign(1,99);check(!candidate.run(0,route,rows,error)&&rows.empty(),"dead remote peer cannot silently recover");
    }
    strata::core::PortableModelOptions model;model.remote_experts=open;
    strata::core::portable_detail::validate_model_modes(model,0);
    model.prompt_chunk_size=2;bool refused=false;try{strata::core::portable_detail::validate_model_modes(model,0);}catch(...){refused=true;}
    check(refused,"prompt chunks refused by remote preflight");
    model.prompt_chunk_size=0;model.expert_devices.emplace_back();refused=false;
    try{strata::core::portable_detail::validate_model_modes(model,1);}catch(...){refused=true;}
    check(refused,"mixed local/remote requires native Q8 local method");
    model.expert_native_q8_observation="receipt";strata::core::portable_detail::validate_model_modes(model,1);
    model.expert_devices.clear();model.expert_native_q8_observation.clear();
    model.remote_experts->native_q8=false;refused=false;try{strata::core::portable_detail::validate_model_modes(model,0);}catch(...){refused=true;}
    check(refused,"wrong numerical method refused by remote preflight");
}
}
int main(){try{run();}catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}return failures?1:0;}
