// Real dispatcher, CPU group, local manifest/services and mapped CUDA merge. Synthetic weights only.
#include "../worker/local_fixture.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/worker/cpu_group.hpp"
#ifdef STRATA_NETWORK_CUDA_WORKER
#include "strata/worker/gpu_worker.hpp"
#include "strata/worker/gpu_group.hpp"
#endif
#include <algorithm>
#include <cuda_runtime.h>
namespace {
struct NoRead final : strata::core::ExpertSource {
    int reads=0;
    const uint8_t* blob(int64_t,int64_t) override {++reads;return nullptr;}
};
struct Device {
    void* p=nullptr;
    explicit Device(size_t n) {require(cudaMalloc(&p,n)==cudaSuccess,"cudaMalloc");}
    ~Device(){if(p)cudaFree(p);}
};
void exercise(int count) {
    Fixture f;auto settings=f.options();settings.residency_forever=true;
    Service a(settings),b(settings);std::string err;
    std::vector<ExpertWorkerGroup::Open> options;
    for(int p=0;p<count;++p) {
        ExpertWorkerGroup::Open o;o.port=p?b.port:a.port;o.local=&f.m;o.resident=true;
        o.n_layers=2;o.n_expert=2;o.n_embd=2560;o.n_ff=640;o.gu_type=o.d_type=8;
        o.partition=count==1?std::vector<uint8_t>{3,3}:std::vector<uint8_t>{(uint8_t)(1<<p),(uint8_t)(1<<p)};
        options.push_back(o);
    }
    ExpertWorkerGroup group;require(group.open(options,{3,3},err),err);
    strata::core::RemoteExpertOpt opt;require(opt.init(err),err);
    strata::kernels::cpu::ExpertPool pool(1);NoRead source;
    // Two synthetic experts repeat five times at K10: T1 stays within kMaxT uses per expert.
    // Alternating T8/T1/T2/T1 uses K2 in the CPU fixture; do not relax the production use bound.
    for(int layer=0;layer<2;++layer) {
        constexpr int T=1;
        constexpr int K=10,H=2560; // production mapped merge has ten routing ranks
        std::vector<int32_t> ids(T*K);std::vector<float> weights(T*K),x(T*H),rows(T*K*H,-7.f);
        for(int i=0;i<T*K;++i){ids[i]=i%2;weights[i]=i%2?-.3f:.7f;}
        for(size_t i=0;i<x.size();++i)x[i]=.1f*(int(i%11)-5);
        strata::core::ExpertDispatch d;d.pool=&pool;d.src=&source;d.n_expert=2;
        d.cpu_group=&group;d.remote_opt=&opt;d.layers=layer;
        opt.begin(weights.data(),0,T);
        strata::core::expert_pool_dispatch_multi(d,x.data(),ids.data(),T,K,rows.data());opt.end();
        require(!d.failed,d.fail?d.fail:"group dispatch failed");
        check(source.reads==0,"group experts never fall back to local source reads");
        check(std::all_of(rows.begin(),rows.end(),[](float v){return v==0.f;}),"worker per-entry rows zeroed");
        std::vector<const uint8_t*> pointers(T*K);
        for(int i=0;i<T*K;++i)pointers[i]=f.blobs[layer*2+ids[i]].data();
        std::vector<float> expected;
        for(int p=0;p<count;++p) {
            std::vector<uint8_t> mask(T*K);for(int i=0;i<T*K;++i)mask[i]=count==1||ids[i]==p;
            std::vector<float> partial;
            require(ExpertWorkerService::direct_native_partial(f.fmt,T,K,mask.data(),weights.data(),x.data(),pointers.data(),partial,err),err);
            if(!p)expected=partial;else for(size_t i=0;i<expected.size();++i)expected[i]=expected[i]+partial[i];
        }
        check(group.served()&&!std::memcmp(group.partials(),expected.data(),expected.size()*4),"dispatch preserves grouped reference bits");
        // The real mapped merge carries the group exactly once into GPU output.
        Device output(expected.size()*4);std::vector<float> combined(expected.size(),0.f);
        require(cudaMemcpy(output.p,combined.data(),combined.size()*4,cudaMemcpyHostToDevice)==cudaSuccess,"upload zero output");
        opt.combine(static_cast<float*>(output.p),nullptr,0,T,nullptr,0,nullptr);
        require(cudaMemcpy(combined.data(),output.p,combined.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess,"download merged output");
        check(!std::memcmp(combined.data(),expected.data(),expected.size()*4),"mapped GPU combine adds grouped partial once");
    }
    group.close();strata::core::ExpertDispatch refused;refused.cpu_group=&group;
    float poison=99.f;strata::core::expert_pool_dispatch_multi(refused,nullptr,nullptr,1,1,&poison);
    check(refused.failed&&poison==99.f,"unavailable group refused before any output or source access");
}
#ifdef STRATA_NETWORK_CUDA_WORKER
class LossRows final : public GpuExecutor {
public:
    std::unique_ptr<GpuExecutor> real=create_cuda_gpu_executor(0);
    std::atomic<bool> lose{false};
    bool supports_rows() const override {return true;}
    bool prepare(const GpuPlan& p,uint64_t reserve,GpuCache& c,GpuCounters& stats,std::string& err) override {
        return real->prepare(p,reserve,c,stats,err);
    }
    bool infer(const GpuPlan& p,const InferRequest& r,InferReply& out,std::string& err) override {return real->infer(p,r,out,err);}
    bool infer_rows(const GpuPlan& p,const InferRequest& r,EntryRowsReply& out,std::string& err) override {
        if(lose.load()){err="controlled GPU loss after CPU post";return false;}
        return real->infer_rows(p,r,out,err);
    }
    void release(GpuCounters& c) override {real->release(c);}
};
void mixed() {
    Fixture f;auto settings=f.options();settings.residency_forever=true;settings.deadline_ms=200;
    Service cpu(settings);std::string err;
    ExpertWorkerGroup::Open option;option.port=cpu.port;option.local=&f.m;option.resident=true;option.entry_rows=true;
    option.n_layers=2;option.n_expert=2;option.n_embd=2560;option.n_ff=640;option.partition={3,3};
    {ExpertWorkerGroup warm;require(warm.open({option},{3,3},err),err);warm.close();}
    auto cache_status=[&] {
        ExpertWorkerClient client;require(client.connect_host("127.0.0.1",cpu.port,3000,err),err);
        WorkerCacheInfo c;require(client.cache_status(c,err),err);client.close();return c;
    };
    const auto before=cache_status();
    option.resident_reuse_only=true;ExpertWorkerGroup group;
    require(group.open({option},{3,3},err,{2,2}),err);
    strata::core::ExpertDispatch missing_gpu;missing_gpu.cpu_group=&group;float poison=99.f;
    strata::core::expert_pool_dispatch_multi(missing_gpu,nullptr,nullptr,1,10,&poison);
    check(missing_gpu.failed&&poison==99.f,"delegated CPU group cannot fall back to a CPU partial without its GPU group");
    check(group.peer_stats(0).cache_hit && group.peer_stats(0).reused_bytes==before.resident_bytes,"mixed fixture reuses full resident ownership");
    GpuWorkerOptions gpu_options;gpu_options.plaintext_test=true;require(make_gpu_plan(f.m,{{0,1},{1,1}},gpu_options.plan,err),err);
    gpu_options.cache_root=(f.root/"gpu-cache").string();fs::create_directory(gpu_options.cache_root);
    auto executor=std::make_unique<LossRows>();auto* loss=executor.get();
    GpuWorkerService gpu_service(gpu_options,std::move(executor));uint16_t port=0;
    require(gpu_service.listen(port,err),err);std::thread gpu_thread([&]{gpu_service.run();});
    struct Stop {GpuWorkerService& service;std::thread& thread;~Stop(){service.stop();if(thread.joinable())thread.join();}} stop{gpu_service,gpu_thread};
    GpuWorkerTier::Open go;go.plaintext_test=true;go.port=port;go.plan=gpu_options.plan;go.entry_rows=true;
    go.cancelled=[&]{return group.dead();};
    go.fetch=[&](int64_t l,int64_t e,uint8_t* dst,uint32_t bytes) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        if(bytes!=f.blobs[(size_t)l*2+e].size())return false;
        std::memcpy(dst,f.blobs[(size_t)l*2+e].data(),bytes);return true;
    };
    GpuWorkerGroup gpu;require(gpu.open({go},err)&&group.probe_idle(err),err);
    check(group.peer_stats(0).requests==0,"existing CPU keeper survives delayed GPU acquisition without inference");
    constexpr int T=1,K=10,H=2560;
    strata::core::RemoteExpertOpt opt;require(opt.init(err),err);NoRead source;strata::kernels::cpu::ExpertPool pool(1);
    std::vector<int32_t> ids(K);std::vector<float> weights(K),x(H),rows(K*H),expected(H),combined(H);
    for(int i=0;i<K;++i)weights[i]=i%2?-.3f:.7f;
    for(int h=0;h<H;++h)x[h]=.01f*(int(h%13)-6);
    Device device(H*4);
    auto dispatch=[&](int layer) {
        strata::core::ExpertDispatch d;d.pool=&pool;d.src=&source;d.n_expert=2;d.layers=layer;
        d.cpu_group=&group;d.gpu_worker=&gpu;d.remote_opt=&opt;
        opt.begin(weights.data(),0,T);
        strata::core::expert_pool_dispatch_multi(d,x.data(),ids.data(),T,K,rows.data());opt.end();
        return d.failed;
    };
    auto download=[&] {
        require(cudaMemset(device.p,0,H*4)==cudaSuccess,"zero mixed merge output");
        opt.combine(static_cast<float*>(device.p),nullptr,0,T,nullptr,0,nullptr);
        require(cudaMemcpy(combined.data(),device.p,H*4,cudaMemcpyDeviceToHost)==cudaSuccess,"read mixed merge");
    };
    // Sparse repeated interleaving, at most five uses per expert in a K10 window.
    for(int layer=0;layer<2;++layer) {
        for(int i=0;i<K;++i)ids[i]=i%2;
        require(!dispatch(layer),"ordered CPU/CUDA dispatcher succeeds");
        check(source.reads==0&&group.rows_ready()&&!group.partials(),"mixed dispatch has no fallback or separate CPU partial");
        check(std::all_of(rows.begin(),rows.end(),[](float v){return v==0.f;}),"network entries stay zero in outer local merge");
        const auto& cpu_rows=group.entry_rows();const auto& gpu_rows=gpu.entry_rows();
        std::vector<float> dense(cpu_rows);
        for(size_t j=0;j<gpu_rows.indices.size();++j)std::copy_n(gpu_rows.rows.data()+j*H,H,dense.data()+gpu_rows.indices[j]*H);
        expected.assign(H,0.f);
        for(int i=0;i<K;++i)for(int h=0;h<H;++h) {const float product=weights[i]*dense[i*H+h];expected[h]=expected[h]+product;}
        download();check(!std::memcmp(combined.data(),expected.data(),H*4),"real mixed dispatcher and mapped combine equal independent original-order row reference");
    }
    for(int i=0;i<K;++i)ids[i]=i%2;
    WorkerStatusInfo prior,after;require(group.peer_status(0,prior,err),err);loss->lose.store(true);
    check(dispatch(0)&&gpu.dead()&&!group.posted(),"mixed GPU failure drains submitted CPU and latches failure");
    require(group.peer_status(0,after,err),err);check(after.requests==prior.requests+1,"mixed failure has no CPU fallback or retry");
    download();check(std::all_of(combined.begin(),combined.end(),[](float v){return v==0.f;}),"mixed failure commits no partial");
    gpu.close();group.close();const auto retained=cache_status();
    check(retained.complete&&retained.resident_bytes==before.resident_bytes&&retained.loaded_bytes==before.loaded_bytes&&
          retained.arena_allocations==before.arena_allocations&&retained.unloads==before.unloads&&retained.expirations==before.expirations,
          "mixed dispatcher preserves full CPU backing without a reload");
}
#endif

}
int main(){try{exercise(1);exercise(2);
#ifdef STRATA_NETWORK_CUDA_WORKER
    mixed();
#endif
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());++failures;}
    return failures?1:0;}
