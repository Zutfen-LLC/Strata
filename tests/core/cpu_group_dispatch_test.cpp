// Real dispatcher, CPU group, local manifest/services and mapped CUDA merge. Synthetic weights only.
#include "../worker/local_fixture.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/worker/cpu_group.hpp"
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
}
int main(){try{exercise(1);exercise(2);}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());++failures;}
    return failures?1:0;}
