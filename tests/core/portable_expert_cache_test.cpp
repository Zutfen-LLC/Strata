#include "../../src/worker/vulkan_expert_cache.hpp"
#include "ggml-cpu.h"
#include "strata/worker/vulkan_executor.hpp"
#include "ggml-backend-impl.h"
#ifdef STRATA_EXPERT_CACHE_VULKAN_TEST
#include "ggml-vulkan.h"
#endif
#include <cstdio>
#include <cstring>
#include <map>
using namespace strata::worker::vulkan_detail;
namespace {
// Intercept the real CPU allocation seam: count backing allocations and exercise
// small backend limits and failures without a product policy or allocator switch.
struct AllocationProbe {
    ggml_backend_buffer_type_t buft=ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_i saved=buft->iface;
    std::map<ggml_backend_buffer_t,ggml_backend_buffer_i> live;
    size_t calls=0,bytes=0,limit=SIZE_MAX,fail=SIZE_MAX,uploads=0,fail_upload=SIZE_MAX,padding=0;
    static AllocationProbe* active;
    AllocationProbe() {
        require(!active,"nested allocation probe");active=this;
        buft->iface.get_max_size=[](ggml_backend_buffer_type_t){return active->limit;};
        buft->iface.get_alloc_size=[](ggml_backend_buffer_type_t b,const ggml_tensor* t){
            auto& p=*active;return (p.saved.get_alloc_size?p.saved.get_alloc_size(b,t):ggml_nbytes(t))+p.padding;
        };
        buft->iface.alloc_buffer=[](ggml_backend_buffer_type_t b,size_t n){
            auto& p=*active;++p.calls;if(p.calls==p.fail)return ggml_backend_buffer_t(nullptr);
            auto result=p.saved.alloc_buffer(b,n);
            if(result) {
                p.live[result]=result->iface;p.bytes+=ggml_backend_buffer_get_size(result);
                result->iface.free_buffer=[](ggml_backend_buffer_t buffer){
                    auto& p=*active;auto original=p.live.at(buffer);p.bytes-=ggml_backend_buffer_get_size(buffer);
                    p.live.erase(buffer);buffer->iface=original;if(original.free_buffer)original.free_buffer(buffer);
                };
                result->iface.set_tensor=[](ggml_backend_buffer_t buffer,ggml_tensor* tensor,const void* data,size_t off,size_t n){
                    auto& p=*active;if(++p.uploads==p.fail_upload)throw std::runtime_error("controlled upload failure");
                    p.live.at(buffer).set_tensor(buffer,tensor,data,off,n);
                };
            }
            return result;
        };
    }
    ~AllocationProbe(){if(!live.empty())std::abort();buft->iface=saved;active=nullptr;}
};
AllocationProbe* AllocationProbe::active=nullptr;
bool same(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float));
}
void equal(const Rows& a,const Rows& b) {
    require(same(a.gate,b.gate)&&same(a.up,b.up)&&same(a.hidden,b.hidden)&&same(a.out,b.out),"packed/private result bytes differ");
}
template<class F> void rejects(F f) {bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,"expected rejection");}
std::vector<uint8_t> fixture(int seed) {
    std::vector<uint8_t> b(5222400);uint32_t state=seed;
    for(size_t off=0;off<b.size();off+=34) {
        b[off]=0;b[off+1]=0x18;
        for(size_t j=2;j<34;++j){state=state*1664525u+1013904223u;b[off+j]=uint8_t(int((state>>16)%31)-15);}
    }
    return b;
}
void parity(ggml_backend_t backend,const cpu::NativeFmt& fmt) {
    const auto a=fixture(1),b=fixture(2),c=fixture(3);
    ExpertCache cache(backend,fmt,3,32u<<20);
    require(cache.allocation_count()==2,"three slots should share one weight block and one scratch buffer");
    cache.upload(0,fmt,a);cache.upload(1,fmt,b);cache.upload(2,fmt,c);
    rejects([&]{cache.upload(0,fmt,a);});rejects([&]{cache.run(3,std::vector<float>(H));});
    std::vector<float> x(H);for(int i=0;i<H;++i)x[i]=float(i%127-63)/128.f;
    std::vector<float> previous;
    for(size_t slot:{size_t(2),size_t(0),size_t(1),size_t(2),size_t(1),size_t(0)}) {
        ExpertGraph reference(backend,fmt,slot==0?a:slot==1?b:c,true);
        auto expected=reference.run(x),got=cache.run(slot,x);equal(expected,got);
        if(previous.empty())previous=got.out;else if(slot==0)require(!same(previous,got.out),"fixtures do not distinguish slots");
        for(auto& value:x)value=-value*.75f;
    }
    std::fill(x.begin(),x.end(),0.f);
    for(size_t slot=0;slot<3;++slot){ExpertGraph reference(backend,fmt,slot==0?a:slot==1?b:c,true);equal(reference.run(x),cache.run(slot,x));}
    std::printf("packed/private Q8 weights, F32 inputs and all result stages: byte identical; weight_storage=%zu scratch=%zu backing_allocations=%zu\n",
                cache.weight_storage_bytes(),cache.scratch_bytes(),cache.allocation_count());
}
}
int main(int argc,char** argv) {
    try {
        cpu::NativeFmt fmt;std::string error;require(cpu::native_fmt(8,8,H,F,fmt,error),error);
        const auto layout=expert_block_layout(5,256,600);
        require(layout.slots_per_block==2 && layout.blocks==3 && layout.bytes==1280,"partial final block");
        require(expert_aligned_size(257,256)==512,"slot alignment");
        rejects([]{expert_aligned_size(SIZE_MAX,256);});rejects([]{expert_aligned_size(1,3);});
        rejects([]{expert_block_layout(SIZE_MAX,256,SIZE_MAX);});rejects([]{expert_block_layout(1,256,255);});
        rejects([]{expert_block_layout(0,256,512);});
        Backend backend(ggml_backend_cpu_init(),ggml_backend_free);ggml_backend_cpu_set_n_threads(backend.get(),1);
#ifdef STRATA_EXPERT_CACHE_VULKAN_TEST
        if(argc>1) {
            require(argc==5 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device","use --device INDEX --expect-device SUBSTRING");
            const std::string index=argv[2];require(!index.empty()&&index.find_first_not_of("0123456789")==std::string::npos,"invalid device index");
            const auto device=std::stoull(index);require(device<size_t(ggml_backend_vk_get_device_count()),"device absent");
            require(std::string(argv[4]).size()>0,"expected device substring is empty");
            backend.reset(ggml_backend_vk_init(device));require(bool(backend),"Vulkan init failed");
            require(std::string(ggml_backend_dev_description(ggml_backend_get_device(backend.get()))).find(argv[4])!=std::string::npos,"unexpected Vulkan device");
            parity(backend.get(),fmt);return 0;
        }
#else
        require(argc==1,"CPU test accepts no arguments");(void)argv;
#endif
        parity(backend.get(),fmt);
        const auto blob=fixture(1);
        size_t owned=0,scratch=0;
        {
            AllocationProbe p;
            {
                ExpertCache cache(backend.get(),fmt,3,32u<<20);
                owned=cache.buffer_bytes();scratch=cache.scratch_bytes();
                require(p.calls==2 && p.live.size()==2 && p.bytes==owned,"actual backing allocation receipts");
                cache.upload(0,fmt,blob);cache.upload(1,fmt,blob);cache.upload(2,fmt,blob);
                auto x=std::vector<float>(H,.1f);cache.run(2,x);cache.run(0,x);
                require(p.calls==2 && p.bytes==owned,"execution must reuse backing storage");
            }
            require(p.live.empty() && p.bytes==0,"backing cleanup");
        }
        {
            AllocationProbe p;p.limit=2*size_t(fmt.bytes);
            ExpertCache cache(backend.get(),fmt,5,32u<<20);
            require(p.calls==4 && p.live.size()==4 && cache.allocation_count()==4 &&
                    cache.scratch_bytes()==scratch && cache.weight_storage_bytes()==5*size_t(fmt.bytes),"real partial final backend block and constant scratch");
            const auto last=fixture(5);
            for(size_t slot=0;slot<5;++slot)cache.upload(slot,fmt,slot==4?last:blob);
            auto x=std::vector<float>(H,.1f);
            {ExpertGraph reference(backend.get(),fmt,last,true);equal(reference.run(x),cache.run(4,x));}
            {ExpertGraph reference(backend.get(),fmt,blob,true);equal(reference.run(x),cache.run(0,x));}
            std::printf("five experts: actual weight_blocks=3 scratch_buffers=1 backing_bytes=%zu scratch=%zu\n",p.bytes,scratch);
        }
        {
            AllocationProbe p;
            { ExpertGraph a(backend.get(),fmt,blob,true),b(backend.get(),fmt,blob,true),c(backend.get(),fmt,blob,true);
              require(p.calls==3 && p.bytes==owned+2*scratch,"old per-expert backing reference");
              std::printf("private reference: allocations=%zu backing_bytes=%zu; packed: allocations=2 backing_bytes=%zu saved_scratch=%zu\n",p.calls,p.bytes,owned,2*scratch); }
        }
        {
            AllocationProbe p;p.padding=256;
            ExpertCache cache(backend.get(),fmt,3,owned+8192);
            require(cache.weight_storage_bytes()==3*(size_t(fmt.bytes)+3*256) && cache.scratch_bytes()==scratch+5*256 &&
                    p.bytes==cache.buffer_bytes(),"weight padding accounted separately from once-owned scratch");
            for(size_t slot=0;slot<3;++slot)cache.upload(slot,fmt,blob);
            ExpertGraph reference(backend.get(),fmt,blob,true);
            equal(reference.run(std::vector<float>(H,.1f)),cache.run(2,std::vector<float>(H,.1f)));
        }
        for(size_t budget:{owned-1,owned}) {
            AllocationProbe p;
            if(budget<owned){rejects([&]{ExpertCache cache(backend.get(),fmt,3,budget);});require(p.calls==0,"one-byte-over budget allocated backing storage");}
            else {ExpertCache cache(backend.get(),fmt,3,budget);require(p.calls==2,"exact budget rejected");}
        }
        for(size_t fail:{size_t(1),size_t(2),size_t(3),size_t(4)}) {
            AllocationProbe p;p.limit=2*size_t(fmt.bytes);p.fail=fail;
            rejects([&]{ExpertCache cache(backend.get(),fmt,5,32u<<20);});
            require(p.live.empty()&&p.bytes==0,"partial allocation failure leaked");
        }
        {
            AllocationProbe p;p.fail_upload=4;
            {ExpertCache cache(backend.get(),fmt,3,owned);cache.upload(0,fmt,blob);
             rejects([&]{cache.upload(1,fmt,blob);});rejects([&]{cache.run(0,std::vector<float>(H));});}
            require(p.live.empty(),"upload failure teardown");
        }
        {
            using namespace strata::worker;
            LocalManifest manifest;manifest.n_layers=1;manifest.n_expert=3;manifest.n_embd=H;manifest.n_ff=F;
            manifest.gu_type=manifest.d_type=8;manifest.identity={1,2};manifest.index_digest={3,4};
            manifest.layers.resize(1);manifest.digests.resize(3,blob_digest(blob));
            GpuPlan plan;require(make_gpu_plan(manifest,{{0,0},{0,1},{0,2}},plan,error),error);
            AllocationProbe p;auto executor=create_vulkan_executor_cpu_test(owned);GpuCounters counters;
            auto reader=[&](const GpuAsset& asset,std::vector<uint8_t>& out,std::string& err){
                if(asset.expert==1){err="controlled reader failure";return false;}out=blob;return true;
            };
            require(!executor->prepare_source(plan,reader,counters,error) && p.live.empty() &&
                    counters.gpu_allocated_bytes==0 && counters.gpu_allocation_count==0,"partial preparation cleanup");
            plan.assets[1]=plan.assets[0];const auto calls=p.calls;
            require(!executor->prepare_source(plan,reader,counters,error)&&p.calls==calls,"duplicate plan allocated storage");
            plan.assets[1].expert=99;
            require(!executor->prepare_source(plan,reader,counters,error)&&p.calls==calls,"invalid ID allocated storage");
        }
        for(int phase:{1,2}) {
            ExpertCache cache(backend.get(),fmt,1,owned);cache.upload(0,fmt,blob);
            auto original=backend->iface.graph_compute;
            static int countdown;countdown=phase;
            static decltype(original) compute;compute=original;
            backend->iface.graph_compute=[](ggml_backend_t b,ggml_cgraph* g){return --countdown ? compute(b,g) : GGML_STATUS_FAILED;};
            rejects([&]{cache.run(0,std::vector<float>(H));});
            backend->iface.graph_compute=original;
            ExpertGraph reference(backend.get(),fmt,blob,true);
            equal(reference.run(std::vector<float>(H,.1f)),cache.run(0,std::vector<float>(H,.1f)));
        }
        {
            Backend other(ggml_backend_cpu_init(),ggml_backend_free);ggml_backend_cpu_set_n_threads(other.get(),1);
            ExpertCache first(backend.get(),fmt,1,owned),second(other.get(),fmt,1,owned),session(backend.get(),fmt,1,owned);
            session.upload(0,fmt,fixture(3));
            first.upload(0,fmt,blob);second.upload(0,fmt,fixture(2));
            auto x=std::vector<float>(H,.1f);auto before=first.run(0,x);second.run(0,x);session.run(0,x);equal(before,first.run(0,x));
        }
        std::puts("portable_expert_cache_test passed: budgets before allocation, backend blocks, upload/compute failures, poisoning and separate ownership");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
