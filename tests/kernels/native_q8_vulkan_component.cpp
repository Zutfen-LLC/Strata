#include "native_q8_fixture.hpp"
#include "strata_vk_native_stages.hpp"
#include "native_q8_observation.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    if(argc==3 && std::string(argv[1])=="--make-fixture") {write_fixture(argv[2],synthetic());return 0;}
    const bool observation=native_q8_observation_requested(argc,argv);
    char** args=argv+(observation?1:0);
    if(observation)std::puts("observation_only=true qualified=false fp32_denorm_mode=unspecified");
    need(argc-(observation?1:0)==9 && std::string(args[1])=="--device" && std::string(args[3])=="--expect-device" &&
         std::string(args[5])=="--fixture" && std::string(args[7])=="--output",
         "use --make-fixture FILE or --device INDEX --expect-device SUBSTRING --fixture FILE --output FILE");
    std::string index=args[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"bad index");
    auto f=read_fixture(args[6]);size_t n=f.x.size(),device=std::stoull(index);
    need(device<size_t(ggml_backend_vk_get_device_count()),"device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend),"backend init failed");
    const char* description=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(args[4]).size() && std::string(description).find(args[4])!=std::string::npos,"unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({16384,nullptr,true}),ggml_free);
    need(bool(context),"metadata failed");
    // Views intentionally start at nonzero, unaligned-to-storage-buffer offsets.
    // Prefix/suffix canaries check both descriptor bias and untouched boundaries.
    auto make=[&](ggml_type type,size_t elements,size_t prefix){
        auto* base=ggml_new_tensor_1d(context.get(),type,elements+prefix+3);
        auto* view=ggml_view_1d(context.get(),base,elements,prefix*4);return std::pair{base,view};};
    auto [xb,x]=make(GGML_TYPE_F32,n,1);auto [gb,g]=make(GGML_TYPE_F32,n,3);
    auto [ub,u]=make(GGML_TYPE_F32,n,5);auto [hb,h]=make(GGML_TYPE_F32,n,7);
    auto [qb,q]=make(GGML_TYPE_I32,n/32*9,9);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(),backend.get()),ggml_backend_buffer_free);
    need(bool(buffer),"tensor allocation failed");
    auto initialize=[](ggml_tensor* t){Bytes b(ggml_nbytes(t),0xa5);ggml_backend_tensor_set(t,b.data(),0,b.size());};
    for(auto* t:{xb,gb,ub,hb,qb})initialize(t);
    ggml_backend_tensor_set(x,f.x.data(),0,n*4);ggml_backend_tensor_set(g,f.gate.data(),0,n*4);
    ggml_backend_tensor_set(u,f.up.data(),0,n*4);
    const std::string effective_path=observation?std::string(args[8])+".effective.spv":std::string();
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> stages(
        observation?strata_vk_native_stages_create_observation(backend.get(),effective_path.c_str()):
                    strata_vk_native_stages_create(backend.get()),strata_vk_native_stages_destroy);
    auto result=blank(n);
    auto run=[&](uint32_t mode,ggml_tensor* input,const std::vector<int32_t>& starts){
        strata_vk_native_stages_run(stages.get(),mode,input,u,h,q,f.n_ff,starts.data(),starts.size());};
    run(0,x,f.starts);ggml_backend_tensor_get(q,result.input.data(),0,result.input.size());
    run(1,g,f.starts);ggml_backend_tensor_get(h,result.hidden.data(),0,result.hidden.size());
    run(0,h,f.starts);ggml_backend_tensor_get(q,result.separate.data(),0,result.separate.size());
    initialize(qb);run(2,g,f.starts);ggml_backend_tensor_get(q,result.fused.data(),0,result.fused.size());
    initialize(qb);run(2,g,{3,3});ggml_backend_tensor_get(q,result.empty.data(),0,result.empty.size());
    bool sentinels_ok=true;
    auto check=[&](bool ok,const char* message){if(!observation)need(ok,message);
        else if(!ok){sentinels_ok=false;std::fprintf(stderr,"observation sentinel: %s\n",message);}};
    for(uint8_t b:result.empty)check(b==0xa5,"empty interval overwritten");
    for(size_t block=0;block<n/32;++block)if(block<size_t(f.starts.front())*f.n_ff/32 || block>=size_t(f.starts.back())*f.n_ff/32)
        for(size_t j=0;j<36;++j)check(result.fused[block*36+j]==0xa5,"inactive fused region overwritten");
    auto canary=[&](ggml_tensor* base,size_t prefix,size_t elems){Bytes b(ggml_nbytes(base));ggml_backend_tensor_get(base,b.data(),0,b.size());
        for(size_t i=0;i<b.size();++i)if(i<prefix*4 || i>=(prefix+elems)*4)check(b[i]==0xa5,"view sentinel overwritten");};
    canary(xb,1,n);canary(gb,3,n);canary(ub,5,n);canary(hb,7,n);canary(qb,9,n/32*9);
    if(observation){
        write_observation_result(args[8],result,sentinels_ok);
        std::ofstream views(std::string(args[8])+".views.bin",std::ios::binary);
        need(bool(views),"cannot create observation view receipt");word(views,0x3156384e);
        for(auto* t:{xb,gb,ub,hb,qb}){Bytes b(ggml_nbytes(t));ggml_backend_tensor_get(t,b.data(),0,b.size());
            word(views,uint32_t(b.size()));views.write(reinterpret_cast<const char*>(b.data()),b.size());}
        need(bool(views),"observation view receipt write failed");
        need(sentinels_ok,"observation: sentinel failure; raw capture and views preserved");
    }else write_result(args[8],result);
    std::printf("device=%s elements=%zu n_ff=%u groups=%zu backing_bytes=%zu; captures written; parity requires native comparison\n",
        description,n,f.n_ff,f.starts.size()-1,ggml_backend_buffer_get_size(buffer.get()));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"native stage component: %s\n",e.what());return 1;}}
