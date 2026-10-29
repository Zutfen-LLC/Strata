#include "native_q8_fixture.hpp"
#include "strata_vk_native_stages.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    if(argc==3 && std::string(argv[1])=="--make-fixture") {write_fixture(argv[2],synthetic());return 0;}
    need(argc==9 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
         std::string(argv[5])=="--fixture" && std::string(argv[7])=="--output",
         "use --make-fixture FILE or --device INDEX --expect-device SUBSTRING --fixture FILE --output FILE");
    std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"bad index");
    auto f=read_fixture(argv[6]);size_t n=f.x.size(),device=std::stoull(index);
    need(device<size_t(ggml_backend_vk_get_device_count()),"device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend),"backend init failed");
    const char* description=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(argv[4]).size() && std::string(description).find(argv[4])!=std::string::npos,"unexpected device");
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
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> stages(
        strata_vk_native_stages_create(backend.get()),strata_vk_native_stages_destroy);
    auto result=blank(n);
    auto run=[&](uint32_t mode,ggml_tensor* input,const std::vector<int32_t>& starts){
        strata_vk_native_stages_run(stages.get(),mode,input,u,h,q,f.n_ff,starts.data(),starts.size());};
    run(0,x,f.starts);ggml_backend_tensor_get(q,result.input.data(),0,result.input.size());
    run(1,g,f.starts);ggml_backend_tensor_get(h,result.hidden.data(),0,result.hidden.size());
    run(0,h,f.starts);ggml_backend_tensor_get(q,result.separate.data(),0,result.separate.size());
    initialize(qb);run(2,g,f.starts);ggml_backend_tensor_get(q,result.fused.data(),0,result.fused.size());
    initialize(qb);run(2,g,{3,3});ggml_backend_tensor_get(q,result.empty.data(),0,result.empty.size());
    for(uint8_t b:result.empty)need(b==0xa5,"empty interval overwritten");
    for(size_t block=0;block<n/32;++block)if(block<size_t(f.starts.front())*f.n_ff/32 || block>=size_t(f.starts.back())*f.n_ff/32)
        for(size_t j=0;j<36;++j)need(result.fused[block*36+j]==0xa5,"inactive fused region overwritten");
    auto canary=[](ggml_tensor* base,size_t prefix,size_t elems){Bytes b(ggml_nbytes(base));ggml_backend_tensor_get(base,b.data(),0,b.size());
        for(size_t i=0;i<b.size();++i)if(i<prefix*4 || i>=(prefix+elems)*4)need(b[i]==0xa5,"view sentinel overwritten");};
    canary(xb,1,n);canary(gb,3,n);canary(ub,5,n);canary(hb,7,n);canary(qb,9,n/32*9);
    write_result(argv[8],result);
    std::printf("device=%s elements=%zu n_ff=%u groups=%zu backing_bytes=%zu; captures written; parity requires native comparison\n",
        description,n,f.n_ff,f.starts.size()-1,ggml_backend_buffer_get_size(buffer.get()));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"native stage component: %s\n",e.what());return 1;}}
