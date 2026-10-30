#include "native_moe_combine_fixture.hpp"
#include "strata_vk_native_moe_combine.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <cfenv>
#include <memory>
using namespace native_moe_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    need(argc==9 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--fixture" && std::string(argv[7])=="--output",
        "usage: native_moe_combine_component --device INDEX --expect-device DESCRIPTION --fixture FILE --output NEW_PREFIX");
    need(std::fesetround(FE_TONEAREST)==0,"combine requires host RNE reference");
    const auto f=read_fixture(argv[6]);const std::string prefix=argv[8];fresh_outputs(prefix);
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"combine invalid device index");
    const size_t device=std::stoull(index);need(device<size_t(ggml_backend_vk_get_device_count()),"combine device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend),"combine Vulkan init failed");
    const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(argv[4]).size() && name && std::string(name).find(argv[4])!=std::string::npos,"combine unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"combine tensor metadata allocation failed");
    const auto before=padded(f);auto after=before;
    std::array<ggml_tensor*,4> base{},views{};
    for(size_t i=0;i<4;++i) {
        base[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,int64_t(before[i].size()));
        const size_t count=before[i].size()-prefixes[i]-7;
        views[i]=ggml_view_1d(ctx.get(),base[i],int64_t(count),prefixes[i]*4);
    }
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer),"combine tensor allocation failed");
    for(size_t i=0;i<4;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);
    std::unique_ptr<StrataVkNativeMoeCombine,decltype(&strata_vk_native_moe_combine_destroy)> component(
        strata_vk_native_moe_combine_create(backend.get(),(prefix+".effective.spv").c_str()),strata_vk_native_moe_combine_destroy);
    strata_vk_native_moe_combine_run(component.get(),views[0],views[1],views[2],views[3],f.tokens);
    for(size_t i=0;i<4;++i)ggml_backend_tensor_get(base[i],after[i].data(),0,after[i].size()*4);
    const auto different=check_and_write(prefix,f,before,after);
    printf("native_moe_combine_k10_observation=true qualified=false quantizer_qualified=false device=%s tokens=%u words=%zu different_bits=%zu tensor_bytes=%zu component_bytes=%zu\n",
        name,f.tokens,size_t(f.tokens)*H,different,ggml_backend_buffer_get_size(buffer.get()),strata_vk_native_moe_combine_bytes(component.get()));
    return different?1:0;
}catch(const std::exception& e){fprintf(stderr,"native MoE combine component: %s\n",e.what());return 1;}}
