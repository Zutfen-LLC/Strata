#include "native_router_trace_fixture.hpp"
#include "strata_vk_native_router_trace.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <cfenv>
#include <memory>
using namespace native_router_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    need(argc==9 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--fixture" && std::string(argv[7])=="--output",
        "usage: native_router_trace_component --device INDEX --expect-device DESCRIPTION --fixture FILE --output NEW_PREFIX");
    need(std::fesetround(FE_TONEAREST)==0,"router requires host RNE reference");
    const auto f=read_fixture(argv[6]);const std::string prefix=argv[8];validate_trace_fixture(f);reserve(prefix);
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"router invalid device index");
    const size_t device=std::stoull(index);need(device<size_t(ggml_backend_vk_get_device_count()),"router device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend),"router Vulkan init failed");
    const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(argv[4]).size() && name && std::string(name).find(argv[4])!=std::string::npos,"router unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"router tensor metadata allocation failed");
    const auto before=padded(f);auto after=before;
    std::array<ggml_tensor*,3> base{},views{};
    for(size_t i=0;i<3;++i) {
        base[i]=ggml_new_tensor_1d(ctx.get(),i==1?GGML_TYPE_I32:GGML_TYPE_F32,int64_t(before[i].size()));
        const size_t count=before[i].size()-prefixes[i]-9;
        views[i]=ggml_view_1d(ctx.get(),base[i],int64_t(count),prefixes[i]*4);
    }
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer),"router tensor allocation failed");
    for(size_t i=0;i<3;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);
    std::unique_ptr<StrataVkNativeRouter,decltype(&strata_vk_native_router_destroy)> component(
        strata_vk_native_router_trace_create(backend.get(),(prefix+"/effective.spv").c_str()),strata_vk_native_router_destroy);
    std::array<uint32_t,26> trace{};
    strata_vk_native_router_trace_run(component.get(),views[0],views[1],views[2],trace.data());
    for(size_t i=0;i<3;++i)ggml_backend_tensor_get(base[i],after[i].data(),0,after[i].size()*4);
    receipt(prefix,f,before,after);
    write_trace(prefix,trace);
    complete_trace(prefix);
    printf("native_router_row2_diagnostic=true qualified=false device=%s tokens=%u native_comparison=required\n",name,f.tokens);
    return 0;
}catch(const std::exception& e){fprintf(stderr,"native router component: %s\n",e.what());return 1;}}
