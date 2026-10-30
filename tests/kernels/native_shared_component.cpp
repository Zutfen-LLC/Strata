#include "native_shared_fixture.hpp"
#include "strata_vk_native_shared.hpp"
#include "strata_vk_native_stages.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_shared_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv) {try {
    arguments(argc,argv);const auto f=read_fixture(argv[6]);const std::string dir=argv[8];reserve(dir);
    const size_t device=std::stoull(argv[2]);need(device<size_t(ggml_backend_vk_get_device_count()),"shared Vulkan device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);need(bool(backend),"shared Vulkan init");
    const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(name && std::string(name).find(argv[4])!=std::string::npos,"shared unexpected Vulkan device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({65536,nullptr,true}),ggml_free);need(bool(ctx),"shared metadata allocation");
    const auto before=padded(f);auto after=before;auto n=counts(f.h,f.f,f.t);
    std::array<ggml_tensor*,N> base{},v{};
    for(size_t i=0;i<N;++i) {
        auto type=(i>=WG && i<=QX) || i==QH?GGML_TYPE_I32:GGML_TYPE_F32;
        base[i]=ggml_new_tensor_1d(ctx.get(),type,int64_t(before[i].size()));
        v[i]=ggml_view_1d(ctx.get(),base[i],int64_t(n[i]),prefix(i)*4);
    }
    const size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));
    need(bound<=256ull*1024*1024,"shared aligned tensor budget exceeded");
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
    need(bool(buffer),"shared tensor allocation");
    need(ggml_backend_buffer_get_size(buffer.get())<=256ull*1024*1024,"shared actual tensor budget exceeded");
    for(size_t i=0;i<N;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);
    std::unique_ptr<StrataVkNativeShared,decltype(&strata_vk_native_shared_destroy)> component(
        strata_vk_native_shared_create(backend.get(),(dir+"/dense-scalar-effective.spv").c_str()),strata_vk_native_shared_destroy);
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> stages(
        strata_vk_native_stages_create_quant_rn_observation(backend.get(),(dir+"/quant-effective.spv").c_str()),strata_vk_native_stages_destroy);
    const int32_t starts[]={0,int32_t(f.t)};
    auto quant=[&](Slot input,Slot dummy,Slot packed,uint32_t width){strata_vk_native_stages_run(stages.get(),0,v[input],v[input],v[dummy],v[packed],width,starts,2);};
    auto dense=[&](Slot weights,Slot input,Slot output,uint32_t ni,uint32_t no){strata_vk_native_shared_run(component.get(),0,v[weights],v[input],v[output],v[DUMMY],ni,no,f.t);};
    quant(X,DOWN,QX,f.h);dense(WG,QX,GATE,f.h,f.f);dense(WU,QX,UP,f.h,f.f);
    strata_vk_native_stages_run(stages.get(),1,v[GATE],v[UP],v[H],v[QH],f.f,starts,2);
    quant(H,H,QH,f.f);dense(WD,QH,DOWN,f.f,f.h);
    strata_vk_native_shared_run(component.get(),1,v[WS],v[X],v[RAW],v[DUMMY],f.h,1,f.t);
    // Scale consumes the raw F32 gate, which needs an F32 descriptor.
    strata_vk_native_shared_run(component.get(),2,v[RAW],v[DOWN],v[OUT],v[SIGMOID],f.h,1,f.t);
    for(size_t i=0;i<N;++i)ggml_backend_tensor_get(base[i],after[i].data(),0,after[i].size()*4);
    receipt(dir,f,before,after);printf("native_shared_observation=true qualified=false device=%s\n",name);return 0;
}catch(const std::exception& e){fprintf(stderr,"native shared component: %s\n",e.what());return 1;}}
