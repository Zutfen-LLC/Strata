#include "strata_vk_native_ple_grouped_norm.hpp"
#include "strata_native_ple_grouped_norm_t1_spv.hpp"
#include "strata_native_ple_grouped_norm_rcp_fma_t1_spv.hpp"
#include <limits>
namespace strata::core::portable_detail {
struct StrataVkNativePleGroupedNorm {std::unique_ptr<StrataVkLogitsGuard> owner;size_t budget=0;uint64_t calls=0;bool rcp_fma=false;};
static StrataVkNativePleGroupedNorm* ple_grouped_norm_create(ggml_backend_t b,const char* path,size_t budget,bool rcp_fma) {
    native_hc_need(b&&ggml_backend_is_vk(b),"native PLE grouped norm: Vulkan backend required");
    auto* ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto& l=ctx->device->properties.limits;
    native_hc_need(l.maxComputeWorkGroupInvocations>=1024&&l.maxComputeWorkGroupSize[0]>=1024&&
                   l.maxComputeSharedMemorySize>=1056*4&&l.maxComputeWorkGroupCount[0]>=4&&
                   l.maxPerStageDescriptorStorageBuffers>=5&&l.maxDescriptorSetStorageBuffers>=5,
                   "native PLE grouped norm: workgroup/shared/descriptors unsupported");
    auto p=std::make_unique<StrataVkNativePleGroupedNorm>();p->budget=budget;p->rcp_fma=rcp_fma;
    const auto* code=rcp_fma?strata_native_ple_grouped_norm_rcp_fma_t1_spv:strata_native_ple_grouped_norm_t1_spv;
    const auto words=rcp_fma?sizeof(strata_native_ple_grouped_norm_rcp_fma_t1_spv)/4:sizeof(strata_native_ple_grouped_norm_t1_spv)/4;
    p->owner=strata_vk_q8_component_owner(b,code,words,
        NativeComponentAddressing::descriptor,rcp_fma?"strata_native_ple_grouped_norm_rcp_fma_observation":"strata_native_ple_grouped_norm_t1_observation",5,6,path,budget,1);
    return p.release();
}
StrataVkNativePleGroupedNorm* strata_vk_native_ple_grouped_norm_create(ggml_backend_t b,const char* path,size_t budget) {return ple_grouped_norm_create(b,path,budget,false);}
StrataVkNativePleGroupedNorm* strata_vk_native_ple_grouped_norm_rcp_fma_observation_create(ggml_backend_t b,const char* path,size_t budget) {return ple_grouped_norm_create(b,path,budget,true);}
void strata_vk_native_ple_grouped_norm_destroy(StrataVkNativePleGroupedNorm* p) noexcept {delete p;}
size_t strata_vk_native_ple_grouped_norm_bytes(const StrataVkNativePleGroupedNorm* p){return p?p->owner->bytes:0;}
uint64_t strata_vk_native_ple_grouped_norm_calls(const StrataVkNativePleGroupedNorm* p) noexcept{return p?p->calls:0;}
uint32_t strata_vk_native_ple_grouped_norm_run(StrataVkNativePleGroupedNorm* p,const ggml_tensor* input,const ggml_tensor* gamma,
                                      ggml_tensor* output,ggml_tensor* diagnostic) {
    native_hc_need(p,"native PLE grouped norm: owner required");auto& g=*p->owner;
    auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const ggml_tensor* tensors[]={input,gamma,output,diagnostic};const size_t bytes[]={40960,40960,40960,p->rcp_fma?size_t(80):size_t(48)};
    std::array<uint32_t,6> off{};size_t used=g.bytes;ggml_backend_buffer_t seen[4]{};size_t nseen=0;
    for(size_t i=0;i<4;++i){const auto* t=tensors[i];
        native_hc_need(t&&t->type==GGML_TYPE_F32&&ggml_nbytes(t)==bytes[i]&&ggml_is_contiguous(t)&&t->buffer&&ggml_backend_buffer_is_vk(t->buffer),"native PLE grouped norm: exact contiguous Vulkan F32 span required");
        auto* b=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native PLE grouped norm: foreign device");
        auto bias=get_misalign_bytes(ctx,t);native_hc_need(bias%4==0&&uint64_t(bias)+ggml_nbytes(t)<=ctx->device->properties.limits.maxStorageBufferRange,"native PLE grouped norm: alignment or descriptor range");off[i]=bias/4;
        bool duplicate=false;for(size_t j=0;j<nseen;++j)duplicate|=seen[j]==t->buffer;
        if(!duplicate){seen[nseen++]=t->buffer;const size_t n=ggml_backend_buffer_get_size(t->buffer);native_hc_need(n<=std::numeric_limits<size_t>::max()-used,"native PLE grouped norm: budget overflow");used+=n;}
        for(size_t j=0;j<i;++j)if((i>=2||j>=2)&&t->buffer==tensors[j]->buffer){auto x=reinterpret_cast<uintptr_t>(t->data),y=reinterpret_cast<uintptr_t>(tensors[j]->data);native_hc_need(x<=y?y-x>=ggml_nbytes(t):x-y>=ggml_nbytes(tensors[j]),"native PLE grouped norm: output overlap");}
    }
    native_hc_need(used<=p->budget,"native PLE grouped norm: allocation budget exceeded");
    ggml_backend_vk_synchronize(g.backend);std::vector<float> values(10240);
    for(const auto* t:{input,gamma}){ggml_backend_tensor_get(t,values.data(),0,40960);for(float x:values)native_hc_normal(x);}
    const auto status_bias=get_misalign_bytes(ctx,g.status);native_hc_need(status_bias%4==0,"native PLE grouped norm: status alignment");off[4]=status_bias/4;off[5]=2560;const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,input,true),ggml_vk_tensor_subbuffer(ctx,gamma,true),
        ggml_vk_tensor_subbuffer(ctx,output,true),ggml_vk_tensor_subbuffer(ctx,diagnostic,true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{4,1,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
    uint32_t status=0;ggml_backend_tensor_get(g.status,&status,0,4);if(!status)++p->calls;return status;
}
}
