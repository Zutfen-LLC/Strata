#include "strata_vk_native_router.hpp"
#include "native_router_layout.hpp"
#include "strata_native_router_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeRouter {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkNativeRouter* strata_vk_native_router_create(ggml_backend_t backend,const char* path) {
    auto result=std::make_unique<StrataVkNativeRouter>();
    // Reuse Q8 device/feature receipt, controls, final-module validation and
    // pipeline/resource ownership. No FMA is introduced by this router.
    result->owner=strata_vk_q8_component_owner(backend,strata_native_router_spv,sizeof(strata_native_router_spv)/4,
        "strata_native_router_k10_observation",4,4,path);
    return result.release();
}
void strata_vk_native_router_destroy(StrataVkNativeRouter* p) noexcept {delete p;}
void strata_vk_native_router_run(StrataVkNativeRouter* s,const ggml_tensor* logits,
    ggml_tensor* ids,ggml_tensor* weights,uint32_t tokens) {
    logits_guard_need(s && logits && ids && weights,"native router: invalid call");
    native_router_shape(512,10,tokens);
    logits_guard_need(ggml_nelements(logits)==int64_t(tokens)*512 && ggml_nelements(ids)==int64_t(tokens)*10 &&
        ggml_nelements(weights)==int64_t(tokens)*10,"native router: tensor shape mismatch");
    auto& g=*s->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const ggml_tensor* tensors[]={logits,ids,weights};std::array<uint32_t,3> biases{};
    for(size_t i=0;i<3;++i) {
        const auto* t=tensors[i];
        logits_guard_need(t->type==(i==1?GGML_TYPE_I32:GGML_TYPE_F32) && t->buffer && ggml_is_contiguous(t) &&
            ggml_backend_buffer_is_vk(t->buffer),"native router: contiguous Vulkan F32/I32/F32 required");
        auto* b=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        logits_guard_need(b->device.lock()==ctx->device,"native router: different device");
        biases[i]=get_misalign_bytes(ctx,t);
    }
    for(size_t i=0;i<3;++i)for(size_t j=0;j<i;++j) {
        const auto* a=tensors[i];const auto* b=tensors[j];
        if(a->buffer==b->buffer) {
            const auto x=reinterpret_cast<uintptr_t>(a->data),y=reinterpret_cast<uintptr_t>(b->data);
            logits_guard_need(!(x<=y?y-x<ggml_nbytes(a):x-y<ggml_nbytes(b)),"native router: overlapping spans");
        }
    }
    const auto params=native_router_parameters(tokens,biases);
    logits_guard_need(ctx->device->properties.limits.maxComputeWorkGroupCount[0]>=tokens,"native router: grid unsupported");
    ggml_backend_vk_synchronize(g.backend);
    uint32_t status=0;ggml_backend_tensor_set(g.status,&status,0,4);
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,logits,true),ggml_vk_tensor_subbuffer(ctx,ids,true),
        ggml_vk_tensor_subbuffer(ctx,weights,true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},params,{tokens,1,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
    ggml_backend_tensor_get(g.status,&status,0,4);
    logits_guard_need(status==0,"native router: nonfinite input refused; outputs invalid");
}
}
