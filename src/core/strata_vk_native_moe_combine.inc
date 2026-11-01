#include "strata_vk_native_moe_combine.hpp"
#include "native_moe_combine_layout.hpp"
#include "strata_native_moe_combine_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeMoeCombine {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkNativeMoeCombine* strata_vk_native_moe_combine_create(ggml_backend_t backend,const char* path) {
    auto result=std::make_unique<StrataVkNativeMoeCombine>();
    // Reuse Q8 device/feature receipt, controls, final-module validation and
    // pipeline/resource ownership. Only the explicit strict build includes us.
    result->owner=strata_vk_q8_component_owner(backend,strata_native_moe_combine_spv,sizeof(strata_native_moe_combine_spv)/4,
        NativeComponentAddressing::descriptor,"strata_native_moe_combine_k10_observation",4,6,path);
    return result.release();
}
size_t strata_vk_native_moe_combine_bytes(const StrataVkNativeMoeCombine* p) {return p?ggml_backend_buffer_get_size(p->owner->buffer.get()):0;}
void strata_vk_native_moe_combine_destroy(StrataVkNativeMoeCombine* p) noexcept {delete p;}
void strata_vk_native_moe_combine_run(StrataVkNativeMoeCombine* s,const ggml_tensor* parts,
    const ggml_tensor* weights,const ggml_tensor* shared,ggml_tensor* output,uint32_t tokens) {
    logits_guard_need(s && parts && weights && shared && output,"native MoE combine: invalid call");
    native_moe_combine_shape(2560,tokens);
    const int64_t rows=int64_t(tokens)*2560;
    logits_guard_need(ggml_nelements(parts)==rows*10 && ggml_nelements(weights)==int64_t(tokens)*10 &&
        ggml_nelements(shared)==rows && ggml_nelements(output)==rows,"native MoE combine: tensor shape mismatch");
    auto& g=*s->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const ggml_tensor* tensors[]={parts,weights,shared,output};std::array<uint32_t,4> biases{};
    for(size_t i=0;i<4;++i) {
        const auto* t=tensors[i];
        logits_guard_need(t->type==GGML_TYPE_F32 && t->buffer && ggml_is_contiguous(t) &&
            ggml_backend_buffer_is_vk(t->buffer),"native MoE combine: contiguous Vulkan F32 required");
        auto* b=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        logits_guard_need(b->device.lock()==ctx->device,"native MoE combine: different device");
        biases[i]=get_misalign_bytes(ctx,t);
    }
    const auto overlap=[](const ggml_tensor* a,const ggml_tensor* b) {
        if(a->buffer!=b->buffer)return false;
        const auto x=reinterpret_cast<uintptr_t>(a->data),y=reinterpret_cast<uintptr_t>(b->data);
        return x<=y?y-x<ggml_nbytes(a):x-y<ggml_nbytes(b);
    };
    for(size_t i=0;i<3;++i)logits_guard_need(!overlap(output,tensors[i]),"native MoE combine: overlapping output");
    const auto params=native_moe_combine_parameters(2560,tokens,biases);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(limits.maxComputeWorkGroupInvocations>=128 && limits.maxComputeWorkGroupSize[0]>=128 &&
        limits.maxComputeWorkGroupCount[0]>=5 && limits.maxComputeWorkGroupCount[1]>=tokens,"native MoE combine: grid unsupported");
    ggml_backend_vk_synchronize(g.backend);
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,parts,true),ggml_vk_tensor_subbuffer(ctx,weights,true),
        ggml_vk_tensor_subbuffer(ctx,shared,true),ggml_vk_tensor_subbuffer(ctx,output,true)},params,{5,tokens,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
}
}
