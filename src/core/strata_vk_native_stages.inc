// Included in the existing pin-checked build-tree Vulkan translation unit.
#include "strata_vk_native_stages.hpp"
#include "strata_native_stages_spv.hpp"
#include "strata_native_quant_trace_spv.hpp"
#include "strata_native_quant_rn_spv.hpp"
#include "native_q8_quant_trace.hpp"
#include "native_q8_gu_layout.hpp"
#include "native_q8_spirv.hpp"
#include "native_q8_float_controls.hpp"
#include "native_q8_observation.hpp"
namespace strata::core::portable_detail {
#include "strata_native_observe_pipeline.inc"
struct StrataVkNativeStages {
    // Reuse the existing pipeline/owned metadata cleanup. Its status buffer
    // holds bounded group metadata; no extra allocator or device is introduced.
    std::unique_ptr<StrataVkLogitsGuard> owner;
    bool quant_trace=false;
};
StrataVkNativeStages* strata_vk_native_stages_create(ggml_backend_t backend) {
    logits_guard_need(backend && ggml_backend_is_vk(backend),"native stages: Vulkan required");
    auto result=std::make_unique<StrataVkNativeStages>();
    result->owner=std::make_unique<StrataVkLogitsGuard>(backend);
    auto& g=*result->owner;
    auto* ctx=static_cast<ggml_backend_vk_context*>(backend->context);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(limits.maxComputeWorkGroupInvocations>=32 && limits.maxComputeWorkGroupSize[0]>=32 &&
                      limits.maxComputeSharedMemorySize>=3*32*4,"native stages: device geometry unsupported");
    const auto properties=ctx->device->physical_device.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceVulkan12Properties>();
    const auto& fp=properties.get<vk::PhysicalDeviceVulkan12Properties>();
    native_q8_require_float_controls(fp,properties.get<vk::PhysicalDeviceProperties2>().properties);
    g.metadata.reset(ggml_init({4096,nullptr,true}));
    logits_guard_need(bool(g.metadata),"native stages: metadata allocation failed");
    g.status=ggml_new_tensor_1d(g.metadata.get(),GGML_TYPE_I32,32);
    g.buffer.reset(ggml_backend_alloc_ctx_tensors(g.metadata.get(),backend));
    logits_guard_need(bool(g.buffer),"native stages: metadata buffer failed");
    g.bytes=ggml_backend_buffer_get_size(g.buffer.get());
    auto& p=g.pipeline;
    p->name="strata_native_q8_stages_standalone";p->parameter_count=6;p->push_constant_size=9*4;
    p->wg_denoms={1,1,1};p->align=1;p->initialized=true;
    const auto code=native_stage_float_controls(strata_native_stages_spv,sizeof(strata_native_stages_spv)/4);
    ggml_vk_create_pipeline_func(ctx->device,p,code.size()*sizeof(uint32_t),code.data(),
                                "main",6,{1,1,1},{},false,false,0);
    return result.release();
}
StrataVkNativeStages* strata_vk_native_stages_create_observation(ggml_backend_t backend,const char* effective_spv_path) {
    logits_guard_need(backend && ggml_backend_is_vk(backend),"native stages: Vulkan required");
    auto result=std::make_unique<StrataVkNativeStages>();
    result->owner=std::make_unique<StrataVkLogitsGuard>(backend);
    auto& g=*result->owner;
    auto* ctx=static_cast<ggml_backend_vk_context*>(backend->context);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(limits.maxComputeWorkGroupInvocations>=32 && limits.maxComputeWorkGroupSize[0]>=32 &&
                      limits.maxComputeSharedMemorySize>=3*32*4,"native stages: device geometry unsupported");
    const auto properties=ctx->device->physical_device.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceVulkan12Properties>();
    const auto& fp=properties.get<vk::PhysicalDeviceVulkan12Properties>();
    native_q8_require_observation_controls(fp);
    logits_guard_need(effective_spv_path && *effective_spv_path,"observation: final shader receipt path required");
    logits_guard_need(bool(fp.shaderDenormPreserveFloat16)==ctx->device->float_controls_denorm_preserve_fp16 &&
        bool(fp.shaderRoundingModeRTEFloat16)==ctx->device->float_controls_rte_fp16,"observation: helper cached controls disagree with query");
    g.metadata.reset(ggml_init({4096,nullptr,true}));
    logits_guard_need(bool(g.metadata),"native stages: metadata allocation failed");
    g.status=ggml_new_tensor_1d(g.metadata.get(),GGML_TYPE_I32,32);
    g.buffer.reset(ggml_backend_alloc_ctx_tensors(g.metadata.get(),backend));
    logits_guard_need(bool(g.buffer),"native stages: metadata buffer failed");
    g.bytes=ggml_backend_buffer_get_size(g.buffer.get());
    auto& p=g.pipeline;
    p->name="strata_native_q8_observation_unspecified_fp32_denorm";p->parameter_count=6;p->push_constant_size=9*4;
    p->wg_denoms={1,1,1};p->align=1;p->initialized=true;
    const auto strict=native_stage_float_controls(strata_native_stages_spv,sizeof(strata_native_stages_spv)/4);
    const auto code=native_q8_observation_code(strict);
    strata_vk_observation_create_pipeline_func(ctx->device,p,code.size()*sizeof(uint32_t),code.data(),
                                "main",6,{1,1,1},{},false,false,0,effective_spv_path);
    return result.release();
}
static StrataVkNativeStages* strata_vk_native_stages_create_variant_observation(ggml_backend_t backend,const char* effective_spv_path,
    const uint32_t* spv,size_t words,const char* name,bool trace,size_t budget=SIZE_MAX) {
    logits_guard_need(backend && ggml_backend_is_vk(backend),"native stages: Vulkan required");
    auto result=std::make_unique<StrataVkNativeStages>();
    result->quant_trace=trace;
    result->owner=std::make_unique<StrataVkLogitsGuard>(backend);
    auto& g=*result->owner;
    auto* ctx=static_cast<ggml_backend_vk_context*>(backend->context);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(limits.maxComputeWorkGroupInvocations>=32 && limits.maxComputeWorkGroupSize[0]>=32 &&
                      limits.maxComputeSharedMemorySize>=3*32*4,"native stages: device geometry unsupported");
    const auto properties=ctx->device->physical_device.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceVulkan12Properties>();
    const auto& fp=properties.get<vk::PhysicalDeviceVulkan12Properties>();
    native_q8_require_observation_controls(fp);
    logits_guard_need(effective_spv_path && *effective_spv_path,"observation: final shader receipt path required");
    logits_guard_need(bool(fp.shaderDenormPreserveFloat16)==ctx->device->float_controls_denorm_preserve_fp16 &&
        bool(fp.shaderRoundingModeRTEFloat16)==ctx->device->float_controls_rte_fp16,"observation: helper cached controls disagree with query");
    g.metadata.reset(ggml_init({4096,nullptr,true}));
    logits_guard_need(bool(g.metadata),"native stages: metadata allocation failed");
    g.status=ggml_new_tensor_1d(g.metadata.get(),GGML_TYPE_I32,32);
    const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(g.metadata.get(),ggml_backend_get_default_buffer_type(backend));
    logits_guard_need(bound<=budget,"native stages: aligned metadata budget exceeded");
    g.buffer.reset(ggml_backend_alloc_ctx_tensors(g.metadata.get(),backend));
    logits_guard_need(bool(g.buffer),"native stages: metadata buffer failed");
    g.bytes=ggml_backend_buffer_get_size(g.buffer.get());
    logits_guard_need(g.bytes<=budget,"native stages: actual metadata budget exceeded");
    auto& p=g.pipeline;
    p->name=name;p->parameter_count=6;p->push_constant_size=9*4;
    p->wg_denoms={1,1,1};p->align=1;p->initialized=true;
    const auto strict=native_stage_float_controls(spv,words);
    const auto code=native_q8_observation_code(strict);
    strata_vk_observation_create_pipeline_func(ctx->device,p,code.size()*sizeof(uint32_t),code.data(),
                                "main",6,{1,1,1},{},false,false,0,effective_spv_path);
    return result.release();
}
StrataVkNativeStages* strata_vk_native_stages_create_quant_trace_observation(ggml_backend_t backend,const char* path) {
    return strata_vk_native_stages_create_variant_observation(backend,path,strata_native_quant_trace_spv,sizeof(strata_native_quant_trace_spv)/4,
        "strata_native_q8_quant_trace_observation",true);
}
StrataVkNativeStages* strata_vk_native_stages_create_quant_rn_observation(ggml_backend_t backend,const char* path,size_t budget) {
    return strata_vk_native_stages_create_variant_observation(backend,path,strata_native_quant_rn_spv,sizeof(strata_native_quant_rn_spv)/4,
        "strata_native_q8_quant_rn_observation",false,budget);
}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages* p) {return p?ggml_backend_buffer_get_size(p->owner->buffer.get()):0;}
void strata_vk_native_stages_destroy(StrataVkNativeStages* s) noexcept {delete s;}
void strata_vk_native_stages_run(StrataVkNativeStages* s,uint32_t mode,const ggml_tensor* gate,
    const ggml_tensor* up,ggml_tensor* hidden,ggml_tensor* packed,uint32_t n_ff,
    const int32_t* starts,size_t n_starts) {
    logits_guard_need(s && mode<=2 && gate && up && hidden && packed,"native stages: invalid call");
    const int64_t n=ggml_nelements(gate);
    if(s->quant_trace)native_q8_quant_trace_shape(mode,n,n_ff,starts,n_starts);
    logits_guard_need(n>0 && n<=1048576 && n_ff>0 && n_ff%32==0 && n%n_ff==0 &&
        gate->type==GGML_TYPE_F32 && up->type==GGML_TYPE_F32 && hidden->type==GGML_TYPE_F32 &&
        packed->type==GGML_TYPE_I32 && ggml_nelements(up)==n && ggml_nelements(hidden)==n &&
        ggml_nelements(packed)==n/32*9,"native stages: shape/type mismatch");
    logits_guard_need(starts && n_starts>=1 && n_starts<=25,"native stages: bounded group starts required");
    for(size_t i=0;i<n_starts;++i) logits_guard_need(starts[i]>=0 && starts[i]<=n/n_ff &&
        (!i || starts[i]>=starts[i-1]),"native stages: invalid active interval");
    auto& g=*s->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    logits_guard_need(uint64_t(n/32)<=ctx->device->properties.limits.maxComputeWorkGroupCount[0],
                      "native stages: dispatch exceeds device limit");
    std::array<uint32_t,9> params{mode,uint32_t(n),n_ff};
    size_t at=3;
    for(const auto* t:{gate,up,static_cast<const ggml_tensor*>(g.status),static_cast<const ggml_tensor*>(g.status),
                       static_cast<const ggml_tensor*>(hidden),static_cast<const ggml_tensor*>(packed)}) {
        logits_guard_need(t->buffer && ggml_is_contiguous(t) && ggml_backend_buffer_is_vk(t->buffer),
                          "native stages: contiguous Vulkan tensors required");
        auto* buffer=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        logits_guard_need(buffer->device.lock()==ctx->device,"native stages: tensor on different device");
        const uint32_t bias=get_misalign_bytes(ctx,t);
        logits_guard_need(bias%4==0,"native stages: word alignment required");params[at++]=bias/4;
    }
    const auto overlaps=[](const ggml_tensor* a,const ggml_tensor* b){
        if(a->buffer!=b->buffer)return false;
        const auto x=reinterpret_cast<uintptr_t>(a->data),y=reinterpret_cast<uintptr_t>(b->data);
        return x<=y ? y-x<ggml_nbytes(a) : x-y<ggml_nbytes(b);
    };
    logits_guard_need(!overlaps(packed,gate) && !overlaps(packed,up) && !overlaps(packed,hidden) &&
        (mode!=1 || (!overlaps(hidden,gate) && !overlaps(hidden,up))),
        "native stages: overlapping input/output regions");
    // 25 start positions cover 24 groups; count stays outside their interval.
    params[6]=params[5]+31;
    const auto metadata=q8_stage_metadata(starts,n_starts);
    ggml_backend_vk_synchronize(g.backend);
    ggml_backend_tensor_set(g.status,metadata.data(),0,sizeof(metadata));
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);
    auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,
        {ggml_vk_tensor_subbuffer(ctx,gate,true),ggml_vk_tensor_subbuffer(ctx,up,true),
         ggml_vk_tensor_subbuffer(ctx,g.status,true),ggml_vk_tensor_subbuffer(ctx,g.status,true),
         ggml_vk_tensor_subbuffer(ctx,hidden,true),ggml_vk_tensor_subbuffer(ctx,packed,true)},params,{uint32_t(n/32),1,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
}
}

#include "strata_vk_q8_gu.inc"

#include "strata_vk_q8_down.inc"

#ifdef STRATA_NATIVE_Q8_STRICT_FMA
#include "strata_vk_native_moe_combine.inc"
#include "strata_vk_native_shared.inc"
#endif

#include "strata_vk_native_router.inc"

#ifdef STRATA_ROUTER_DIAGNOSTIC
#include "strata_vk_native_router_trace.inc"
#endif
