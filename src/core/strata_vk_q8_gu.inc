// Standalone consumer of already packed inputs; included in the pin-checked TU.
#include "strata_vk_q8_gu.hpp"
#include "strata_q8_gu_spv.hpp"
#include "native_q8_gu_spirv.hpp"
namespace strata::core::portable_detail {
struct StrataVkQ8Gu {std::unique_ptr<StrataVkLogitsGuard> owner;};
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t backend,
    const uint32_t* shader,size_t shader_words,const char* name,uint32_t parameters,uint32_t push_words,const char* effective_spv_path) {
    logits_guard_need(backend && ggml_backend_is_vk(backend),"Q8 component: Vulkan required");
    auto owner=std::make_unique<StrataVkLogitsGuard>(backend);
    auto& g=*owner;auto* ctx=static_cast<ggml_backend_vk_context*>(backend->context);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(limits.maxComputeWorkGroupInvocations>=256 && limits.maxComputeWorkGroupSize[0]>=256 &&
        limits.maxComputeSharedMemorySize>=256*4,"Q8 component: geometry unsupported");
    // This build's pin-checked factory always constructs the derived receipt.
    const auto& dot=static_cast<const StrataVkQ8Device*>(ctx->device.get())->strata_q8_dot;
    std::printf("native_q8_integer_dot requested=%d queried_support=%d logical_enabled=%d ggml_accelerated=%d device=%s\n",
        int(dot.requested),int(dot.supported),int(dot.enabled),int(ctx->device->integer_dot_product),ctx->device->properties.deviceName.data());
    dot.require_enabled(ctx->device->integer_dot_product);
    const auto properties=ctx->device->physical_device.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceVulkan12Properties>();
    const auto& fp=properties.get<vk::PhysicalDeviceVulkan12Properties>();q8_gu_require_controls(fp);
    logits_guard_need(effective_spv_path && *effective_spv_path,"Q8 component: effective shader receipt required");
    logits_guard_need(bool(fp.shaderDenormPreserveFloat16)==ctx->device->float_controls_denorm_preserve_fp16 &&
        bool(fp.shaderRoundingModeRTEFloat16)==ctx->device->float_controls_rte_fp16,"Q8 component: cached controls differ from query");
    g.metadata.reset(ggml_init({4096,nullptr,true}));logits_guard_need(bool(g.metadata),"Q8 component: metadata allocation failed");
    g.status=ggml_new_tensor_1d(g.metadata.get(),GGML_TYPE_I32,128);
    g.buffer.reset(ggml_backend_alloc_ctx_tensors(g.metadata.get(),backend));logits_guard_need(bool(g.buffer),"Q8 component: buffer allocation failed");
    g.bytes=ggml_backend_buffer_get_size(g.buffer.get());auto& pipe=g.pipeline;
    pipe->name=name;pipe->parameter_count=parameters;pipe->push_constant_size=push_words*4;
    pipe->wg_denoms={1,1,1};pipe->align=1;pipe->initialized=true;
    const auto code=q8_gu_float_controls(shader,shader_words);
    strata_vk_observation_create_pipeline_func(ctx->device,pipe,code.size()*4,code.data(),"main",parameters,{1,1,1},{},false,false,0,effective_spv_path);
    return owner;
}
StrataVkQ8Gu* strata_vk_q8_gu_create(ggml_backend_t backend,const char* effective_spv_path) {
    auto result=std::make_unique<StrataVkQ8Gu>();
    result->owner=strata_vk_q8_component_owner(backend,strata_q8_gu_spv,sizeof(strata_q8_gu_spv)/4,
        "strata_native_gu_q8_0_fullwarp_component",5,8,effective_spv_path);
    return result.release();
}
void strata_vk_q8_gu_destroy(StrataVkQ8Gu* p) noexcept {delete p;}
void strata_vk_q8_gu_run(StrataVkQ8Gu* s,const ggml_tensor* weights,const ggml_tensor* packed,
    ggml_tensor* gate,ggml_tensor* up,const Q8GuLayout& l) {
    logits_guard_need(s && weights && packed && gate && up,"Q8 GU: invalid call");
    const auto metadata=q8_gu_metadata(l,ggml_nbytes(weights));
    logits_guard_need(weights->type==GGML_TYPE_I32 && packed->type==GGML_TYPE_I32 &&
        gate->type==GGML_TYPE_F32 && up->type==GGML_TYPE_F32 &&
        ggml_nelements(packed)==int64_t(l.tokens)*(l.h/32)*9 &&
        ggml_nelements(gate)==int64_t(l.entries)*l.f && ggml_nelements(up)==ggml_nelements(gate),"Q8 GU: tensor shape/type mismatch");
    auto& g=*s->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(2*l.f/8<=limits.maxComputeWorkGroupCount[0] && l.grid_groups<=limits.maxComputeWorkGroupCount[1],"Q8 GU: grid exceeds limits");
    std::array<uint32_t,8> params{l.h,l.f};params[7]=0x3f800000;size_t at=2;
    const ggml_tensor* tensors[]={weights,packed,g.status,gate,up};
    for(const auto* t:tensors){
        logits_guard_need(t->buffer && ggml_is_contiguous(t) && ggml_backend_buffer_is_vk(t->buffer),"Q8 GU: contiguous Vulkan tensor required");
        auto* buffer=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        logits_guard_need(buffer->device.lock()==ctx->device,"Q8 GU: different device");
        const uint32_t bias=get_misalign_bytes(ctx,t);logits_guard_need(bias%4==0,"Q8 GU: word alignment required");params[at++]=bias/4;
    }
    const auto overlaps=[](const ggml_tensor* a,const ggml_tensor* b){if(a->buffer!=b->buffer)return false;
        const auto x=reinterpret_cast<uintptr_t>(a->data),y=reinterpret_cast<uintptr_t>(b->data);
        return x<=y?y-x<ggml_nbytes(a):x-y<ggml_nbytes(b);};
    logits_guard_need(!overlaps(gate,up) && !overlaps(gate,weights) && !overlaps(gate,packed) &&
        !overlaps(up,weights) && !overlaps(up,packed),"Q8 GU: overlapping output");
    ggml_backend_vk_synchronize(g.backend);ggml_backend_tensor_set(g.status,metadata.data(),0,sizeof(metadata));
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,weights,true),ggml_vk_tensor_subbuffer(ctx,packed,true),
        ggml_vk_tensor_subbuffer(ctx,g.status,true),ggml_vk_tensor_subbuffer(ctx,gate,true),ggml_vk_tensor_subbuffer(ctx,up,true)},
        params,{2*l.f/8,l.grid_groups,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
}
}
