// Standalone consumer of already packed inputs; included in the pin-checked TU.
#include "strata_vk_q8_gu.hpp"
#include "native_q8_address.hpp"
#include "strata_q8_gu_spv.hpp"
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
#include "strata_q8_gu_strict_fma_spv.hpp"
#endif
#include "native_q8_gu_spirv.hpp"
namespace strata::core::portable_detail {
size_t strata_vk_q8_resident_block_limit(ggml_backend_t backend) {
    logits_guard_need(backend && ggml_backend_is_vk(backend),"native Q8: Vulkan required");
    auto* ctx=static_cast<ggml_backend_vk_context*>(backend->context);
    q8_require_address_features(ctx->device->buffer_device_address,ctx->device->shader_int64);
    std::printf("native_q8_addressing logical_bufferDeviceAddress=%d logical_shaderInt64=%d shader64BitIndexing_required=false\n",
        int(ctx->device->buffer_device_address),int(ctx->device->shader_int64));
    return q8_resident_limit(ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend)),
        ctx->device->max_buffer_size,ctx->device->max_memory_allocation_size);
}
static uint64_t strata_vk_q8_weight_address(ggml_backend_vk_context* ctx,const ggml_tensor* t,uint64_t offset,size_t required) {
    logits_guard_need(t && t->buffer && t->type==GGML_TYPE_I32 && ggml_is_contiguous(t) &&
        ggml_backend_buffer_is_vk(t->buffer),"native Q8: contiguous Vulkan weight owner required");
    auto* owner=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
    logits_guard_need(owner->device.lock()==ctx->device && owner->dev_buffer &&
        owner->dev_buffer->device==ctx->device && owner->dev_buffer->buffer && owner->dev_buffer->device_memory,
        "native Q8: unavailable or foreign weight owner");
    return q8_weight_address(owner->dev_buffer->bda_addr,owner->dev_buffer->size,vk_tensor_offset(t),
        t->view_offs,ggml_nbytes(t),offset,required);
}
struct StrataVkQ8Gu {std::unique_ptr<StrataVkLogitsGuard> owner;};
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t backend,
    const uint32_t* shader,size_t shader_words,NativeComponentAddressing addressing,const char* name,uint32_t parameters,uint32_t push_words,const char* effective_spv_path,size_t budget=size_t(-1),size_t metadata_words=128) {
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
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
    const auto& fma=static_cast<const StrataVkQ8Device*>(ctx->device.get())->strata_q8_fma;
    fma.require_enabled();
    std::printf("native_q8_strict_fma requested=%d queried_support=%d logical_enabled=%d qualified=false\n",
        int(fma.requested),int(fma.supported),int(fma.enabled));
#endif
    const auto properties=ctx->device->physical_device.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceVulkan12Properties>();
    const auto& fp=properties.get<vk::PhysicalDeviceVulkan12Properties>();q8_gu_require_controls(fp);
    logits_guard_need(effective_spv_path && *effective_spv_path,"Q8 component: effective shader receipt required");
    logits_guard_need(bool(fp.shaderDenormPreserveFloat16)==ctx->device->float_controls_denorm_preserve_fp16 &&
        bool(fp.shaderRoundingModeRTEFloat16)==ctx->device->float_controls_rte_fp16,"Q8 component: cached controls differ from query");
    g.metadata.reset(ggml_init({4096,nullptr,true}));logits_guard_need(bool(g.metadata),"Q8 component: metadata allocation failed");
    g.status=ggml_new_tensor_1d(g.metadata.get(),GGML_TYPE_I32,metadata_words);
    const size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(g.metadata.get(),ggml_backend_get_default_buffer_type(backend));
    logits_guard_need(bound<=budget,"Q8 component: aligned metadata budget exceeded");
    g.buffer.reset(ggml_backend_alloc_ctx_tensors(g.metadata.get(),backend));logits_guard_need(bool(g.buffer),"Q8 component: buffer allocation failed");
    logits_guard_need(ggml_backend_buffer_get_size(g.buffer.get())<=budget,"Q8 component: actual metadata budget exceeded");
    g.bytes=ggml_backend_buffer_get_size(g.buffer.get());auto& pipe=g.pipeline;
    pipe->name=name;pipe->parameter_count=parameters;pipe->push_constant_size=push_words*4;
    pipe->wg_denoms={1,1,1};pipe->align=1;pipe->initialized=true;
    const auto code=native_component_float_controls(shader,shader_words,addressing);
    strata_vk_observation_create_pipeline_func(ctx->device,pipe,code.size()*4,code.data(),"main",parameters,{1,1,1},{},false,false,0,effective_spv_path);
    return owner;
}
StrataVkQ8Gu* strata_vk_q8_gu_create(ggml_backend_t backend,const char* effective_spv_path) {
    strata_vk_q8_resident_block_limit(backend);
    auto result=std::make_unique<StrataVkQ8Gu>();
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
    result->owner=strata_vk_q8_component_owner(backend,strata_q8_gu_strict_fma_spv,sizeof(strata_q8_gu_strict_fma_spv)/4,
        NativeComponentAddressing::physical64,"strata_native_gu_q8_0_strict_fma_component",4,7,effective_spv_path,size_t(-1),q8_metadata_words);
#else
    result->owner=strata_vk_q8_component_owner(backend,strata_q8_gu_spv,sizeof(strata_q8_gu_spv)/4,
        NativeComponentAddressing::physical64,"strata_native_gu_q8_0_fullwarp_component",4,7,effective_spv_path,size_t(-1),q8_metadata_words);
#endif
    return result.release();
}
size_t strata_vk_q8_gu_bytes(const StrataVkQ8Gu* p) {return p?ggml_backend_buffer_get_size(p->owner->buffer.get()):0;}
const ggml_tensor* strata_vk_q8_gu_metadata_tensor(const StrataVkQ8Gu* p) {return p?p->owner->status:nullptr;}
void strata_vk_q8_gu_destroy(StrataVkQ8Gu* p) noexcept {delete p;}
void strata_vk_q8_gu_run(StrataVkQ8Gu* s,const ggml_tensor* weights,const ggml_tensor* packed,
    ggml_tensor* gate,ggml_tensor* up,const Q8GuLayout& l) {
    strata_vk_q8_gu_run_blocks(s,std::vector<const ggml_tensor*>(l.weight_offsets.size(),weights),packed,gate,up,l);
}
void strata_vk_q8_gu_run_blocks(StrataVkQ8Gu* s,const std::vector<const ggml_tensor*>& weights,const ggml_tensor* packed,
    ggml_tensor* gate,ggml_tensor* up,const Q8GuLayout& l) {
    logits_guard_need(s && packed && gate && up,"Q8 GU: invalid call");
    logits_guard_need(weights.size()==l.weight_offsets.size(),"Q8 GU: weight owner count mismatch");
    std::vector<size_t> sizes;for(auto* t:weights){logits_guard_need(t,"Q8 GU: null weight owner");sizes.push_back(ggml_nbytes(t));}
    auto metadata=q8_gu_metadata(l,sizes);
    logits_guard_need(packed->type==GGML_TYPE_I32 &&
        gate->type==GGML_TYPE_F32 && up->type==GGML_TYPE_F32 &&
        ggml_nelements(packed)==int64_t(l.tokens)*(l.h/32)*9 &&
        ggml_nelements(gate)==int64_t(l.entries)*l.f && ggml_nelements(up)==ggml_nelements(gate),"Q8 GU: tensor shape/type mismatch");
    auto& g=*s->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(2*l.f/8<=limits.maxComputeWorkGroupCount[0] && l.grid_groups<=limits.maxComputeWorkGroupCount[1],"Q8 GU: grid exceeds limits");
    std::array<uint32_t,7> params{l.h,l.f};params[6]=0x3f800000;size_t at=2;
    const ggml_tensor* tensors[]={packed,g.status,gate,up};
    for(const auto* t:tensors){
        logits_guard_need(t->buffer && ggml_is_contiguous(t) && ggml_backend_buffer_is_vk(t->buffer),"Q8 GU: contiguous Vulkan tensor required");
        auto* buffer=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        logits_guard_need(buffer->device.lock()==ctx->device,"Q8 GU: different device");
        const uint32_t bias=get_misalign_bytes(ctx,t);logits_guard_need(bias%4==0,"Q8 GU: word alignment required");params[at++]=bias/4;
    }
    const auto overlaps=[](const ggml_tensor* a,const ggml_tensor* b){if(a->buffer!=b->buffer)return false;
        const auto x=reinterpret_cast<uintptr_t>(a->data),y=reinterpret_cast<uintptr_t>(b->data);
        return x<=y?y-x<ggml_nbytes(a):x-y<ggml_nbytes(b);};
    logits_guard_need(!overlaps(gate,up) && !overlaps(gate,packed) && !overlaps(up,packed),"Q8 GU: overlapping output");
    q8_require_address_features(ctx->device->buffer_device_address,ctx->device->shader_int64);
    for(size_t group=0;group<weights.size();++group) {
        logits_guard_need(!overlaps(gate,weights[group]) && !overlaps(up,weights[group]),"Q8 GU: overlapping weight owner");
        if(l.starts[group]!=l.starts[group+1])q8_store_address(metadata,group,
            strata_vk_q8_weight_address(ctx,weights[group],l.weight_offsets[group],2*size_t(l.f)*(l.h/32)*34));
    }
    logits_guard_need(ggml_nbytes(g.status)==sizeof(metadata),"Q8 component: metadata allocation/upload size mismatch");
    ggml_backend_vk_synchronize(g.backend);ggml_backend_tensor_set(g.status,metadata.data(),0,sizeof(metadata));
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,packed,true),
        ggml_vk_tensor_subbuffer(ctx,g.status,true),ggml_vk_tensor_subbuffer(ctx,gate,true),ggml_vk_tensor_subbuffer(ctx,up,true)},
        params,{2*l.f/8,l.grid_groups,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
}
}
