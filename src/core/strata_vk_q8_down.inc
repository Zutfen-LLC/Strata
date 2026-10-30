#include "strata_vk_q8_down.hpp"
#include "strata_q8_down_spv.hpp"
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
#include "strata_q8_down_strict_fma_spv.hpp"
#endif
namespace strata::core::portable_detail {
struct StrataVkQ8Down {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkQ8Down* strata_vk_q8_down_create(ggml_backend_t backend,const char* path) {
    strata_vk_q8_resident_block_limit(backend);
    auto result=std::make_unique<StrataVkQ8Down>();
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
    result->owner=strata_vk_q8_component_owner(backend,strata_q8_down_strict_fma_spv,sizeof(strata_q8_down_strict_fma_spv)/4,
        "strata_native_down_q8_0_strict_fma_component",3,6,path,size_t(-1),q8_metadata_words);
#else
    result->owner=strata_vk_q8_component_owner(backend,strata_q8_down_spv,sizeof(strata_q8_down_spv)/4,
        "strata_native_down_q8_0_fullwarp_component",3,6,path,size_t(-1),q8_metadata_words);
#endif
    return result.release();
}
size_t strata_vk_q8_down_bytes(const StrataVkQ8Down* p) {return p?ggml_backend_buffer_get_size(p->owner->buffer.get()):0;}
const ggml_tensor* strata_vk_q8_down_metadata_tensor(const StrataVkQ8Down* p) {return p?p->owner->status:nullptr;}
void strata_vk_q8_down_destroy(StrataVkQ8Down* p) noexcept {delete p;}
void strata_vk_q8_down_run(StrataVkQ8Down* s,const ggml_tensor* weights,const ggml_tensor* packed,
    ggml_tensor* output,const Q8DownLayout& l) {
    strata_vk_q8_down_run_blocks(s,std::vector<const ggml_tensor*>(l.weight_offsets.size(),weights),packed,output,l);
}
void strata_vk_q8_down_run_blocks(StrataVkQ8Down* s,const std::vector<const ggml_tensor*>& weights,const ggml_tensor* packed,
    ggml_tensor* output,const Q8DownLayout& l) {
    logits_guard_need(s && packed && output,"Q8 down: invalid call");
    logits_guard_need(weights.size()==l.weight_offsets.size(),"Q8 down: weight owner count mismatch");
    std::vector<size_t> sizes;for(auto* t:weights){logits_guard_need(t,"Q8 down: null weight owner");sizes.push_back(ggml_nbytes(t));}
    auto metadata=q8_down_metadata(l,sizes);
    logits_guard_need(packed->type==GGML_TYPE_I32 &&
        output->type==GGML_TYPE_F32 && ggml_nelements(packed)==int64_t(l.entries)*(l.f/32)*9 &&
        ggml_nelements(output)==int64_t(l.destinations)*l.h,"Q8 down: tensor shape/type mismatch");
    auto& g=*s->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(l.h/8<=limits.maxComputeWorkGroupCount[0] && l.grid_groups<=limits.maxComputeWorkGroupCount[1],"Q8 down: grid exceeds limits");
    std::array<uint32_t,6> params{l.h,l.f};params[5]=0x3f800000;size_t at=2;
    const ggml_tensor* tensors[]={packed,g.status,output};
    for(const auto* t:tensors){
        logits_guard_need(t->buffer && ggml_is_contiguous(t) && ggml_backend_buffer_is_vk(t->buffer),"Q8 down: contiguous Vulkan tensor required");
        auto* buffer=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        logits_guard_need(buffer->device.lock()==ctx->device,"Q8 down: different device");
        const uint32_t bias=get_misalign_bytes(ctx,t);logits_guard_need(bias%4==0,"Q8 down: word alignment required");params[at++]=bias/4;
    }
    const auto overlaps=[](const ggml_tensor* a,const ggml_tensor* b){if(a->buffer!=b->buffer)return false;
        const auto x=reinterpret_cast<uintptr_t>(a->data),y=reinterpret_cast<uintptr_t>(b->data);
        return x<=y?y-x<ggml_nbytes(a):x-y<ggml_nbytes(b);};
    logits_guard_need(!overlaps(output,packed),"Q8 down: overlapping output");
    q8_require_address_features(ctx->device->buffer_device_address,ctx->device->shader_int64);
    for(size_t group=0;group<weights.size();++group) {
        logits_guard_need(!overlaps(output,weights[group]),"Q8 down: overlapping weight owner");
        if(l.starts[group]!=l.starts[group+1])q8_store_address(metadata,group,
            strata_vk_q8_weight_address(ctx,weights[group],l.weight_offsets[group],size_t(l.h)*(l.f/32)*34));
    }
    logits_guard_need(ggml_nbytes(g.status)==sizeof(metadata),"Q8 component: metadata allocation/upload size mismatch");
    ggml_backend_vk_synchronize(g.backend);ggml_backend_tensor_set(g.status,metadata.data(),0,sizeof(metadata));
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,packed,true),
        ggml_vk_tensor_subbuffer(ctx,g.status,true),ggml_vk_tensor_subbuffer(ctx,output,true)},
        params,{l.h/8,l.grid_groups,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
}
}
