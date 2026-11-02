#include "strata_vk_q8_wo.hpp"
#include "native_q8_wo_layout.hpp"
#include "native_qsa_qkv_layout.hpp"
#include "native_gdn_input_layout.hpp"
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
#include "strata_q8_wo_spv.hpp"
#include "strata_qsa_qkv_spv.hpp"
#endif
#include <limits>
namespace strata::core::portable_detail {
struct StrataVkQ8Wo {std::unique_ptr<StrataVkLogitsGuard> owner;size_t budget;bool qkv=false,gdn=false;uint64_t submitted=0,completed=0,submission_uncertain=0;};
StrataVkQ8Wo* strata_vk_q8_wo_create(ggml_backend_t backend,const char* path,size_t budget) {
#ifndef STRATA_NATIVE_Q8_STRICT_FMA
    (void)backend;(void)path;(void)budget;
    throw std::runtime_error("Q8 WO: strict FMA component build required");
#else
    auto result=std::make_unique<StrataVkQ8Wo>();result->budget=budget;
    result->owner=strata_vk_q8_component_owner(backend,strata_q8_wo_spv,sizeof(strata_q8_wo_spv)/4,
        NativeComponentAddressing::descriptor,"strata_native_wo_q8_0_candidate",3,4,path,budget,1);
    return result.release();
#endif
}
StrataVkQ8Wo* strata_vk_qsa_qkv_create(ggml_backend_t backend,const char* path,size_t budget) {
#ifndef STRATA_NATIVE_Q8_STRICT_FMA
    (void)backend;(void)path;(void)budget;
    throw std::runtime_error("QSA QKV: strict FMA component build required");
#else
    auto result=std::make_unique<StrataVkQ8Wo>();result->budget=budget;result->qkv=true;
    result->owner=strata_vk_q8_component_owner(backend,strata_qsa_qkv_spv,sizeof(strata_qsa_qkv_spv)/4,
        NativeComponentAddressing::descriptor,"strata_native_qsa_qkv_q8_0_candidate",3,4,path,budget,1);
    return result.release();
#endif
}
StrataVkQ8Wo* strata_vk_gdn_input_create(ggml_backend_t backend,const char* path,size_t budget) {
    auto* result=strata_vk_qsa_qkv_create(backend,path,budget);
    result->qkv=false;result->gdn=true;return result;
}
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo* p){return p?p->owner->bytes:0;}
uint64_t strata_vk_q8_wo_submitted(const StrataVkQ8Wo* p) noexcept{return p?p->submitted:0;}
uint64_t strata_vk_q8_wo_completed(const StrataVkQ8Wo* p) noexcept{return p?p->completed:0;}
uint64_t strata_vk_q8_wo_submission_uncertain(const StrataVkQ8Wo* p) noexcept{return p?p->submission_uncertain:0;}
void strata_vk_q8_wo_destroy(StrataVkQ8Wo* p) noexcept {delete p;}
void strata_vk_q8_wo_run(StrataVkQ8Wo* s,const ggml_tensor* weights,const ggml_tensor* packed,
                         ggml_tensor* output,size_t rows) {
    logits_guard_need(s && weights && packed && output,"Q8 WO: invalid call");
    if(s->gdn)gdn_input_require_layout(rows,ggml_nbytes(weights),ggml_nbytes(packed),ggml_nbytes(output));
    else if(s->qkv)qsa_qkv_require_layout(rows,ggml_nbytes(weights),ggml_nbytes(packed),ggml_nbytes(output));
    else q8_wo_require_layout(rows,ggml_nbytes(weights),ggml_nbytes(packed),ggml_nbytes(output));
    logits_guard_need(weights->type==GGML_TYPE_I32 && packed->type==GGML_TYPE_I32 && output->type==GGML_TYPE_F32,
        "Q8 WO: incorrect tensor type");
    auto& g=*s->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    logits_guard_need(rows<=ctx->device->properties.limits.maxComputeWorkGroupCount[0],"Q8 WO: grid exceeds limits");
    const ggml_tensor* tensors[]={weights,packed,output};std::array<uint32_t,4> params{uint32_t(rows)};
    size_t used=g.bytes;ggml_backend_buffer_t seen[3]{};size_t nseen=0;
    for(size_t i=0;i<3;++i){const auto* t=tensors[i];
        logits_guard_need(t->buffer && ggml_is_contiguous(t) && ggml_backend_buffer_is_vk(t->buffer),"Q8 WO: contiguous Vulkan tensors required");
        auto* buffer=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        logits_guard_need(buffer->device.lock()==ctx->device,"Q8 WO: different device");
        const uint32_t bias=get_misalign_bytes(ctx,t);logits_guard_need(bias%4==0,"Q8 WO: word alignment required");params[i+1]=bias/4;
        bool duplicate=false;for(size_t j=0;j<nseen;++j)duplicate|=seen[j]==t->buffer;
        if(!duplicate){seen[nseen++]=t->buffer;const size_t n=ggml_backend_buffer_get_size(t->buffer);
            logits_guard_need(n<=std::numeric_limits<size_t>::max()-used,"Q8 WO: budget overflow");used+=n;}
    }
    const auto span=[](const ggml_tensor* t){return Q8WoSpan{t->buffer,reinterpret_cast<uintptr_t>(t->data),ggml_nbytes(t)};};
    q8_wo_require_nonoverlap(span(weights),span(packed),span(output));
    q8_wo_require_budget(used,s->budget);
    ggml_backend_vk_synchronize(g.backend);ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);
    auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,weights,true),
        ggml_vk_tensor_subbuffer(ctx,packed,true),ggml_vk_tensor_subbuffer(ctx,output,true)},params,{uint32_t(rows),1,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();
    try { ggml_backend_vk_synchronize(g.backend); }
    catch(...) { ++s->submission_uncertain;throw; }
    ++s->submitted;++s->completed;
}
}
