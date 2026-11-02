#include "strata_vk_native_shared.hpp"
#include "native_shared_layout.hpp"
#include "strata_native_shared_spv.hpp"
#include "strata_qsa_indexer_bf16_spv.hpp"
#include <limits>
namespace strata::core::portable_detail {
struct StrataVkNativeShared {std::unique_ptr<StrataVkLogitsGuard> owner;bool matrix=false;size_t budget=0;uint64_t submitted=0,completed=0,uncertain=0;};
StrataVkNativeShared* strata_vk_native_shared_create(ggml_backend_t backend,const char* path,size_t budget) {
    auto result=std::make_unique<StrataVkNativeShared>();
    result->owner=strata_vk_q8_component_owner(backend,strata_native_shared_spv,sizeof(strata_native_shared_spv)/4,
        NativeComponentAddressing::descriptor,"strata_native_shared_dense_mmvq_mmvf_observation",4,9,path,budget);
    return result.release();
}
StrataVkNativeShared* strata_vk_qsa_indexer_bf16_create(ggml_backend_t backend,const char* path,size_t budget) {
    auto result=std::make_unique<StrataVkNativeShared>();result->matrix=true;result->budget=budget;
    result->owner=strata_vk_q8_component_owner(backend,strata_qsa_indexer_bf16_spv,sizeof(strata_qsa_indexer_bf16_spv)/4,
        NativeComponentAddressing::descriptor,"strata_native_qsa_indexer_bf16_mmvf_candidate",4,9,path,budget);
    return result.release();
}
uint64_t strata_vk_native_shared_submitted(const StrataVkNativeShared* p) noexcept{return p?p->submitted:0;}
uint64_t strata_vk_native_shared_completed(const StrataVkNativeShared* p) noexcept{return p?p->completed:0;}
uint64_t strata_vk_native_shared_uncertain(const StrataVkNativeShared* p) noexcept{return p?p->uncertain:0;}
size_t strata_vk_native_shared_bytes(const StrataVkNativeShared* p) {return p?p->owner->bytes:0;}
void strata_vk_native_shared_destroy(StrataVkNativeShared* p) noexcept {delete p;}
void strata_vk_native_shared_run(StrataVkNativeShared* s,uint32_t mode,const ggml_tensor* weights,
    const ggml_tensor* input,ggml_tensor* output,ggml_tensor* aux,uint32_t ni,uint32_t no,uint32_t tokens) {
    native_shared_need(s && mode<=2 && weights && input && output && aux,"native shared: invalid call");
    native_shared_shape(ni,mode==0?no:32,tokens);
    if(s->matrix)native_shared_need(mode==1 && ni==2560 && (no==128 || no==512) && tokens==1,"QSA indexer BF16: unsupported geometry");
    else native_shared_need(mode==0 || no==1,"native shared: scalar/scale require no=1");
    native_shared_need(weights->type==(mode==2?GGML_TYPE_F32:GGML_TYPE_I32) &&
        input->type==(mode==0?GGML_TYPE_I32:GGML_TYPE_F32) && output->type==GGML_TYPE_F32 && aux->type==GGML_TYPE_F32,
        "native shared: tensor type mismatch");
    const uint64_t weight_bytes=mode==0?native_shared_weight_bytes(ni,no):mode==1?uint64_t(ni)*2*(s->matrix?no:1):uint64_t(tokens)*4;
    const uint64_t in_bytes=mode==0?uint64_t(tokens)*(ni/32)*36:uint64_t(tokens)*ni*4;
    const uint64_t out_bytes=uint64_t(tokens)*(mode==0?no:mode==1?(s->matrix?no:1):ni)*4;
    native_shared_need(ggml_nbytes(weights)==weight_bytes && ggml_nbytes(input)==in_bytes && ggml_nbytes(output)==out_bytes &&
        ggml_nbytes(aux)==uint64_t(mode==2?tokens:1)*4,"native shared: tensor span mismatch");
    auto& g=*s->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const uint32_t grid=mode==0?no:mode==1?(s->matrix?no:1):(ni+255)/256;
    const auto& limits=ctx->device->properties.limits;
    native_shared_need(grid<=limits.maxComputeWorkGroupCount[0] && tokens<=limits.maxComputeWorkGroupCount[1] &&
        limits.maxComputeSharedMemorySize>=288*4,"native shared: device grid/shared memory unsupported");
    std::array<uint32_t,9> params{mode,ni,no,tokens,0,0,0,0,native_shared_mmvf_block(ni)};
    const ggml_tensor* tensors[]={weights,input,output,aux};
    size_t used=g.bytes;ggml_backend_buffer_t seen[4]{};size_t nseen=0;
    for(size_t i=0;i<4;++i) {
        const auto* t=tensors[i];
        native_shared_need(t->buffer && ggml_is_contiguous(t) && ggml_backend_buffer_is_vk(t->buffer),"native shared: contiguous Vulkan tensors required");
        auto* buffer=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        native_shared_need(buffer->device.lock()==ctx->device,"native shared: different device");
        const uint32_t bias=get_misalign_bytes(ctx,t);native_shared_need(bias%4==0,"native shared: word alignment required");params[4+i]=bias/4;
        if(s->matrix){bool duplicate=false;for(size_t j=0;j<nseen;++j)duplicate|=seen[j]==t->buffer;
            if(!duplicate){seen[nseen++]=t->buffer;size_t n=ggml_backend_buffer_get_size(t->buffer);native_shared_need(n<=std::numeric_limits<size_t>::max()-used,"QSA indexer BF16: budget overflow");used+=n;}}
        for(size_t j=0;j<i;++j)if(t->buffer==tensors[j]->buffer) {
            const auto x=reinterpret_cast<uintptr_t>(t->data),y=reinterpret_cast<uintptr_t>(tensors[j]->data);
            native_shared_need(!(x<=y?y-x<ggml_nbytes(t):x-y<ggml_nbytes(tensors[j])),"native shared: overlapping spans");
        }
    }
    if(s->matrix)native_shared_need(used<=s->budget,"QSA indexer BF16: allocation exceeds budget");
    ggml_backend_vk_synchronize(g.backend);ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);
    auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    try{
        ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,weights,true),ggml_vk_tensor_subbuffer(ctx,input,true),
            ggml_vk_tensor_subbuffer(ctx,output,true),ggml_vk_tensor_subbuffer(ctx,aux,true)},params,{grid,tokens,1});
        ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
    }catch(...){if(s->matrix)++s->uncertain;throw;}
    if(s->matrix){++s->submitted;++s->completed;}
}
}
