#include "native_hc_pending.hpp"
#include "strata_native_hc_pending_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeHcPending {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkNativeHcPending* strata_vk_native_hc_pending_create(ggml_backend_t backend,const char* path,size_t budget){
 native_hc_environment();auto p=std::make_unique<StrataVkNativeHcPending>();p->owner=strata_vk_q8_component_owner(backend,strata_native_hc_pending_spv,sizeof(strata_native_hc_pending_spv)/4,NativeComponentAddressing::descriptor,"strata_native_hc_pending_observation",7,7,path,budget);return p.release();
}
size_t strata_vk_native_hc_pending_bytes(const StrataVkNativeHcPending* p){return p?p->owner->bytes:0;}
void strata_vk_native_hc_pending_destroy(StrataVkNativeHcPending* p) noexcept {delete p;}
void strata_vk_native_hc_pending_run(StrataVkNativeHcPending* p,const ggml_tensor* residual,const ggml_tensor* bo,const ggml_tensor* inject,ggml_tensor* candidate,ggml_tensor* gate,ggml_tensor* exponential){
 native_hc_environment();native_hc_need(p&&std::fegetround()==FE_TONEAREST,"HC pending: owner/RNE required");auto& g=*p->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
 const ggml_tensor* tensors[]={residual,bo,inject,candidate,gate,exponential};const size_t sizes[]={40960,10240,16,40960,16,16};std::array<uint32_t,7> params{};
 for(size_t i=0;i<6;++i){auto* t=tensors[i];native_hc_need(t&&t->type==GGML_TYPE_F32&&ggml_nbytes(t)==sizes[i],"HC pending: tensor type/span");native_hc_need(t->buffer&&ggml_is_contiguous(t)&&ggml_backend_buffer_is_vk(t->buffer),"HC pending: contiguous Vulkan spans");auto* b=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);native_hc_need(b->device.lock()==ctx->device,"HC pending: foreign device");const auto bias=get_misalign_bytes(ctx,t);native_hc_need(bias%4==0&&uint64_t(bias)/4<=UINT32_MAX,"HC pending: alignment/offset range");params[i]=bias/4;
  for(size_t j=0;j<i;++j)if(t->buffer==tensors[j]->buffer){const auto x=reinterpret_cast<uintptr_t>(t->data),y=reinterpret_cast<uintptr_t>(tensors[j]->data);native_hc_need(!(x<=y?y-x<ggml_nbytes(t):x-y<ggml_nbytes(tensors[j])),"HC pending: overlapping input/candidate spans");}}
 const auto& limits=ctx->device->properties.limits;native_hc_need(limits.maxComputeWorkGroupCount[0]>=4&&limits.maxComputeWorkGroupInvocations>=256&&limits.maxComputeWorkGroupSize[0]>=256&&limits.maxComputeSharedMemorySize>=4,"HC pending: grid/shared unsupported");
 ggml_backend_vk_synchronize(g.backend);std::vector<float> r(10240),b(2560),i(4);ggml_backend_tensor_get(residual,r.data(),0,40960);ggml_backend_tensor_get(bo,b.data(),0,10240);ggml_backend_tensor_get(inject,i.data(),0,16);(void)native_hc_pending_control(r,b,i);
 const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);params[6]=get_misalign_bytes(ctx,g.status)/4;ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
 ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,residual,true),ggml_vk_tensor_subbuffer(ctx,bo,true),ggml_vk_tensor_subbuffer(ctx,inject,true),ggml_vk_tensor_subbuffer(ctx,candidate,true),ggml_vk_tensor_subbuffer(ctx,gate,true),ggml_vk_tensor_subbuffer(ctx,exponential,true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},params,{4,1,1});ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
 uint32_t status=0;ggml_backend_tensor_get(g.status,&status,0,4);native_hc_need(status==0,"HC pending: arithmetic domain refusal; no candidate accepted");
}
}
