#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include "strata_native_hc_spv.hpp"
#include <cfenv>
namespace strata::core::portable_detail {
struct StrataVkNativeHc {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t backend,const char* path,size_t budget){
 native_hc_environment();auto p=std::make_unique<StrataVkNativeHc>();p->owner=strata_vk_q8_component_owner(backend,strata_native_hc_spv,sizeof(strata_native_hc_spv)/4,NativeComponentAddressing::descriptor,"strata_native_hc_norm_inject_observation",7,9,path,budget);return p.release();
}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc* p){return p?p->owner->bytes:0;}
void strata_vk_native_hc_destroy(StrataVkNativeHc* p) noexcept {delete p;}
void strata_vk_native_hc_run(StrataVkNativeHc* p,const ggml_tensor* residual,const ggml_tensor* norm_weight,const ggml_tensor* weight,ggml_tensor* normalized,ggml_tensor* rs,ggml_tensor* inject,uint32_t tokens){
 native_hc_environment();native_hc_shape(2560,4,tokens);native_hc_need(p&&std::fegetround()==FE_TONEAREST,"native HC: component and host round-to-nearest required");auto& g=*p->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
 const ggml_tensor* tensors[]={residual,norm_weight,weight,normalized,rs,inject};const uint64_t sizes[]={uint64_t(tokens)*40960,40960,81920,uint64_t(tokens)*40960,uint64_t(tokens)*16,uint64_t(tokens)*16};std::array<uint32_t,9> params{0,tokens,0,0,0,0,0,0,0};
 for(size_t i=0;i<6;++i){auto* t=tensors[i];native_hc_need(t&&t->type==(i==2?GGML_TYPE_I32:GGML_TYPE_F32)&&ggml_nbytes(t)==sizes[i],"native HC: tensor type/span mismatch");native_hc_need(t->buffer&&ggml_is_contiguous(t)&&ggml_backend_buffer_is_vk(t->buffer),"native HC: contiguous Vulkan spans required");auto* b=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native HC: foreign device");auto bias=get_misalign_bytes(ctx,t);native_hc_need(bias%4==0,"native HC: word alignment");params[2+i]=bias/4;
  for(size_t j=0;j<i;++j)if(t->buffer==tensors[j]->buffer){auto x=reinterpret_cast<uintptr_t>(t->data),y=reinterpret_cast<uintptr_t>(tensors[j]->data);native_hc_need(!(x<=y?y-x<ggml_nbytes(t):x-y<ggml_nbytes(tensors[j])),"native HC: overlapping spans");}}
 auto& limits=ctx->device->properties.limits;native_hc_need(limits.maxComputeWorkGroupCount[0]>=4&&limits.maxComputeWorkGroupCount[1]>=tokens&&limits.maxComputeSharedMemorySize>=4112,"native HC: grid/shared memory unsupported");
 // Bounded normal-domain preflight, not a CPU fallback or numeric acceptance tolerance.
 ggml_backend_vk_synchronize(g.backend);std::vector<float> x(size_t(tokens)*10240),w(10240);std::vector<uint16_t> wi(40960);ggml_backend_tensor_get(residual,x.data(),0,x.size()*4);ggml_backend_tensor_get(norm_weight,w.data(),0,w.size()*4);ggml_backend_tensor_get(weight,wi.data(),0,wi.size()*2);(void)native_hc_oracle(x,w,wi,tokens);
 const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);params[8]=get_misalign_bytes(ctx,g.status)/4;
 for(uint32_t mode=0;mode<2;++mode){params[0]=mode;ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
  ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,residual,true),ggml_vk_tensor_subbuffer(ctx,norm_weight,true),ggml_vk_tensor_subbuffer(ctx,weight,true),ggml_vk_tensor_subbuffer(ctx,normalized,true),ggml_vk_tensor_subbuffer(ctx,rs,true),ggml_vk_tensor_subbuffer(ctx,inject,true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},params,{mode?4u:1u,tokens,1});ggml_vk_sync_buffers(ctx,compute);compute.reset();}
 ggml_backend_vk_synchronize(g.backend);uint32_t status=0;ggml_backend_tensor_get(g.status,&status,0,4);native_hc_need(status==0,"native HC: device nonfinite/subnormal arithmetic; no result accepted");
}
}
#include "strata_vk_native_hc_mix.inc"
