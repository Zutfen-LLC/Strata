#include "strata_vk_native_gdn_log1p.hpp"
#include "native_gdn_log1p_domain.hpp"
#include "strata_native_gdn_log1p_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeGdnLog1p{std::unique_ptr<StrataVkLogitsGuard>owner;};
StrataVkNativeGdnLog1p* strata_vk_native_gdn_log1p_create(ggml_backend_t b,const char*path,size_t budget){auto p=std::make_unique<StrataVkNativeGdnLog1p>();p->owner=strata_vk_q8_component_owner(b,strata_native_gdn_log1p_spv,sizeof(strata_native_gdn_log1p_spv)/4,NativeComponentAddressing::descriptor,"strata_native_gdn_log1p",7,7,path,budget);return p.release();}
void strata_vk_native_gdn_log1p_destroy(StrataVkNativeGdnLog1p*p)noexcept{delete p;}
uint32_t strata_vk_native_gdn_log1p_run(StrataVkNativeGdnLog1p*p,ggml_tensor*const t[6]){native_hc_need(p,"GDN log1p: owner");auto&g=*p->owner;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);std::array<uint32_t,7>off{};
 for(size_t i=0;i<6;++i){auto*x=t[i];native_hc_need(x&&x->type==GGML_TYPE_F32&&ggml_nbytes(x)==1024&&ggml_is_contiguous(x)&&x->buffer&&ggml_backend_buffer_is_vk(x->buffer),"GDN log1p: fixed contiguous F32[256] Vulkan spans");auto*b=static_cast<ggml_backend_vk_buffer_context*>(x->buffer->context);native_hc_need(b->device.lock()==ctx->device,"GDN log1p: foreign device");auto bias=get_misalign_bytes(ctx,x);native_hc_need(bias%4==0,"GDN log1p: alignment");off[i]=bias/4;
  for(size_t j=0;j<i;++j)if(x->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(x->data),b=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need((a>b?a-b:b-a)>=1024,"GDN log1p: overlapping input/output spans");}}
 ggml_backend_vk_synchronize(g.backend);std::array<float,256>input;ggml_backend_tensor_get(t[0],input.data(),0,1024);for(float x:input)native_gdn_log1p_input(x);
 uint32_t status=0;ggml_backend_tensor_set(g.status,&status,0,4);off[6]=get_misalign_bytes(ctx,g.status)/4;
 ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3],true),ggml_vk_tensor_subbuffer(ctx,t[4],true),ggml_vk_tensor_subbuffer(ctx,t[5],true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{1,1,1});ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);ggml_backend_tensor_get(g.status,&status,0,4);return status;
}
}
