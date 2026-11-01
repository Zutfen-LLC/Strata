#include "strata_vk_native_gdn.hpp"
#include "strata_native_gdn_t1_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeGdn {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkNativeGdn* strata_vk_native_gdn_create(ggml_backend_t b,const char*path,size_t budget){auto p=std::make_unique<StrataVkNativeGdn>();p->owner=strata_vk_q8_component_owner(b,strata_native_gdn_t1_spv,sizeof(strata_native_gdn_t1_spv)/4,NativeComponentAddressing::descriptor,"strata_native_gdn_t1",11,14,path,budget);return p.release();}
void strata_vk_native_gdn_destroy(StrataVkNativeGdn*p)noexcept{delete p;}
size_t strata_vk_native_gdn_bytes(const StrataVkNativeGdn*p){return p?p->owner->bytes:0;}
void strata_vk_native_gdn_run(StrataVkNativeGdn*p,ggml_tensor*const t[13],ggml_tensor*aux){
 native_hc_need(p,"native GDN: owner required");auto&g=*p->owner;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
 const size_t count[]={786432,10240,48,48,6144,128,786432,6144,48,6144,49,6144,6144};std::array<uint32_t,14>off{};
 const auto&l=ctx->device->properties.limits;native_hc_need(l.maxComputeWorkGroupInvocations>=512&&l.maxComputeWorkGroupSize[0]>=128&&l.maxComputeWorkGroupSize[1]>=4&&l.maxComputeSharedMemorySize>=6160&&l.maxPerStageDescriptorStorageBuffers>=11&&l.maxDescriptorSetStorageBuffers>=11,"native GDN: S128/RG4 workgroup required");
 for(size_t i=0;i<13;++i){auto*x=t[i];native_hc_need(x&&x->type==GGML_TYPE_F32&&ggml_nbytes(x)==count[i]*4&&ggml_is_contiguous(x)&&x->buffer&&ggml_backend_buffer_is_vk(x->buffer),"native GDN: exact F32 Vulkan spans required");auto*b=static_cast<ggml_backend_vk_buffer_context*>(x->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native GDN: foreign device");auto bias=get_misalign_bytes(ctx,x);native_hc_need(bias%4==0,"native GDN: alignment");off[i]=bias/4;
 for(size_t j=0;j<i;++j)if(x->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(x->data),b=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need(!(a<=b?b-a<ggml_nbytes(x):a-b<ggml_nbytes(t[j])),"native GDN: overlapping spans");}}
 native_hc_need(aux&&aux->type==GGML_TYPE_F32&&ggml_is_contiguous(aux)&&aux->buffer&&ggml_backend_buffer_is_vk(aux->buffer),"native GDN: packed auxiliary descriptor");
 for(size_t i:{8,10,11,12}){auto a=reinterpret_cast<uintptr_t>(aux->data),b=reinterpret_cast<uintptr_t>(t[i]->data);native_hc_need(t[i]->buffer==aux->buffer&&b>=a&&b-a<=ggml_nbytes(aux)&&ggml_nbytes(t[i])<=ggml_nbytes(aux)-(b-a),"native GDN: auxiliary view range");off[i]=(get_misalign_bytes(ctx,aux)+b-a)/4;}
 ggml_backend_vk_synchronize(g.backend);
 // Read one input at a time; 3 MiB maximum temporary, included by the layer host bound.
 for(size_t i=0;i<6;++i){std::vector<float>v(count[i]);ggml_backend_tensor_get(t[i],v.data(),0,v.size()*4);for(float x:v){native_hc_normal(x);if(i==2)native_hc_need(x<=0.f&&x>=-87.33651733398438f,"native GDN: decay exp domain");if(i==4)native_hc_need(std::fabs(x)<=87.33651733398438f,"native GDN: sigmoid normal domain");}}
 const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);off[13]=get_misalign_bytes(ctx,g.status)/4;
 ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
 ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3],true),ggml_vk_tensor_subbuffer(ctx,t[4],true),ggml_vk_tensor_subbuffer(ctx,t[5],true),ggml_vk_tensor_subbuffer(ctx,t[6],true),ggml_vk_tensor_subbuffer(ctx,t[7],true),ggml_vk_tensor_subbuffer(ctx,aux,true),ggml_vk_tensor_subbuffer(ctx,t[9],true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{48,1,1});ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);uint32_t status;ggml_backend_tensor_get(g.status,&status,0,4);native_hc_need(status==0,"native GDN: nonfinite/subnormal candidate refused; committed state unchanged");
}
}
