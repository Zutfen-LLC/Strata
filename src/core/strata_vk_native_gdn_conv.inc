#include "strata_vk_native_gdn_conv.hpp"
#include "strata_native_gdn_conv_t1_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeGdnConv {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkNativeGdnConv* strata_vk_native_gdn_conv_create(ggml_backend_t b,const char*path,size_t budget){native_hc_need(b&&ggml_backend_is_vk(b),"native GDN conv: Vulkan backend required");auto*ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto&l=ctx->device->properties.limits;native_hc_need(l.maxComputeWorkGroupInvocations>=128&&l.maxComputeWorkGroupSize[0]>=128&&l.maxComputeSharedMemorySize>=512&&l.maxPerStageDescriptorStorageBuffers>=11&&l.maxDescriptorSetStorageBuffers>=11,"native GDN conv: unsupported workgroup/descriptors");auto p=std::make_unique<StrataVkNativeGdnConv>();p->owner=strata_vk_q8_component_owner(b,strata_native_gdn_conv_t1_spv,sizeof(strata_native_gdn_conv_t1_spv)/4,NativeComponentAddressing::descriptor,"strata_native_gdn_conv_t1",11,12,path,budget);return p.release();}
void strata_vk_native_gdn_conv_destroy(StrataVkNativeGdnConv*p)noexcept{delete p;}
size_t strata_vk_native_gdn_conv_bytes(const StrataVkNativeGdnConv*p){return p?p->owner->bytes:0;}
void strata_vk_native_gdn_conv_run(StrataVkNativeGdnConv*p,ggml_tensor*const t[10],bool commit){
 native_hc_need(p,"native GDN conv: owner required");auto&g=*p->owner;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);const auto&l=ctx->device->properties.limits;
 native_hc_need(l.maxComputeWorkGroupSize[0]>=128&&l.maxComputeSharedMemorySize>=512&&l.maxPerStageDescriptorStorageBuffers>=11&&l.maxDescriptorSetStorageBuffers>=11,"native GDN conv: workgroup/descriptors");
 const size_t count[]={30720,10240,40960,30720,10240,10240,10240,10240,32,32};std::array<uint32_t,12>off{};
 for(size_t i=0;i<10;++i){auto*x=t[i];native_hc_need(x&&x->type==GGML_TYPE_F32&&ggml_nbytes(x)==count[i]*4&&ggml_is_contiguous(x)&&x->buffer&&ggml_backend_buffer_is_vk(x->buffer),"native GDN conv: exact contiguous Vulkan F32 shape");auto*b=static_cast<ggml_backend_vk_buffer_context*>(x->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native GDN conv: foreign device");auto bias=get_misalign_bytes(ctx,x);native_hc_need(bias%4==0,"native GDN conv: alignment");off[i]=bias/4;
  for(size_t j=0;j<i;++j)if(x->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(x->data),b=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need(!(a<=b?b-a<ggml_nbytes(x):a-b<ggml_nbytes(t[j])),"native GDN conv: overlapping history/candidate/input spans");}}
 ggml_backend_vk_synchronize(g.backend);
 for(size_t i=0;i<3;++i){std::vector<float>v(count[i]);ggml_backend_tensor_get(t[i],v.data(),0,v.size()*4);for(float x:v)native_hc_normal(x);}
 const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);off[10]=get_misalign_bytes(ctx,g.status)/4;off[11]=commit;
 ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
 ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3],true),ggml_vk_tensor_subbuffer(ctx,t[4],true),ggml_vk_tensor_subbuffer(ctx,t[5],true),ggml_vk_tensor_subbuffer(ctx,t[6],true),ggml_vk_tensor_subbuffer(ctx,t[7],true),ggml_vk_tensor_subbuffer(ctx,t[8],true),ggml_vk_tensor_subbuffer(ctx,t[9],true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{80,1,1});ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);uint32_t status;ggml_backend_tensor_get(g.status,&status,0,4);native_hc_need(!status,"native GDN conv: unsupported numerical candidate; input history unchanged");
}
}
