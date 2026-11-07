#include "strata_vk_native_gdn_conv_rows.hpp"
#include "strata_native_gdn_conv_rows_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeGdnConvRows { std::unique_ptr<StrataVkLogitsGuard> owner; size_t budget=0; };
StrataVkNativeGdnConvRows* strata_vk_native_gdn_conv_rows_create(ggml_backend_t b,const char* path,size_t budget) {
 native_hc_need(b&&ggml_backend_is_vk(b),"native GDN conv rows: Vulkan backend required");
 auto*ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto&l=ctx->device->properties.limits;
 native_hc_need(l.maxComputeWorkGroupInvocations>=128&&l.maxComputeWorkGroupSize[0]>=128&&l.maxComputeSharedMemorySize>=512&&l.maxComputeWorkGroupCount[0]>=80&&l.maxComputeWorkGroupCount[1]>=2&&l.maxPerStageDescriptorStorageBuffers>=11&&l.maxDescriptorSetStorageBuffers>=11,"native GDN conv rows: unsupported workgroup/descriptors");
 auto p=std::make_unique<StrataVkNativeGdnConvRows>();p->budget=budget;
 p->owner=strata_vk_q8_component_owner(b,strata_native_gdn_conv_rows_spv,sizeof(strata_native_gdn_conv_rows_spv)/4,NativeComponentAddressing::descriptor,"strata_native_gdn_conv_rows",11,12,path,budget,1);return p.release();
}
void strata_vk_native_gdn_conv_rows_destroy(StrataVkNativeGdnConvRows*p)noexcept{delete p;}
size_t strata_vk_native_gdn_conv_rows_bytes(const StrataVkNativeGdnConvRows*p){return p?p->owner->bytes:0;}
void strata_vk_native_gdn_conv_rows_run(StrataVkNativeGdnConvRows*p,ggml_tensor*const t[10],uint32_t rows) {
 native_hc_need(p&&p->owner&&t&&(rows==1||rows==2),"native GDN conv rows: owner/tensors/rows");
 auto&g=*p->owner;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);const auto&l=ctx->device->properties.limits;
 native_hc_need(l.maxComputeWorkGroupCount[0]>=80&&l.maxComputeWorkGroupCount[1]>=rows,"native GDN conv rows: grid limit");
 const size_t counts[]={30720,10240*rows,40960,30720,10240*rows,10240*rows,10240*rows,10240*rows,32*rows,32*rows};
 std::array<uint32_t,12>off{};size_t used=g.bytes;ggml_backend_buffer_t seen[11]{};size_t nseen=0;
 auto validate=[&](const ggml_tensor*x,size_t bytes,ggml_type type){
  native_hc_need(x&&x->data&&x->type==type&&ggml_nbytes(x)==bytes&&ggml_is_contiguous(x)&&x->buffer&&ggml_backend_buffer_is_vk(x->buffer),"native GDN conv rows: typed span");
  auto*b=static_cast<ggml_backend_vk_buffer_context*>(x->buffer->context);native_hc_need(b&&b->device.lock()==ctx->device,"native GDN conv rows: foreign device");
  auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(x->buffer)),data=reinterpret_cast<uintptr_t>(x->data);size_t capacity=ggml_backend_buffer_get_size(x->buffer);
  native_hc_need(data>=base&&data-base<=capacity&&bytes<=capacity-(data-base),"native GDN conv rows: outside allocation");
  auto bias=get_misalign_bytes(ctx,x);native_hc_need(bias%4==0&&uint64_t(bias)+bytes<=l.maxStorageBufferRange,"native GDN conv rows: descriptor range");
  bool duplicate=false;for(size_t j=0;j<nseen;++j)duplicate|=seen[j]==x->buffer;if(!duplicate){seen[nseen++]=x->buffer;native_hc_need(used<=p->budget&&capacity<=p->budget-used,"native GDN conv rows: budget");used+=capacity;}
  return bias/4;
 };
 for(size_t i=0;i<10;++i){off[i]=validate(t[i],counts[i]*4,GGML_TYPE_F32);for(size_t j=0;j<i;++j)if(t[i]->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(t[i]->data),b=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need(a<=b?b-a>=counts[i]*4:a-b>=counts[j]*4,"native GDN conv rows: overlap");}}
 native_hc_need(g.status&&ggml_nbytes(g.status)>=4,"native GDN conv rows: status extent");
 const size_t status_bytes=ggml_nbytes(g.status);off[10]=validate(g.status,status_bytes,GGML_TYPE_I32);off[11]=rows;
 for(size_t i=0;i<10;++i)if(t[i]->buffer==g.status->buffer){auto a=reinterpret_cast<uintptr_t>(t[i]->data),b=reinterpret_cast<uintptr_t>(g.status->data);native_hc_need(a<=b?b-a>=counts[i]*4:a-b>=status_bytes,"native GDN conv rows: status overlap");}
 ggml_backend_vk_synchronize(g.backend);
 for(size_t i=0;i<3;++i){std::vector<float>v(counts[i]);ggml_backend_tensor_get(t[i],v.data(),0,v.size()*4);for(float x:v)native_hc_normal(x);}
 const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);
 ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
 ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3],true),ggml_vk_tensor_subbuffer(ctx,t[4],true),ggml_vk_tensor_subbuffer(ctx,t[5],true),ggml_vk_tensor_subbuffer(ctx,t[6],true),ggml_vk_tensor_subbuffer(ctx,t[7],true),ggml_vk_tensor_subbuffer(ctx,t[8],true),ggml_vk_tensor_subbuffer(ctx,t[9],true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{80,rows,1});
 ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);uint32_t status=0;ggml_backend_tensor_get(g.status,&status,0,4);native_hc_need(status==0,"native GDN conv rows: arithmetic status");
}
}
