#include "strata_vk_native_ple_conv.hpp"
#include "strata_native_ple_conv_t1_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativePleConv {std::unique_ptr<StrataVkLogitsGuard> owner;size_t budget=0;};
StrataVkNativePleConv* strata_vk_native_ple_conv_create(ggml_backend_t b,const char* path,size_t budget) {
 native_hc_need(b&&ggml_backend_is_vk(b),"native PLE conv: Vulkan backend required");
 auto*ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto&l=ctx->device->properties.limits;
 native_hc_need(l.maxComputeWorkGroupInvocations>=256&&l.maxComputeWorkGroupSize[0]>=256&&l.maxComputeWorkGroupCount[0]>=40&&l.maxPerStageDescriptorStorageBuffers>=10&&l.maxDescriptorSetStorageBuffers>=10,"native PLE conv: unsupported workgroup/descriptors");
 auto p=std::make_unique<StrataVkNativePleConv>();p->budget=budget;
 p->owner=strata_vk_q8_component_owner(b,strata_native_ple_conv_t1_spv,sizeof(strata_native_ple_conv_t1_spv)/4,NativeComponentAddressing::descriptor,"strata_native_ple_conv_t1_observation",10,10,path,budget,1);return p.release();
}
void strata_vk_native_ple_conv_destroy(StrataVkNativePleConv*p)noexcept{delete p;}
size_t strata_vk_native_ple_conv_bytes(const StrataVkNativePleConv*p){return p?p->owner->bytes:0;}
void strata_vk_native_ple_conv_run(StrataVkNativePleConv*p,ggml_tensor* const t[10]) {
 native_hc_need(p&&t,"native PLE conv: owner and tensors required");auto&g=*p->owner;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
 const size_t counts[]={92160,10240,20480,10240,10240,92160,10240,10240,10240,10240};std::array<uint32_t,10>off{};size_t used=g.bytes;ggml_backend_buffer_t seen[10]{};size_t seen_count=0;
 for(size_t i=0;i<10;++i){auto*x=t[i];native_hc_need(x&&x->type==(i==2?GGML_TYPE_I32:GGML_TYPE_F32)&&ggml_nbytes(x)==counts[i]*4&&ggml_is_contiguous(x)&&x->buffer&&ggml_backend_buffer_is_vk(x->buffer),"native PLE conv: contiguous typed Vulkan span required");auto*b=static_cast<ggml_backend_vk_buffer_context*>(x->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native PLE conv: foreign device");auto bias=get_misalign_bytes(ctx,x);native_hc_need(bias%4==0&&uint64_t(bias)+ggml_nbytes(x)<=ctx->device->properties.limits.maxStorageBufferRange,"native PLE conv: alignment/range");off[i]=bias/4;
  bool duplicate=false;for(size_t j=0;j<seen_count;++j)duplicate|=seen[j]==x->buffer;if(!duplicate){seen[seen_count++]=x->buffer;auto bytes=ggml_backend_buffer_get_size(x->buffer);native_hc_need(used<=p->budget&&bytes<=p->budget-used,"native PLE conv: allocation budget");used+=bytes;}
  for(size_t j=0;j<i;++j)if(x->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(x->data),b0=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need(!(a<=b0?b0-a<ggml_nbytes(x):a-b0<ggml_nbytes(t[j])),"native PLE conv: overlapping spans");}}
 ggml_backend_vk_synchronize(g.backend);
 for(size_t i:{size_t(0),size_t(1),size_t(3),size_t(4)}){std::vector<float>v(counts[i]);ggml_backend_tensor_get(t[i],v.data(),0,v.size()*4);for(float x:v)native_hc_normal(x);}
 std::vector<uint32_t>w(counts[2]);ggml_backend_tensor_get(t[2],w.data(),0,w.size()*4);for(auto x:w)native_hc_need((x&0x7c00u)!=0x7c00u&&((x>>16)&0x7c00u)!=0x7c00u,"native PLE conv: nonfinite F16 weight");
 ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
 ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3],true),ggml_vk_tensor_subbuffer(ctx,t[4],true),ggml_vk_tensor_subbuffer(ctx,t[5],true),ggml_vk_tensor_subbuffer(ctx,t[6],true),ggml_vk_tensor_subbuffer(ctx,t[7],true),ggml_vk_tensor_subbuffer(ctx,t[8],true),ggml_vk_tensor_subbuffer(ctx,t[9],true)},off,{40,1,1});
 ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
}
}
