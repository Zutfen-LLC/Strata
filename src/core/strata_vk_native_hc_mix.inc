#include "strata_native_hc_mix_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeHcMix {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkNativeHcMix* strata_vk_native_hc_mix_create(ggml_backend_t backend,const char* path,size_t budget){
 native_hc_environment();auto p=std::make_unique<StrataVkNativeHcMix>();p->owner=strata_vk_q8_component_owner(backend,strata_native_hc_mix_spv,sizeof(strata_native_hc_mix_spv)/4,NativeComponentAddressing::descriptor,"strata_native_hc_mix_observation",7,9,path,budget);return p.release();
}
void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix* p) noexcept {delete p;}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix* p,const ggml_tensor* input,const ggml_tensor* down,const ggml_tensor* up,ggml_tensor* lp,ggml_tensor* low,ggml_tensor* le,ggml_tensor* upre,ggml_tensor* gates,ggml_tensor* ue,ggml_tensor* mixed){
 native_hc_environment();native_hc_shape(2560,4,1);native_hc_need(p&&std::fegetround()==FE_TONEAREST,"native HC mix: component/RNE");auto& g=*p->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
 const ggml_tensor* tensors[]={input,down,up,lp,low,le,upre,gates,ue,mixed};const uint64_t bytes[]={40960,6553600,6553600,1280,1280,1280,40960,40960,40960,10240};uint32_t offsets[10];
 for(size_t i=0;i<10;++i){auto* t=tensors[i];native_hc_need(t&&t->type==(i==1||i==2?GGML_TYPE_I32:GGML_TYPE_F32)&&ggml_nbytes(t)==bytes[i],"native HC mix: type/span");native_hc_need(t->buffer&&ggml_is_contiguous(t)&&ggml_backend_buffer_is_vk(t->buffer),"native HC mix: Vulkan contiguous span");auto* b=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native HC mix: foreign device");auto bias=get_misalign_bytes(ctx,t);native_hc_need(bias%4==0,"native HC mix: alignment");offsets[i]=bias/4;
  for(size_t j=0;j<i;++j)if(t->buffer==tensors[j]->buffer){auto x=reinterpret_cast<uintptr_t>(t->data),y=reinterpret_cast<uintptr_t>(tensors[j]->data);native_hc_need(!(x<=y?y-x<ggml_nbytes(t):x-y<ggml_nbytes(tensors[j])),"native HC mix: overlapping spans");}}
 const auto& limits=ctx->device->properties.limits;native_hc_need(limits.maxComputeWorkGroupCount[0]>=10240&&limits.maxComputeSharedMemorySize>=1024,"native HC mix: grid/shared limit");
 ggml_backend_vk_synchronize(g.backend);
 // Bounded standalone domain preflight, not a numerical fallback.
 for(size_t i=0;i<3;++i){std::vector<uint32_t> data(bytes[i]/4);ggml_backend_tensor_get(tensors[i],data.data(),0,bytes[i]);if(i==0){for(auto b:data)native_hc_normal(native_hc_value(b));}else for(auto word:data){native_hc_normal(native_hc_value(word<<16));native_hc_normal(native_hc_value(word&0xffff0000u));}}
 const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);
 const size_t map[3][6]={{0,0,1,3,4,5},{4,4,2,6,7,8},{0,7,1,3,4,9}};
 for(uint32_t mode=0;mode<3;++mode){const auto* m=map[mode];std::array<uint32_t,9> params{mode,1,offsets[m[0]],offsets[m[1]],offsets[m[2]],offsets[m[3]],offsets[m[4]],offsets[m[5]],uint32_t(get_misalign_bytes(ctx,g.status)/4)};ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
  ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,tensors[m[0]],true),ggml_vk_tensor_subbuffer(ctx,tensors[m[1]],true),ggml_vk_tensor_subbuffer(ctx,tensors[m[2]],true),ggml_vk_tensor_subbuffer(ctx,tensors[m[3]],true),ggml_vk_tensor_subbuffer(ctx,tensors[m[4]],true),ggml_vk_tensor_subbuffer(ctx,tensors[m[5]],true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},params,{mode==0?320u:mode==1?10240u:10u,1,1});ggml_vk_sync_buffers(ctx,compute);compute.reset();}
 ggml_backend_vk_synchronize(g.backend);uint32_t status=0;ggml_backend_tensor_get(g.status,&status,0,4);native_hc_need(status==0,"native HC mix: nonfinite/subnormal or activation outside [-16,16]; no result accepted");
}
}
