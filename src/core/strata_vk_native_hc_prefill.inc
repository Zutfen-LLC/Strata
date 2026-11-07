#ifndef STRATA_HC_PREFILL_MOCK_SPV
#include "strata_native_hc_prefill_spv.hpp"
#endif
namespace strata::core::portable_detail {
// The selected owner's explicit host envelope reserves 2 MiB for scalar and
// pending component construction, including concurrent SPIR-V rewrite copies.
static_assert(sizeof(strata_native_hc_prefill_spv)<=128u*1024u,"HC prefill shader exceeds host envelope");
struct StrataVkNativeHcPrefill {std::unique_ptr<StrataVkLogitsGuard> owner;};
StrataVkNativeHcPrefill* strata_vk_native_hc_prefill_create(ggml_backend_t backend,const char* path,size_t budget){
#if !defined(STRATA_NATIVE_Q8_STRICT_FMA) && !defined(STRATA_HC_PREFILL_MOCK_SPV)
 throw std::runtime_error("HC prefill: strict FMA device capability required");
#endif
 native_hc_environment();auto p=std::make_unique<StrataVkNativeHcPrefill>();p->owner=strata_vk_q8_component_owner(backend,strata_native_hc_prefill_spv,sizeof(strata_native_hc_prefill_spv)/4,NativeComponentAddressing::descriptor,"strata_native_hc_prefill",7,9,path,budget);return p.release();
}
size_t strata_vk_native_hc_prefill_bytes(const StrataVkNativeHcPrefill* p){return p?p->owner->bytes:0;}
void strata_vk_native_hc_prefill_destroy(StrataVkNativeHcPrefill* p) noexcept{delete p;}
void strata_vk_native_hc_prefill_run(StrataVkNativeHcPrefill* p,uint32_t mode,uint32_t rows,const std::array<ggml_tensor*,6>& t){
 native_hc_environment();native_hc_need(p&&std::fegetround()==FE_TONEAREST&&(rows==1||rows==2)&&mode<3,"HC prefill: mode/rows/RNE");
 auto& g=*p->owner;auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
 // Unused bindings are supplied as null and rebound to an existing valid span.
 const uint64_t norm[]={40960ull*rows,40960,40960ull*rows,40960ull*rows,20480ull*rows,16ull*rows};
 const uint64_t silu[]={1280ull*rows,0,1280ull*rows,1280ull*rows,640ull*rows,0};
 const uint64_t mix[]={40960ull*rows,40960ull*rows,10240ull*rows,0,5120ull*rows,5120ull*rows};
 const auto* sizes=mode==0?norm:mode==1?silu:mix;std::array<uint32_t,9> params{mode,rows,0,0,0,0,0,0,0};
 for(size_t i=0;i<6;++i){if(!sizes[i]){native_hc_need(!t[i],"HC prefill: unused binding");continue;}auto* v=t[i];
  const auto type=(i==4||(i==5&&mode==2))?GGML_TYPE_I32:GGML_TYPE_F32;
  native_hc_need(v&&v->type==type&&ggml_nbytes(v)==sizes[i]&&v->buffer&&ggml_is_contiguous(v)&&ggml_backend_buffer_is_vk(v->buffer),"HC prefill: type/span/backend");
  auto* b=static_cast<ggml_backend_vk_buffer_context*>(v->buffer->context);native_hc_need(b->device.lock()==ctx->device,"HC prefill: foreign device");
  auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(v->buffer)),data=reinterpret_cast<uintptr_t>(v->data);size_t capacity=ggml_backend_buffer_get_size(v->buffer);native_hc_need(data>=base&&data-base<=capacity&&sizes[i]<=capacity-(data-base),"HC prefill: backing span");
  auto bias=get_misalign_bytes(ctx,v);native_hc_need(bias%4==0&&uint64_t(bias)/4+sizes[i]/4-1<=UINT32_MAX,"HC prefill: offset");params[2+i]=bias/4;
  for(size_t j=0;j<i;++j)if(sizes[j]&&v->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(v->data),b2=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need(a<=b2?b2-a>=sizes[i]:a-b2>=sizes[j],"HC prefill: overlap");}
 }
 const auto& limits=ctx->device->properties.limits;uint32_t gx=mode==0?4u:mode==1?1u:5u;
 native_hc_need(limits.maxComputeWorkGroupCount[0]>=gx&&limits.maxComputeWorkGroupCount[1]>=rows&&limits.maxComputeWorkGroupInvocations>=256&&limits.maxComputeWorkGroupSize[0]>=256&&limits.maxComputeSharedMemorySize>=1024,"HC prefill: grid/shared limits");
 // Domain-only preflight; the device computes every arithmetic stage.
 ggml_backend_vk_synchronize(g.backend);
 for(size_t i=0;i<2;++i)if(sizes[i]){std::vector<float> v(sizes[i]/4);ggml_backend_tensor_get(t[i],v.data(),0,sizes[i]);for(float x:v)native_hc_normal(x);}
 const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);params[8]=get_misalign_bytes(ctx,g.status)/4;
 auto* fallback=t[0];ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
 ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1]?t[1]:fallback,true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3]?t[3]:fallback,true),ggml_vk_tensor_subbuffer(ctx,t[4],true),ggml_vk_tensor_subbuffer(ctx,t[5]?t[5]:fallback,true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},params,{gx,rows,1});
 ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);uint32_t status=0;ggml_backend_tensor_get(g.status,&status,0,4);native_hc_need(status==0,"HC prefill: device domain refusal");
}
}
