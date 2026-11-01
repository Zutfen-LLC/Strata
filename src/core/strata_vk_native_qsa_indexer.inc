#include "strata_vk_native_qsa_indexer.hpp"
#include "native_qsa_indexer_layout.hpp"
#include "strata_native_qsa_indexer_t1_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeQsaIndexer{std::unique_ptr<StrataVkLogitsGuard>owner;uint64_t calls=0;};
StrataVkNativeQsaIndexer* strata_vk_native_qsa_indexer_create(ggml_backend_t b,const char*path,size_t budget){native_hc_need(b&&ggml_backend_is_vk(b),"native QSA indexer: Vulkan backend required");auto*ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto&l=ctx->device->properties.limits;native_hc_need(l.maxComputeWorkGroupInvocations>=256&&l.maxComputeWorkGroupSize[0]>=256&&l.maxComputeSharedMemorySize>=1568&&l.maxPerStageDescriptorStorageBuffers>=6&&l.maxDescriptorSetStorageBuffers>=6,"native QSA indexer: workgroup/shared/descriptors unsupported");auto p=std::make_unique<StrataVkNativeQsaIndexer>();p->owner=strata_vk_q8_component_owner(b,strata_native_qsa_indexer_t1_spv,sizeof(strata_native_qsa_indexer_t1_spv)/4,NativeComponentAddressing::descriptor,"strata_native_qsa_indexer",6,10,path,budget);return p.release();}
void strata_vk_native_qsa_indexer_destroy(StrataVkNativeQsaIndexer*p)noexcept{delete p;}
size_t strata_vk_native_qsa_indexer_bytes(const StrataVkNativeQsaIndexer*p){return p?p->owner->bytes:0;}
uint64_t strata_vk_native_qsa_indexer_calls(const StrataVkNativeQsaIndexer*p)noexcept{return p?p->calls:0;}
uint32_t strata_vk_native_qsa_indexer_run(StrataVkNativeQsaIndexer*p,ggml_tensor*const t[5],uint32_t cells,int32_t pos,int32_t base,uint32_t theta){
 native_hc_need(p,"native QSA indexer: owner required");native_qsa_indexer_geometry(cells,pos,base,theta);auto&g=*p->owner;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);std::array<uint32_t,10>off{};const size_t n[]={128,128,qsa_idx_words(cells),qsa_idx_words(cells),386};
 for(size_t i=0;i<5;++i){auto*x=t[i];native_hc_need(x&&x->type==(i==2||i==3?GGML_TYPE_I32:GGML_TYPE_F32)&&ggml_nbytes(x)==n[i]*4&&ggml_is_contiguous(x)&&x->buffer&&ggml_backend_buffer_is_vk(x->buffer),"native QSA indexer: exact contiguous typed span");auto*b=static_cast<ggml_backend_vk_buffer_context*>(x->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native QSA indexer: foreign device");auto bias=get_misalign_bytes(ctx,x);native_hc_need(bias%4==0,"native QSA indexer: alignment");off[i]=bias/4;
  for(size_t j=0;j<i;++j)if(x->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(x->data),b=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need((a>=b?a-b>=ggml_nbytes(t[j]):b-a>=ggml_nbytes(x)),"native QSA indexer: partial or candidate/input alias");}}
 ggml_backend_vk_synchronize(g.backend);std::vector<float>raw(128),weight(128);std::vector<uint32_t>before(n[2]);ggml_backend_tensor_get(t[0],raw.data(),0,512);ggml_backend_tensor_get(t[1],weight.data(),0,512);ggml_backend_tensor_get(t[2],before.data(),0,before.size()*4);
 for(float v:raw){native_hc_need(std::isfinite(v)&&std::abs(v)<65520.f,"native QSA indexer: finite F16-roundable raw required");}
 for(float v:weight){native_hc_normal(v);}
 if(pos!=0&&pos%4==3){for(size_t i=0;i<384;++i){float v;std::memcpy(&v,&before[i],4);native_hc_normal(v);}}
 if(pos!=0){for(size_t i=384;i<512;++i){float v;std::memcpy(&v,&before[i],4);native_hc_normal(v);}}
 off[5]=get_misalign_bytes(ctx,g.status)/4;off[6]=cells;off[7]=uint32_t(pos);off[8]=uint32_t(base);off[9]=theta;const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
 ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3],true),ggml_vk_tensor_subbuffer(ctx,t[4],true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{1,1,1});ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);uint32_t status;ggml_backend_tensor_get(g.status,&status,0,4);++p->calls;return status;
}
}
