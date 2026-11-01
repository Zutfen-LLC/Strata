#include "strata_vk_native_qsa_select.hpp"
#include "native_qsa_select_layout.hpp"
#include "strata_native_qsa_scores_t1_spv.hpp"
#include "strata_native_qsa_select_t1_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeQsaSelect {std::unique_ptr<StrataVkLogitsGuard>score,select;uint64_t scores=0,selections=0;};
StrataVkNativeQsaSelect* strata_vk_native_qsa_select_create(ggml_backend_t b,const char*score,const char*select,size_t budget){
 native_hc_need(b&&ggml_backend_is_vk(b)&&budget>=2,"native QSA scores/select: Vulkan owner/budget required");native_qsa_select_environment();auto*ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto&l=ctx->device->properties.limits;
 native_hc_need(l.maxComputeWorkGroupInvocations>=1024&&l.maxComputeWorkGroupSize[0]>=1024&&l.maxComputeSharedMemorySize>=37004&&l.maxPerStageDescriptorStorageBuffers>=6&&l.maxDescriptorSetStorageBuffers>=6,"native QSA scores/select: native 1024-thread register variant plus 37004-byte logical-warp storage unsupported");
 auto p=std::make_unique<StrataVkNativeQsaSelect>();p->score=strata_vk_q8_component_owner(b,strata_native_qsa_scores_t1_spv,sizeof(strata_native_qsa_scores_t1_spv)/4,NativeComponentAddressing::descriptor,"strata_native_qsa_scores_t1",6,7,score,budget/2);p->select=strata_vk_q8_component_owner(b,strata_native_qsa_select_t1_spv,sizeof(strata_native_qsa_select_t1_spv)/4,NativeComponentAddressing::descriptor,"strata_native_qsa_select_t1",4,6,select,budget-budget/2);return p.release();
}
void strata_vk_native_qsa_select_destroy(StrataVkNativeQsaSelect*p)noexcept{delete p;}
size_t strata_vk_native_qsa_select_bytes(const StrataVkNativeQsaSelect*p){return p?p->score->bytes+p->select->bytes:0;}
uint64_t strata_vk_native_qsa_score_calls(const StrataVkNativeQsaSelect*p)noexcept{return p?p->scores:0;}
uint64_t strata_vk_native_qsa_select_calls(const StrataVkNativeQsaSelect*p)noexcept{return p?p->selections:0;}
static std::array<uint32_t,7> qsa_select_bind(StrataVkLogitsGuard&g,ggml_tensor*const*t,const std::vector<size_t>&n,const std::vector<ggml_type>&type){
 auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);std::array<uint32_t,7>off{};
 for(size_t i=0;i<n.size();++i){auto*x=t[i];native_hc_need(x&&x->type==type[i]&&ggml_is_contiguous(x)&&ggml_nbytes(x)==n[i]*4&&x->buffer&&ggml_backend_buffer_is_vk(x->buffer),"native QSA scores/select: exact typed span");auto*b=static_cast<ggml_backend_vk_buffer_context*>(x->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native QSA scores/select: foreign device");auto bias=get_misalign_bytes(ctx,x);native_hc_need(bias%4==0&&bias/4<=UINT32_MAX,"native QSA scores/select: offset alignment/overflow");off[i]=uint32_t(bias/4);
 for(size_t j=0;j<i;++j)if(x->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(x->data),b=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need(a>=b?a-b>=ggml_nbytes(t[j]):b-a>=ggml_nbytes(x),"native QSA scores/select: candidate/input alias");}}
 return off;
}
static std::array<int32_t,4> qsa_select_step(ggml_tensor*t,uint32_t cells){std::array<int32_t,4>s{};ggml_backend_tensor_get(t,s.data(),0,16);native_qsa_select_geometry(cells,s);return s;}
uint32_t strata_vk_native_qsa_scores_run(StrataVkNativeQsaSelect*p,ggml_tensor*const t[5],uint32_t cells){
 native_hc_need(p&&cells>=1&&cells<=qsa_select_cells,"native QSA scores: owner/capacity required");auto&g=*p->score;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);uint32_t blocks=qsa_select_blocks(cells);auto off=qsa_select_bind(g,t,{size_t(blocks)*128,128,512,4,blocks},{GGML_TYPE_F32,GGML_TYPE_F32,GGML_TYPE_F32,GGML_TYPE_I32,GGML_TYPE_F32});ggml_backend_vk_synchronize(g.backend);auto step=qsa_select_step(t[3],cells);
 for(size_t i=0;i<3;++i){size_t count=i==0?size_t(step[2])*128:i==1?128:512;std::vector<float>v(count);if(count)ggml_backend_tensor_get(t[i],v.data(),0,count*4);for(float x:v)native_hc_normal(x);}
 off[5]=get_misalign_bytes(ctx,g.status)/4;off[6]=blocks;uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto c=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,c);ggml_vk_dispatch_pipeline(ctx,c,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3],true),ggml_vk_tensor_subbuffer(ctx,t[4],true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{256,1,1});ggml_vk_sync_buffers(ctx,c);c.reset();ggml_backend_vk_synchronize(g.backend);uint32_t status;ggml_backend_tensor_get(g.status,&status,0,4);++p->scores;return status;
}
uint32_t strata_vk_native_qsa_select_run(StrataVkNativeQsaSelect*p,ggml_tensor*const t[3],uint32_t cells){
 native_hc_need(p&&cells>=1&&cells<=qsa_select_cells,"native QSA selection: owner/capacity required");auto&g=*p->select;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);uint32_t blocks=qsa_select_blocks(cells);auto offsets=qsa_select_bind(g,t,{blocks,4,qsa_select_cap},{GGML_TYPE_F32,GGML_TYPE_I32,GGML_TYPE_I32});std::array<uint32_t,6>off{};std::copy_n(offsets.begin(),3,off.begin());ggml_backend_vk_synchronize(g.backend);qsa_select_step(t[1],cells);off[3]=get_misalign_bytes(ctx,g.status)/4;off[4]=blocks;off[5]=qsa_select_cap;uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto c=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,c);ggml_vk_dispatch_pipeline(ctx,c,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{1,1,1});ggml_vk_sync_buffers(ctx,c);c.reset();ggml_backend_vk_synchronize(g.backend);uint32_t status;ggml_backend_tensor_get(g.status,&status,0,4);++p->selections;return status;
}
}
