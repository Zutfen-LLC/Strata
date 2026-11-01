#pragma once
// Actual GDN graph/coordinator/owner with mocked device HC arithmetic. No GPU proof.
#include "portable_gdn_fixture.hpp"
#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include "native_hc_pending.hpp"
#include "portable_model_preflight.hpp"
#include "strata/core/layer0_observer.hpp"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <limits>
using namespace strata::core;
using namespace strata::core::portable_detail;
using namespace gdn_fixture;
namespace {
bool inspect_producer=false;std::vector<float> actual_projected,actual_attention_inject;size_t ffn_mixed_consumers=0;
size_t write_owners=0,write_calls=0;bool fail_write_create=false,fail_write_run=false;
const ggml_tensor* live_ffn_residual=nullptr;const ggml_tensor* live_ffn_inject=nullptr;std::vector<float> write_seen_block;
size_t pending_owners=0,pending_calls=0;bool fail_pending_create=false,fail_pending_run=false;
size_t owners=0,calls=0,mix_owners=0,mix_calls=0;bool fail_mix=false,fail_mix_create=false;float mixed_scale=1;const ggml_tensor* live_mixed=nullptr;size_t mixed_consumers=0;
std::vector<uint16_t> expected_down,expected_up;bool fail_run=false,fail_create=false;float inject_bias=0,norm_scale=1;
std::vector<float> uploaded,expected_xn,expected_rs,expected_inject;
std::vector<uint16_t> expected_weights;
const ggml_tensor* bound_weight=nullptr;const ggml_tensor* live_normalized=nullptr;size_t direct_consumers=0;
decltype(ggml_backend_i::graph_compute) original_compute=nullptr;
ggml_status check_compute(ggml_backend_t b,ggml_cgraph*g){for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);if(t->op==GGML_OP_MUL_MAT&&t->src[0]&&std::string(ggml_get_name(t->src[0]))=="blk.0.hc_attn_down.weight"&&live_normalized){auto*root=t->src[1];while(root->view_src)root=root->view_src;check(root==live_normalized&&ggml_is_contiguous(t->src[1])&&t->src[1]->ne[0]==10240&&t->src[1]->ne[1]==1,"native normalized backing/contiguous consumer geometry changed");++direct_consumers;}}for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);if(t->op==GGML_OP_MUL_MAT&&t->src[0]&&std::string(ggml_get_name(t->src[0]))=="blk.0.attn_qkv.weight"&&live_mixed){auto*root=t->src[1];while(root->view_src)root=root->view_src;check(root==live_mixed,"composed mixed backing ignored/reused");++mixed_consumers;}}if(inspect_producer)for(int n=0;n<ggml_graph_n_nodes(g);++n){auto*t=ggml_graph_node(g,n);if(t->op==GGML_OP_MUL_MAT&&t->src[0]&&std::string(ggml_get_name(t->src[0]))=="blk.0.ffn_gate_inp.weight"){
 auto* root=t->src[1];while(root->view_src)root=root->view_src;check(root==live_mixed,"FFN router ignored native mixed leaf");++ffn_mixed_consumers;}}
 const auto status=original_compute(b,g);
 if(inspect_producer)for(int n=0;n<ggml_graph_n_nodes(g);++n){auto*t=ggml_graph_node(g,n);if(t->op==GGML_OP_MUL_MAT&&t->src[0]&&std::string(ggml_get_name(t->src[0]))=="blk.0.ssm_out.weight"){
 check(t->flags&GGML_TENSOR_FLAG_OUTPUT,"actual producer was not retained as output");actual_projected.resize(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,actual_projected.data(),0,actual_projected.size()*4);}}
 return status;}
std::vector<float> get(const ggml_tensor* t){std::vector<float>x(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,x.data(),0,x.size()*4);return x;}
void put(ggml_tensor*t,const std::vector<float>&x){ggml_backend_tensor_set(t,x.data(),0,x.size()*4);}
}
namespace strata::core::portable_detail {
struct StrataVkNativeHcPending{bool final_write=false;};
StrataVkNativeHcPending* strata_vk_native_hc_pending_create(ggml_backend_t,const char* path,size_t budget){check(!fail_pending_create,"injected pending create refusal");check(budget>=512,"pending metadata budget");std::ofstream(path)<<"mock; not shader evidence\n";++pending_owners;return new StrataVkNativeHcPending;}
size_t strata_vk_native_hc_pending_bytes(const StrataVkNativeHcPending*){return 512;}
StrataVkNativeHcPending* strata_vk_native_hc_write_create(ggml_backend_t,const char*path,size_t budget){check(!fail_write_create&&budget>=512,"injected write create/budget refusal");std::ofstream(path)<<"mock; not shader evidence\n";++write_owners;return new StrataVkNativeHcPending{true};}
void strata_vk_native_hc_pending_destroy(StrataVkNativeHcPending*p)noexcept{if(p){if(p->final_write)--write_owners;else --pending_owners;}delete p;}
void strata_vk_native_hc_pending_run(StrataVkNativeHcPending*owner,const ggml_tensor*r,const ggml_tensor*b,const ggml_tensor*i,ggml_tensor*rp,ggml_tensor*g,ggml_tensor*e){
 if(owner->final_write){
 check(!fail_write_run,"injected final write refusal");check(r==live_ffn_residual&&i==live_ffn_inject,"final write did not consume retained actual FFN operands");check(r!=rp&&b!=rp&&i!=rp,"final write producer/candidate alias");
 auto x=get(r),bo=get(b),inj=get(i);write_seen_block=bo;std::vector<float> result(10240),gate(4),exp(4);
 for(size_t c=0;c<4;++c){const float scaled=std::fma(.25f,inj[c],0.f);exp[c]=std::exp(-scaled);gate[c]=std::fma(2.f,1.f/(1.f+exp[c]),0.f);for(size_t d=0;d<2560;++d)result[c*2560+d]=std::fma(bo[d],gate[c],x[c*2560+d]);}
 put(rp,result);put(g,gate);put(e,exp);++write_calls;return;}
 check(!fail_pending_run,"injected pending component refusal");check(r!=rp&&b!=rp&&i!=rp,"pending input/candidate alias");
 auto out=native_hc_pending_control(get(r),get(b),get(i));put(rp,out.residual);put(g,out.gate);put(e,out.exponential);++pending_calls;
}
struct StrataVkNativeHcMix{};
StrataVkNativeHcMix* strata_vk_native_hc_mix_create(ggml_backend_t,const char* path,size_t budget){check(!fail_mix_create,"injected mix create refusal");check(budget>=512,"mix metadata budget");std::ofstream(path)<<"mock; not shader evidence\n";++mix_owners;return new StrataVkNativeHcMix;}
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*){return 512;}
void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix*p)noexcept{if(p)--mix_owners;if(!mix_owners)live_mixed=nullptr;delete p;}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor*x,const ggml_tensor*down,const ggml_tensor*up,ggml_tensor*lp,ggml_tensor*low,ggml_tensor*le,ggml_tensor*upre,ggml_tensor*gates,ggml_tensor*ue,ggml_tensor*mixed){
 check(!fail_mix,"injected mix refusal");check(x==live_normalized,"composition normalized operand differs");
 size_t wi=0;for(auto*w:{down,up}){check(w->type==GGML_TYPE_I32&&ggml_nbytes(w)==6553600,"converted down/up span");std::vector<uint16_t> b(3276800);ggml_backend_tensor_get(w,b.data(),0,b.size()*2);check(b==(wi++?expected_up:expected_down),"original BF16 role conversion");}
 for(auto*t:{lp,low,le,upre,gates,ue})put(t,std::vector<float>(size_t(ggml_nelements(t)),0));
 auto normalized=get(x);std::vector<float>values(2560);for(size_t i=0;i<values.size();++i)values[i]=normalized[i]*mixed_scale;
 put(mixed,values);live_mixed=mixed;++mix_calls;
}
struct StrataVkNativeHc{};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* path,size_t budget){check(!fail_create,"injected create refusal");check(budget>=512,"component budget");std::ofstream(path)<<"mock; not shader evidence\n";++owners;return new StrataVkNativeHc;}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 512;}
void strata_vk_native_hc_destroy(StrataVkNativeHc*p)noexcept{if(p)--owners;if(!owners){bound_weight=nullptr;live_normalized=nullptr;}delete p;}
void strata_vk_native_hc_run(StrataVkNativeHc*,const ggml_tensor*x,const ggml_tensor*w,const ggml_tensor*wi,ggml_tensor*xn,ggml_tensor*rs,ggml_tensor*inject,uint32_t t){
 native_hc_environment();check(x->ne[0]==2560&&x->ne[1]==4&&!x->view_src&&xn->ne[0]==10240&&xn->ne[1]==1&&!xn->view_src,"original input/normalized geometry");live_normalized=xn;
 check(t==1&&wi->type==GGML_TYPE_I32&&ggml_nbytes(wi)==81920,"native weight/T binding");
 check(!fail_run,"injected HC failure");
 std::vector<uint16_t>weight(40960);ggml_backend_tensor_get(wi,weight.data(),0,81920);check(weight==expected_weights,"BF16 replacement differs from original pack conversion");
 if(std::string(ggml_get_name(wi))=="blk.0.hc_attn_inject.weight") {if(bound_weight)check(wi==bound_weight,"weight tensor identity changed");else bound_weight=wi;}
 uploaded=get(x);auto out=native_hc_oracle(uploaded,get(w),weight,1);for(float&v:out.inject)v+=inject_bias;for(float&v:out.xn)v*=norm_scale;
 expected_xn=out.xn;expected_rs=out.rs;expected_inject=out.inject;
 put(xn,out.xn);put(rs,out.rs);put(inject,out.inject);if(std::string(ggml_get_name(wi))=="blk.0.hc_ffn_inject.weight"){live_ffn_residual=x;live_ffn_inject=inject;}if(inspect_producer&&std::string(ggml_get_name(wi))=="blk.0.hc_attn_inject.weight")actual_attention_inject=out.inject;++calls;
}
}
