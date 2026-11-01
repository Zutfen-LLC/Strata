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
size_t owners=0,calls=0,mix_owners=0,mix_calls=0;bool fail_mix_domain=false,fail_mix_domain_ffn=false,last_norm_ffn=false;bool fail_mix=false,fail_mix_create=false;float mixed_scale=1;const ggml_tensor* live_mixed=nullptr;size_t mixed_consumers=0;
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
 put(mixed,values);live_mixed=mixed;if(fail_mix_domain||(fail_mix_domain_ffn&&last_norm_ffn))throw NativeHcMixDomainRefusal(8u<<8);++mix_calls;
}
struct StrataVkNativeHc{};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* path,size_t budget){check(!fail_create,"injected create refusal");check(budget>=512,"component budget");std::ofstream(path)<<"mock; not shader evidence\n";++owners;return new StrataVkNativeHc;}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 512;}
void strata_vk_native_hc_destroy(StrataVkNativeHc*p)noexcept{if(p)--owners;if(!owners){bound_weight=nullptr;live_normalized=nullptr;}delete p;}
void strata_vk_native_hc_run(StrataVkNativeHc*,const ggml_tensor*x,const ggml_tensor*w,const ggml_tensor*wi,ggml_tensor*xn,ggml_tensor*rs,ggml_tensor*inject,uint32_t t){
 native_hc_environment();last_norm_ffn=std::string(ggml_get_name(wi)).ends_with(".hc_ffn_inject.weight");check(x->ne[0]==2560&&x->ne[1]==4&&!x->view_src&&xn->ne[0]==10240&&xn->ne[1]==1&&!xn->view_src,"original input/normalized geometry");live_normalized=xn;
 check(t==1&&wi->type==GGML_TYPE_I32&&ggml_nbytes(wi)==81920,"native weight/T binding");
 check(!fail_run,"injected HC failure");
 std::vector<uint16_t>weight(40960);ggml_backend_tensor_get(wi,weight.data(),0,81920);check(weight==expected_weights,"BF16 replacement differs from original pack conversion");
 if(std::string(ggml_get_name(wi))=="blk.0.hc_attn_inject.weight") {if(bound_weight)check(wi==bound_weight,"weight tensor identity changed");else bound_weight=wi;}
 uploaded=get(x);auto out=native_hc_oracle(uploaded,get(w),weight,1);for(float&v:out.inject)v+=inject_bias;for(float&v:out.xn)v*=norm_scale;
 expected_xn=out.xn;expected_rs=out.rs;expected_inject=out.inject;
 put(xn,out.xn);put(rs,out.rs);put(inject,out.inject);if(std::string(ggml_get_name(wi)).ends_with(".hc_ffn_inject.weight")){live_ffn_residual=x;live_ffn_inject=inject;}if(inspect_producer&&std::string(ggml_get_name(wi))=="blk.0.hc_attn_inject.weight")actual_attention_inject=out.inject;++calls;
}
}

// GDN production seam control. Device recurrence is mocked only in coordinator tests.
#include "strata_vk_native_gdn.hpp"
#include "../kernels/native_gdn_t1_fixture.hpp"
namespace {size_t gdn_calls=0;bool fail_gdn=false;}
namespace strata::core::portable_detail {
struct StrataVkNativeGdn{};
StrataVkNativeGdn* strata_vk_native_gdn_create(ggml_backend_t,const char*path,size_t budget){gdn_fixture::check(budget>=512,"GDN metadata budget");std::ofstream(path)<<"mock GDN; not shader evidence\n";return new StrataVkNativeGdn;}
void strata_vk_native_gdn_destroy(StrataVkNativeGdn*p)noexcept{delete p;}
size_t strata_vk_native_gdn_bytes(const StrataVkNativeGdn*){return 512;}
void strata_vk_native_gdn_run(StrataVkNativeGdn*,ggml_tensor*const t[13],ggml_tensor*){
 native_gdn_t1::Fixture f;for(size_t i=0;i<6;++i)f[i]=get(t[i]);std::vector<float>eg(48),rs(49,1.f),sig(6144),se(6144);for(size_t i=0;i<48;++i)eg[i]=std::exp(f[2][i]);for(size_t i=0;i<6144;++i){se[i]=std::exp(-f[4][i]);sig[i]=1.f/(1.f+se[i]);}rs[48]=1.f/std::sqrt(128.f);auto partial=native_gdn_t1::control(f,eg,rs,sig);for(size_t i=0;i<48;++i)rs[i]=1.f/std::sqrt(partial.arg[i]);auto ref=native_gdn_t1::control(f,eg,rs,sig);put(t[6],ref.state);put(t[7],ref.y);put(t[8],eg);put(t[9],ref.oc);put(t[10],rs);put(t[11],se);put(t[12],sig);gdn_fixture::check(!fail_gdn,"injected GDN candidate refusal");++gdn_calls;
}
}

#include "strata_vk_native_gdn_conv.hpp"
#include "strata_vk_native_gdn_ab.hpp"
#include "../kernels/native_gdn_conv_fixture.hpp"
#include "../kernels/native_gdn_ab_fixture.hpp"
namespace {bool fail_prep_conv=false,fail_prep_ab=false;}
namespace strata::core::portable_detail {
struct StrataVkNativeGdnConv{uint64_t calls=0;uint32_t status=0;};
struct StrataVkNativeGdnAb{uint64_t calls=0;};
StrataVkNativeGdnConv* strata_vk_native_gdn_conv_create(ggml_backend_t,const char*p,size_t b){gdn_fixture::check(b>=512,"conv budget");std::ofstream(p)<<"mock convolution; not shader evidence\n";return new StrataVkNativeGdnConv;}
void strata_vk_native_gdn_conv_destroy(StrataVkNativeGdnConv*p)noexcept{delete p;}
size_t strata_vk_native_gdn_conv_bytes(const StrataVkNativeGdnConv*){return 512;}
uint64_t strata_vk_native_gdn_conv_calls(const StrataVkNativeGdnConv*p)noexcept{return p->calls;}
uint32_t strata_vk_native_gdn_conv_status(const StrataVkNativeGdnConv*p)noexcept{return p->status;}
void strata_vk_native_gdn_conv_run(StrataVkNativeGdnConv*p,ggml_tensor*const t[10],bool commit){
 native_gdn_conv::Fixture f;for(size_t i=0;i<3;++i)f[i]=get(t[i]);std::vector<float>ex(10240),rs(32,1.f);
 for(size_t c=0;c<10240;++c){float v=std::fma(f[0][c*3],f[2][c*4],f[0][c*3+1]*f[2][c*4+1]);v=std::fma(f[0][c*3+2],f[2][c*4+2],v);v=std::fma(f[1][c],f[2][c*4+3],v);ex[c]=std::exp(-v);}
 auto a=native_gdn_conv::control(f,ex,rs,commit);for(size_t i=0;i<32;++i)rs[i]=1.f/std::sqrt(a.arg[i]);a=native_gdn_conv::control(f,ex,rs,commit);
 put(t[3],a.history);put(t[4],a.h);put(t[5],a.pre);put(t[6],ex);put(t[7],a.silu);put(t[8],rs);put(t[9],a.arg);++p->calls;p->status=fail_prep_conv?1:0;gdn_fixture::check(!p->status,"injected completed convolution refusal");
}
StrataVkNativeGdnAb* strata_vk_native_gdn_ab_create(ggml_backend_t,const char*p,size_t b){gdn_fixture::check(b>=512,"ab budget");std::ofstream(p)<<"mock a/b; not shader evidence\n";return new StrataVkNativeGdnAb;}
void strata_vk_native_gdn_ab_destroy(StrataVkNativeGdnAb*p)noexcept{delete p;}
size_t strata_vk_native_gdn_ab_bytes(const StrataVkNativeGdnAb*){return 512;}
uint64_t strata_vk_native_gdn_ab_calls(const StrataVkNativeGdnAb*p)noexcept{return p->calls;}
uint32_t strata_vk_native_gdn_ab_run(StrataVkNativeGdnAb*p,ggml_tensor*const t[8]){
 native_gdn_ab::Fixture f;f.x=get(t[0]);f.dt=get(t[3]);f.a=get(t[4]);f.wa.resize(2560*48);f.wb.resize(2560*48);
 for(size_t i=0;i<2;++i){gdn_fixture::check(t[i+1]->type==GGML_TYPE_BF16,"actual native replacement type");auto&w=i?f.wb:f.wa;ggml_backend_tensor_get(t[i+1],w.data(),0,w.size()*2);}
 auto dots=native_gdn_ab::dot(f);std::vector<float>diag(336),gate(48),beta(48);
 for(size_t h=0;h<48;++h){float v=dots[0][h]+f.dt[h],ex=v>20.f?0.f:std::exp(v),sp=v>20.f?v:std::log1p(ex),be=std::exp(-dots[1][h]),den=1.f+be;float a[]={dots[0][h],dots[1][h],v,ex,sp,be,den};for(size_t i=0;i<7;++i)diag[i*48+h]=a[i];gate[h]=sp*f.a[h];beta[h]=1.f/den;}
 put(t[5],gate);put(t[6],beta);put(t[7],diag);++p->calls;return fail_prep_ab?1:0;
}
}

#include "../kernels/native_qsa_norm_rope_fixture.hpp"
#include "strata_vk_native_qsa_norm_rope.hpp"
namespace { bool fail_qsa_norm=false;size_t qsa_norm_calls=0;std::vector<float>qsa_last_raw; }
namespace strata::core::portable_detail {
struct StrataVkNativeQsaNormRope{uint64_t calls=0;};
StrataVkNativeQsaNormRope* strata_vk_native_qsa_norm_rope_create(ggml_backend_t,const char*path,size_t budget){check(budget>=512,"QSA component metadata budget");std::ofstream(path)<<"mock, not shader evidence\n";return new StrataVkNativeQsaNormRope;}
void strata_vk_native_qsa_norm_rope_destroy(StrataVkNativeQsaNormRope*p)noexcept{delete p;}
size_t strata_vk_native_qsa_norm_rope_bytes(const StrataVkNativeQsaNormRope*){return 512;}
uint64_t strata_vk_native_qsa_norm_rope_calls(const StrataVkNativeQsaNormRope*p)noexcept{return p?p->calls:0;}
uint32_t strata_vk_native_qsa_norm_rope_run(StrataVkNativeQsaNormRope*p,ggml_tensor*const t[6],uint32_t cols,uint32_t rows,uint32_t stride,uint32_t theta){
 check(t[0]->flags&GGML_TENSOR_FLAG_OUTPUT,"actual QSA projection lifetime missing OUTPUT");
 native_qsa_nr::Fixture f;f.cols=cols;f.rows=rows;f.stride=stride;f.theta=theta;f.x=get(t[0]);f.gamma=get(t[1]);f.pos.resize(rows);ggml_backend_tensor_get(t[2],f.pos.data(),0,rows*4);native_qsa_nr::validate(f);qsa_last_raw=f.x;
 ++p->calls;++qsa_norm_calls;if(fail_qsa_norm)return 1;
 // Independent host primitive provider, not Vulkan/CUDA numerical evidence.
 std::vector<float>d(rows*130);
 for(size_t r=0;r<rows;++r){std::array<float,256>a{};for(size_t c=0;c<cols;++c)a[c]=f.x[r*stride+c]*f.x[r*stride+c];for(int off:{16,8,4,2,1}){auto b=a;for(size_t c=0;c<256;++c)a[c]=b[c]+b[c^off];}std::array<float,32>b{};for(size_t c=0;c<8;++c)b[c]=a[c*32];for(int off:{16,8,4,2,1}){auto old=b;for(size_t c=0;c<32;++c)b[c]=old[c]+old[c^off];}d[r*130]=b[0]/float(cols)+1e-6f;d[r*130+1]=1.f/std::sqrt(d[r*130]);
  for(size_t c=0;c<32;++c){float freq=::powf(native_qsa_nr::value(theta),float(c)),angle=float(f.pos[r])*freq;d[r*130+2+c]=freq;d[r*130+34+c]=angle;d[r*130+66+c]=::cosf(angle);d[r*130+98+c]=::sinf(angle);}}
 auto out=native_qsa_nr::control(f,d);put(t[3],out.normalized);put(t[4],out.rotated);put(t[5],d);return 0;
}
}
