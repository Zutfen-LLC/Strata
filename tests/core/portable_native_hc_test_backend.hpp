#pragma once
// Actual GDN graph/coordinator/owner with mocked device HC arithmetic. No GPU proof.
#include "portable_gdn_fixture.hpp"
#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include "native_hc_pending.hpp"
#include "strata_vk_native_shared.hpp"
#include "native_qsa_indexer_bf16.hpp"
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
bool observe_wo_pending=false;size_t wo_pending_links=0;std::vector<float>actual_native_wo;
bool observe_qkv_norm=false;size_t qkv_norm_links=0;std::array<std::vector<float>,3>actual_qkv;
bool observe_indexer_projection=false;int fail_indexer_projection=0;size_t indexer_projection_links=0;std::array<std::vector<float>,2>actual_indexer_projection;
size_t owners=0,calls=0,mix_owners=0,mix_calls=0;bool fail_mix_domain=false,fail_mix_domain_ffn=false,last_norm_ffn=false;bool fail_mix=false,fail_mix_create=false;float mixed_scale=1;const ggml_tensor* live_mixed=nullptr;size_t mixed_consumers=0;
std::vector<uint16_t> expected_down,expected_up;bool fail_run=false,fail_create=false;float inject_bias=0,norm_scale=1;
std::vector<uint16_t> expected_ffn_down,expected_ffn_up,expected_ffn_inject;
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
 if(observe_wo_pending&&!actual_native_wo.empty()&&get(b)==actual_native_wo)++wo_pending_links;
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
 size_t wi=0;for(auto*w:{down,up}){check(w->type==GGML_TYPE_I32&&ggml_nbytes(w)==6553600,"converted down/up span");std::vector<uint16_t> b(3276800);ggml_backend_tensor_get(w,b.data(),0,b.size()*2);const auto&want=last_norm_ffn&&!expected_ffn_down.empty()?(wi?expected_ffn_up:expected_ffn_down):(wi?expected_up:expected_down);++wi;check(b==want,"original BF16 role conversion");}
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
 std::vector<uint16_t>weight(40960);ggml_backend_tensor_get(wi,weight.data(),0,81920);check(weight==(last_norm_ffn&&!expected_ffn_inject.empty()?expected_ffn_inject:expected_weights),"BF16 replacement differs from original pack conversion");
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
 if(observe_qkv_norm&&cols==256){const size_t role=rows==24?0:1;check(f.x==actual_qkv[role],"native norm/RoPE ignored QKV candidate");++qkv_norm_links;}
 if(observe_indexer_projection&&cols==128&&rows==4){check(f.x==actual_indexer_projection[1],"native norm ignored indexer query projection");++indexer_projection_links;}
 ++p->calls;++qsa_norm_calls;if(fail_qsa_norm)return 1;
 // Independent host primitive provider, not Vulkan/CUDA numerical evidence.
 std::vector<float>d(rows*130);
 for(size_t r=0;r<rows;++r){std::array<float,256>a{};for(size_t c=0;c<cols;++c)a[c]=f.x[r*stride+c]*f.x[r*stride+c];for(int off:{16,8,4,2,1}){auto b=a;for(size_t c=0;c<256;++c)a[c]=b[c]+b[c^off];}std::array<float,32>b{};for(size_t c=0;c<8;++c)b[c]=a[c*32];for(int off:{16,8,4,2,1}){auto old=b;for(size_t c=0;c<32;++c)b[c]=old[c]+old[c^off];}d[r*130]=b[0]/float(cols)+1e-6f;d[r*130+1]=1.f/std::sqrt(d[r*130]);
  for(size_t c=0;c<32;++c){float freq=::powf(native_qsa_nr::value(theta),float(c)),angle=float(f.pos[r])*freq;d[r*130+2+c]=freq;d[r*130+34+c]=angle;d[r*130+66+c]=::cosf(angle);d[r*130+98+c]=::sinf(angle);}}
 auto out=native_qsa_nr::control(f,d);put(t[3],out.normalized);put(t[4],out.rotated);put(t[5],d);return 0;
}
}
#include "../kernels/native_qsa_indexer_fixture.hpp"
#include "strata_vk_native_qsa_indexer.hpp"
namespace {bool fail_qsa_indexer=false;std::vector<uint32_t>last_indexer_candidate;}
namespace strata::core::portable_detail {
struct StrataVkNativeQsaIndexer{uint64_t calls=0;};
StrataVkNativeQsaIndexer* strata_vk_native_qsa_indexer_create(ggml_backend_t,const char*path,size_t budget){check(budget>=512,"QSA indexer metadata budget");std::ofstream(path)<<"mock, not GPU proof\n";return new StrataVkNativeQsaIndexer;}
void strata_vk_native_qsa_indexer_destroy(StrataVkNativeQsaIndexer*p)noexcept{delete p;}
size_t strata_vk_native_qsa_indexer_bytes(const StrataVkNativeQsaIndexer*){return 512;}
uint64_t strata_vk_native_qsa_indexer_calls(const StrataVkNativeQsaIndexer*p)noexcept{return p?p->calls:0;}
uint32_t strata_vk_native_qsa_indexer_run(StrataVkNativeQsaIndexer*p,ggml_tensor*const t[5],uint32_t cells,int32_t pos,int32_t base,uint32_t theta){
 check(t[0]->flags&GGML_TENSOR_FLAG_OUTPUT,"actual raw indexer projection lifetime missing OUTPUT");
 native_qsa_idx::Fixture f;f.cells=cells;f.base=base;f.first=pos;f.steps=1;f.theta=theta;f.raw=get(t[0]);f.gamma=get(t[1]);f.before.resize(qsa_idx_words(cells));ggml_backend_tensor_get(t[2],f.before.data(),0,f.before.size()*4);native_qsa_idx::validate(f);++p->calls;if(fail_qsa_indexer)return 1;
 if(observe_indexer_projection){check(f.raw==actual_indexer_projection[0],"indexer append ignored key projection");++indexer_projection_links;}
 std::vector<float>d(386);native_qsa_nr::Fixture n;n.cols=128;n.rows=1;n.stride=128;n.gamma=f.gamma;n.x.resize(128);n.pos={pos==0?0:base+4*(pos/4)};
 for(size_t i=0;i<128;++i){d[i]=native_qsa_idx::rounded(f.raw[i]);float sum=pos==0?d[i]:native_qsa_idx::value(f.before[i]);for(int j=1;j<4;++j)sum=sum+(pos==0||j==3?d[i]:native_qsa_idx::value(f.before[j*128+i]));n.x[i]=std::fma(.25f,sum,0.f);if(pos==0||pos%4==3)d[128+i]=n.x[i];}
 if(pos==0||pos%4==3){std::vector<float>nd(130);auto z=native_qsa_nr::control(n,nd);nd[0]=z.arg[0];nd[1]=1.f/std::sqrt(nd[0]);for(size_t i=0;i<32;++i){float freq=::powf(native_qsa_idx::value(theta),float(i)),angle=float(n.pos[0])*freq;nd[2+i]=freq;nd[34+i]=angle;nd[66+i]=::cosf(angle);nd[98+i]=::sinf(angle);}std::copy(nd.begin(),nd.end(),d.begin()+256);}
 auto out=native_qsa_idx::control(f,0,f.before,d);last_indexer_candidate=out;ggml_backend_tensor_set(t[3],out.data(),0,out.size()*4);put(t[4],d);return 0;
}
}
#include "../kernels/native_qsa_select_fixture.hpp"
#include "strata_vk_native_qsa_select.hpp"
namespace {bool fail_qsa_score=false,fail_qsa_select=false;}
namespace strata::core::portable_detail {
struct StrataVkNativeQsaSelect{uint64_t scores=0,selections=0;};
StrataVkNativeQsaSelect* strata_vk_native_qsa_select_create(ggml_backend_t,const char*s,const char*t,size_t budget){check(budget>=1024,"QSA score/selection metadata budget");std::ofstream(s)<<"mock, not GPU proof\n";std::ofstream(t)<<"mock, not GPU proof\n";return new StrataVkNativeQsaSelect;}
void strata_vk_native_qsa_select_destroy(StrataVkNativeQsaSelect*p)noexcept{delete p;}
size_t strata_vk_native_qsa_select_bytes(const StrataVkNativeQsaSelect*){return 1024;}
uint64_t strata_vk_native_qsa_score_calls(const StrataVkNativeQsaSelect*p)noexcept{return p?p->scores:0;}
uint64_t strata_vk_native_qsa_select_calls(const StrataVkNativeQsaSelect*p)noexcept{return p?p->selections:0;}
uint32_t strata_vk_native_qsa_scores_run(StrataVkNativeQsaSelect*p,ggml_tensor*const t[5],uint32_t cells){
 check((t[2]->flags&GGML_TENSOR_FLAG_OUTPUT)||(t[2]->op==GGML_OP_NONE&&t[2]->buffer&&t[2]->data),"actual projected query lifetime missing");native_qsa_ss::Fixture f;f.cells=cells;ggml_backend_tensor_get(t[3],f.step.data(),0,16);f.pooled=get(t[0]);f.dead=get(t[1]);f.query=get(t[2]);f.seed.assign(qsa_select_blocks(cells),0);
 check(last_indexer_candidate.size()==qsa_idx_words(cells),"missing actual indexer candidate producer");check(std::memcmp(f.pooled.data(),last_indexer_candidate.data()+qsa_idx_pool,(last_indexer_candidate.size()-qsa_idx_pool-1)*4)==0,"score used old pooled state");check(std::memcmp(f.dead.data(),last_indexer_candidate.data()+384,512)==0,"score lost actual retained dead/spare");++p->scores;if(fail_qsa_score)return 1;auto out=native_qsa_ss::score_control(f);ggml_backend_tensor_set(t[4],out.data(),0,out.size()*4);return 0;
}
uint32_t strata_vk_native_qsa_select_run(StrataVkNativeQsaSelect*p,ggml_tensor*const t[3],uint32_t cells){
 native_qsa_ss::Fixture f;f.cells=cells;f.compute=0;ggml_backend_tensor_get(t[1],f.step.data(),0,16);f.pooled.assign(qsa_select_blocks(cells)*128,0);f.dead.assign(128,0);f.query.assign(512,0);f.seed.resize(qsa_select_blocks(cells));ggml_backend_tensor_get(t[0],f.seed.data(),0,f.seed.size()*4);++p->selections;if(fail_qsa_select)return 1;auto out=native_qsa_ss::select_control(f,f.seed);ggml_backend_tensor_set(t[2],out.data(),0,out.size()*4);return 0;
}
}

#ifdef STRATA_PORTABLE_NATIVE_HC
#include "../kernels/native_qsa_attention_fixture.hpp"
#include "strata_vk_native_qsa_attention.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
#include "../kernels/native_q8_wo_fixture.hpp"
namespace {bool fail_qsa_attention=false;std::vector<float>actual_native_gated;}
namespace {bool fail_qsa_wo=false,fail_qsa_qkv=false;const ggml_tensor*actual_packed_producer=nullptr;const ggml_tensor*actual_qkv_packed=nullptr;}
namespace strata::core::portable_detail {
struct StrataVkNativeStages{uint64_t submitted=0,completed=0;};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t,const char*path,size_t budget){check(budget>=512,"quantizer budget");std::ofstream(path)<<"host mock only\n";return new StrataVkNativeStages;}
void strata_vk_native_stages_destroy(StrataVkNativeStages*p) noexcept{delete p;}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 512;}
uint64_t strata_vk_native_stages_quant_submitted(const StrataVkNativeStages*p) noexcept{return p?p->submitted:0;}
uint64_t strata_vk_native_stages_quant_completed(const StrataVkNativeStages*p) noexcept{return p?p->completed:0;}
uint64_t strata_vk_native_stages_quant_submission_uncertain(const StrataVkNativeStages*) noexcept{return 0;}
void strata_vk_native_stages_run(StrataVkNativeStages*p,uint32_t mode,const ggml_tensor*gate,const ggml_tensor*,ggml_tensor*,ggml_tensor*packed,uint32_t n,const int32_t*,size_t){
 check(mode==0&&(n==6144||n==2560),"unexpected quantizer geometry");auto values=get(gate);
 if(n==6144)check(values==actual_native_gated,"quantizer ignored actual native attention output");
 else check(gate==live_mixed,"QKV quantizer ignored actual HC mixed output");
 ++p->submitted;auto bytes=native_q8_fixture::quantize(values);ggml_backend_tensor_set(packed,bytes.data(),0,bytes.size());if(n==6144)actual_packed_producer=packed;else actual_qkv_packed=packed;++p->completed;
}
struct StrataVkQ8Wo{uint64_t submitted=0,completed=0,uncertain=0;bool qkv=false;};
StrataVkQ8Wo* strata_vk_q8_wo_create(ggml_backend_t,const char*path,size_t budget){check(budget>=16,"WO budget");std::ofstream(path)<<"host mock only\n";return new StrataVkQ8Wo;}
StrataVkQ8Wo* strata_vk_qsa_qkv_create(ggml_backend_t,const char*path,size_t budget){check(budget>=16,"QKV budget");std::ofstream(path)<<"host mock only\n";auto*p=new StrataVkQ8Wo;p->qkv=true;return p;}
void strata_vk_q8_wo_destroy(StrataVkQ8Wo*p)noexcept{delete p;}
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*){return 16;}
uint64_t strata_vk_q8_wo_submitted(const StrataVkQ8Wo*p)noexcept{return p?p->submitted:0;}
uint64_t strata_vk_q8_wo_completed(const StrataVkQ8Wo*p)noexcept{return p?p->completed:0;}
uint64_t strata_vk_q8_wo_submission_uncertain(const StrataVkQ8Wo*p)noexcept{return p?p->uncertain:0;}
void strata_vk_q8_wo_run(StrataVkQ8Wo*p,const ggml_tensor*weights,const ggml_tensor*packed,ggml_tensor*out,size_t rows){
 if(p->qkv){check(packed==actual_qkv_packed&&(rows==12288||rows==512),"QKV ignored actual packed producer");if(fail_qsa_qkv){++p->uncertain;throw std::runtime_error("injected QKV synchronize failure");}size_t role=size_t(p->completed%3);check((role==0&&rows==12288)||(role>0&&rows==512),"QKV role order");++p->submitted;actual_qkv[role].resize(rows);for(size_t i=0;i<rows;++i)actual_qkv[role][i]=float(int((i+role*3)%17)-8)*0.03125f;ggml_backend_tensor_set(out,actual_qkv[role].data(),0,rows*4);++p->completed;return;}
 check(packed==actual_packed_producer&&rows==2560,"WO ignored actual packed producer");if(fail_qsa_wo){++p->uncertain;throw std::runtime_error("injected WO synchronize failure");}++p->submitted;
 native_q8_wo_fixture::Fixture f{rows,native_q8_fixture::Bytes(rows*q8_wo_row_bytes),native_q8_fixture::Bytes(q8_wo_input_bytes)};
 ggml_backend_tensor_get(weights,f.weights.data(),0,f.weights.size());ggml_backend_tensor_get(packed,f.input.data(),0,f.input.size());
 actual_native_wo.resize(rows);for(size_t i=0;i<rows;++i)actual_native_wo[i]=native_q8_wo_fixture::reference(f,i).value;
 ggml_backend_tensor_set(out,actual_native_wo.data(),0,rows*4);++p->completed;
}
struct StrataVkNativeQsaAttention{std::array<uint64_t,3>calls{};};
StrataVkNativeQsaAttention* strata_vk_native_qsa_attention_create(ggml_backend_t,const char*a,const char*b,const char*c,size_t budget){check(budget>=3072,"QSA attention metadata budget");for(auto p:{a,b,c})std::ofstream(p)<<"mock, not GPU proof\n";return new StrataVkNativeQsaAttention;}
void strata_vk_native_qsa_attention_destroy(StrataVkNativeQsaAttention*p)noexcept{delete p;}
size_t strata_vk_native_qsa_attention_bytes(const StrataVkNativeQsaAttention*){return 3072;}
uint64_t strata_vk_native_qsa_attention_calls(const StrataVkNativeQsaAttention*p,unsigned i)noexcept{return p&&i<3?p->calls[i]:0;}
uint32_t strata_vk_native_qsa_attention_run(StrataVkNativeQsaAttention*p,ggml_tensor*const t[13],uint32_t cells){native_qsa_attn::Fixture f;f.cells=cells;auto n=native_qsa_attn::counts(cells);for(size_t i=0;i<7;++i){f.in[i].resize(n[i]);check(ggml_nbytes(t[i])==n[i]*4,"actual attention typed transport span");ggml_backend_tensor_get(t[i],f.in[i].data(),0,n[i]*4);}++p->calls[0];if(fail_qsa_attention)return 0x10001;auto out=native_qsa_attn::control(f);++p->calls[1];++p->calls[2];for(size_t i=0;i<6;++i)ggml_backend_tensor_set(t[i+7],out.out[i].data(),0,out.out[i].size()*4);actual_native_gated.resize(6144);std::memcpy(actual_native_gated.data(),out.out[5].data(),24576);return 0;}
}
#endif
namespace strata::core::portable_detail {
struct StrataVkNativeShared {uint64_t submitted=0,completed=0,uncertain=0;size_t budget=0;};
StrataVkNativeShared* strata_vk_qsa_indexer_bf16_create(ggml_backend_t,const char*path,size_t budget){check(budget>=512,"indexer projection budget");std::ofstream(path)<<"host mock only\n";auto*p=new StrataVkNativeShared;p->budget=budget;return p;}
size_t strata_vk_native_shared_bytes(const StrataVkNativeShared*){return 512;}
void strata_vk_native_shared_destroy(StrataVkNativeShared*p)noexcept{delete p;}
uint64_t strata_vk_native_shared_submitted(const StrataVkNativeShared*p)noexcept{return p?p->submitted:0;}
uint64_t strata_vk_native_shared_completed(const StrataVkNativeShared*p)noexcept{return p?p->completed:0;}
uint64_t strata_vk_native_shared_uncertain(const StrataVkNativeShared*p)noexcept{return p?p->uncertain:0;}
void strata_vk_native_shared_run(StrataVkNativeShared*p,uint32_t mode,const ggml_tensor*w,const ggml_tensor*x,ggml_tensor*out,ggml_tensor*aux,uint32_t ni,uint32_t no,uint32_t tokens){
 check(mode==1&&ni==2560&&tokens==1&&(no==128||no==512)&&x==live_mixed,"indexer projection did not consume actual HC mixed tensor");
 std::array<ggml_backend_buffer_t,4>seen{};size_t nseen=0,used=512;for(auto*t:std::array<const ggml_tensor*,4>{{w,x,out,aux}}){bool duplicate=false;for(size_t i=0;i<nseen;++i)duplicate|=seen[i]==t->buffer;if(!duplicate){seen[nseen++]=t->buffer;used+=ggml_backend_buffer_get_size(t->buffer);}}check(used<=p->budget,"indexer projection aggregate budget exceeded");
 const size_t role=no==128?0:1;if(fail_indexer_projection==int(role+1)){++p->uncertain;throw std::runtime_error("injected indexer projection failure");}
 auto in=get(x);std::vector<uint16_t>weights(size_t(no)*2560);ggml_backend_tensor_get(w,weights.data(),0,weights.size()*2);
 actual_indexer_projection[role]=qsa_indexer_bf16_control(in.data(),weights.data(),no);ggml_backend_tensor_set(out,actual_indexer_projection[role].data(),0,no*4);++p->submitted;++p->completed;
}
}
