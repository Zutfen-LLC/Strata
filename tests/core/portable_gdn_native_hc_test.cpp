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
int main(){try{
 Fixture f;std::string error;auto root=std::filesystem::temp_directory_path()/("hc-serial-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));check(std::filesystem::create_directory(root),"fresh root");
 for(const auto&v:f.views)if(v.name=="blk.0.hc_attn_inject.weight")expected_weights=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);
 for(const auto&v:f.views) {
  if(v.name=="blk.0.hc_attn_down.weight")expected_down=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
  if(v.name=="blk.0.hc_attn_up.weight")expected_up=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
 }
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
 original_compute=cpu->iface.graph_compute;cpu->iface.graph_compute=check_compute;
 PortableGdnOptions o;o.execution.threads=1;o.execution.accelerator=cpu.get();o.execution.request_accelerator=true;o.execution.allow_cpu_fallback=false;o.resident_attention_weights=true;
 auto baseline=PortableGdnLayer::create(0,f.views,o,error);check(bool(baseline),error);check(baseline->reset(error),error);PortableRoute route;PortableGdnTrace ordinary;
 check(baseline->step(f.input(0),experts(),route,error,&ordinary),error);const auto disabled=baseline->last_residual();check(!calls&&!ordinary.native_hc_norm_inject,"disabled mode invoked native HC");
 o.native_hc_norm_inject=true;o.native_hc_receipt_prefix=(root/"hc").string();
 // Explicit refusals occur before device arithmetic and do not manufacture a mode.
 for(int invalid=0;invalid<5;++invalid){auto bad=o;if(invalid==0)bad.execution.allow_cpu_fallback=true;if(invalid==1)bad.native_hc_receipt_prefix.clear();if(invalid==2)bad.batch_input_projections=true;if(invalid==3)bad.policy=PortableGdnPolicy::device_production;if(invalid==4)bad.resident_attention_weights=false;check(!PortableGdnLayer::create(0,f.views,bad,error),"invalid native mode accepted");check(!owners,"invalid mode leaked owner");}
 {auto bad=o;bad.execution.weight_budget=baseline->weight_bytes()+38399;check(!PortableGdnLayer::create(0,f.views,bad,error),"converted weight cap ignored");check(!owners,"weight refusal leaked owner");}
 {auto bad=o;bad.execution.graph_budget=1;check(!PortableGdnLayer::create(0,f.views,bad,error),"graph cap ignored");check(!owners,"graph refusal leaked owner");}
 fail_create=true;check(!PortableGdnLayer::create(0,f.views,o,error),"create failure accepted");check(!owners,"failed create leaked owner");fail_create=false;
 o.scratch=PortableGdnScratch::create(o.execution,error);check(bool(o.scratch),error);
 auto layer=PortableGdnLayer::create(0,f.views,o,error);check(bool(layer),error);check(owners==1,"owner count");check(layer->weight_bytes()==baseline->weight_bytes()+38400,"injection replacement accounting (no duplicate Q8 bank)");
 check(layer->reset(error),error);const auto initial=layer->recurrent_state();
 fail_run=true;check(!layer->step(f.input(0),experts(),route,error),"component failure accepted");check(layer->tokens()==0&&layer->recurrent_state()==initial&&!layer->native_hc_calls(),"failed component committed GDN state");fail_run=false;
 check(!layer->prepare_chunk(1,error),"chunk planning silently accepted");
 PortableGdnTrace a;check(layer->step(f.input(0),experts(),route,error,&a),error);
 check(uploaded==f.input(0),"native HC missed current residual upload");check(a.native_hc_norm_inject&&a.hc_residual==uploaded&&a.hc_normalized==expected_xn&&a.hc_rs==expected_rs&&a.hc_inject==expected_inject,"native buffers did not survive graph consumers");
 std::filesystem::path receipt;
 for(const auto& entry:std::filesystem::directory_iterator(root))
  if(std::filesystem::exists(entry.path()/"first-input.f32le")){check(receipt.empty(),"duplicate initial receipt");receipt=entry.path();}
 check(!receipt.empty(),"initial graph receipt missing");
 auto receipt_bytes=[&](const char* name){std::ifstream f(receipt/name,std::ios::binary);return std::vector<char>(std::istreambuf_iterator<char>(f),{});};
 const auto recorded=receipt_bytes("first-normalized.f32le");
 check(recorded.size()==a.hc_normalized.size()*4&&!std::memcmp(recorded.data(),a.hc_normalized.data(),recorded.size()),"initial receipt differs from live post-graph value");
 check(receipt_bytes("first-rs.f32le").size()==16&&receipt_bytes("first-inject.f32le").size()==16,"initial receipt shape");
 check(layer->native_hc_calls()==1&&layer->tokens()==1,"successful native counter");const auto first=layer->last_residual();
 check(layer->reset(error),error);PortableGdnTrace repeat;check(layer->step(f.input(0),experts(),route,error,&repeat),error);check(first==layer->last_residual()&&a.hc_inject==repeat.hc_inject,"reset/replay changed result");
 check(layer->reset(error),error);inject_bias=1;check(layer->step(f.input(0),experts(),route,error),error);check(layer->last_residual()!=first,"graph ignored native injection leaf");inject_bias=0;
 check(layer->reset(error),error);norm_scale=2;PortableGdnTrace changed;check(layer->step(f.input(0),experts(),route,error,&changed),error);check(changed.qkv!=a.qkv,"graph ignored native normalized leaf");norm_scale=1;
 check(layer->reset(error),error);check(layer->step(f.input(1),experts(),route,error),error);check(uploaded==f.input(1),"subsequent residual stale");
 check(receipt_bytes("first-normalized.f32le")==recorded,"later/reset forward overwrote initial receipt");
 check(direct_consumers>=calls,"native normalized direct consumer missing");layer.reset();check(!owners,"owner not released");
 o.resident_state=true;layer=PortableGdnLayer::create(0,f.views,o,error);check(bool(layer),error);check(layer->reset(error),error);
 check(layer->step(f.input(0),experts(),route,error),error);const auto resident=layer->last_residual();const auto state=layer->recurrent_state();
 check(layer->reset(error),error);check(layer->step(f.input(0),experts(),route,error),error);check(layer->last_residual()==resident&&layer->recurrent_state()==state,"resident state reset/replay changed");layer.reset();check(!owners,"resident owner not released");o.resident_state=false;

 // Composed coordinator test: device arithmetic is deliberately mocked;
 // exact shader/source arithmetic is qualified independently by component tests.
 o.native_hc_mix=true;fail_mix_create=true;check(!PortableGdnLayer::create(0,f.views,o,error),"mix create refusal accepted");check(!owners&&!mix_owners,"mix create failure leaked owner");fail_mix_create=false;
 auto composed=PortableGdnLayer::create(0,f.views,o,error);check(bool(composed),error);
 check(composed->state_bytes()==baseline->state_bytes()+PortableGdnOptions::native_hc_read_host_bound,"composed host bound not charged");
 check(mix_owners==1&&composed->weight_bytes()==baseline->weight_bytes()+38400+6144000,"composed replacement accounting");
 check(composed->reset(error),error);fail_mix=true;const auto old_calls=calls;
 check(!composed->step(f.input(0),experts(),route,error),"mix refusal accepted");
 check(composed->tokens()==0&&composed->native_hc_calls()==1&&composed->native_hc_mix_calls()==0&&calls==old_calls+1,"partial component work/state accounting");fail_mix=false;
 check(composed->reset(error),error);PortableGdnTrace composed_trace;check(composed->step(f.input(0),experts(),route,error,&composed_trace),error);
 check(composed_trace.native_hc_mix&&composed_trace.hc_mixed.size()==2560&&composed_trace.hc_low.size()==320,"composed trace/liveness");
 bool initial_mix=false;for(const auto&e:std::filesystem::directory_iterator(root))if(std::filesystem::exists(e.path()/"first-mixed.f32le"))initial_mix=true;
 check(initial_mix,"first successful composed graph receipt missing after prior refusal");
 const auto composed_first=composed->last_residual();check(composed->reset(error),error);check(composed->step(f.input(0),experts(),route,error),error);check(composed_first==composed->last_residual(),"composed reset differs");
 check(composed->reset(error),error);mixed_scale=2;check(composed->step(f.input(0),experts(),route,error),error);check(composed_first!=composed->last_residual(),"graph ignored mixed leaf");mixed_scale=1;
 check(composed->reset(error),error);const auto before_provider=composed->recurrent_state();PortableExpertRows no=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional provider refusal";return false;};
 check(!composed->step(f.input(0),no,route,error)&&composed->tokens()==0&&composed->recurrent_state()==before_provider,"composed provider refusal committed state");
 check(mixed_consumers>=mix_calls,"missing composed graph consumer");composed.reset();check(!mix_owners&&!owners,"composed owner leak");
 {auto bad=o;bad.native_hc_norm_inject=false;check(!PortableGdnLayer::create(0,f.views,bad,error),"mix without norm accepted");}
 {auto bad=o;bad.execution.weight_budget=baseline->weight_bytes()+38400+6143999;check(!PortableGdnLayer::create(0,f.views,bad,error),"composed weight cap ignored");check(!mix_owners&&!owners,"composed refusal leak");}
 // Explicit diagnostic pending tuple: no model flag or FFN role substitution.
 for(bool resident_pending:{false,true}) {
 o.resident_state=resident_pending;
 o.scratch=resident_pending?PortableGdnScratch::create(o.execution,error):nullptr;if(resident_pending)check(bool(o.scratch),error);
 o.native_hc_pending=true;fail_pending_create=true;
 check(!PortableGdnLayer::create(0,f.views,o,error),"pending create refusal accepted");check(!owners&&!mix_owners&&!pending_owners,"pending create refusal leaked components");fail_pending_create=false;
 {auto bad=o;bad.native_hc_mix=false;check(!PortableGdnLayer::create(0,f.views,bad,error),"pending without composed read accepted");}
#ifndef _WIN32
 {const char* prior=std::getenv("STRATA_LAYER0_OBSERVER");const bool had=prior!=nullptr;const std::string saved=prior?prior:"";unsetenv("STRATA_LAYER0_OBSERVER");Layer0Observer observer;if(had)setenv("STRATA_LAYER0_OBSERVER",saved.c_str(),1);auto bad=o;bad.execution.layer0_observer=&observer;check(!PortableGdnLayer::create(0,f.views,bad,error),"legacy observer silently omitted pending tuple");}
#endif
 {auto bad=o;bad.state_budget=baseline->state_bytes()+PortableGdnOptions::native_hc_pending_host_bound-1;check(!PortableGdnLayer::create(0,f.views,bad,error),"pending host admission short room accepted");}
 auto pending=PortableGdnLayer::create(0,f.views,o,error);check(bool(pending),error);check(pending_owners==1,"pending owner count");
 check(pending->weight_bytes()==baseline->weight_bytes()+6182400,"pending duplicated weight banks");
 check(pending->state_bytes()==baseline->state_bytes()+PortableGdnOptions::native_hc_pending_host_bound,"pending host bound not charged");
 const auto original=f.input(0);std::vector<float> bo(2560),inj(4,0.f);for(size_t d=0;d<bo.size();++d)bo[d]=float(int(d%11)-5)*0.001953125f;
 const auto saved_bo=bo,saved_inj=inj;const auto expected_pending=native_hc_pending_control(original,bo,inj).residual;
 check(pending->reset(error),error);const auto pending_state=pending->recurrent_state();
 check(!pending->step(original,experts(),route,error),"implicit missing tuple accepted");
 {auto bad=bo;bad.pop_back();check(!pending->step_pending(original,bad,inj,experts(),route,error),"bad pending shape accepted");}
 fail_pending_run=true;check(!pending->step_pending(original,bo,inj,experts(),route,error),"pending component refusal accepted");fail_pending_run=false;
 check(pending->native_hc_pending_calls()==0&&pending->tokens()==0&&pending->recurrent_state()==pending_state,"pending failure committed work/state");
 fail_run=true;check(!pending->step_pending(original,bo,inj,experts(),route,error),"norm refusal after pending accepted");fail_run=false;
 check(pending->native_hc_pending_calls()==1&&pending->native_hc_calls()==0&&pending->native_hc_mix_calls()==0&&pending->tokens()==0,"partial pending work accounting");
 fail_mix=true;check(!pending->step_pending(original,bo,inj,experts(),route,error),"mix refusal after pending accepted");fail_mix=false;
 check(pending->native_hc_pending_calls()==2&&pending->native_hc_calls()==1&&pending->native_hc_mix_calls()==0&&pending->recurrent_state()==pending_state,"partial pending norm/state accounting");
 PortableGdnTrace pt;check(pending->step_pending(original,bo,inj,experts(),route,error,&pt),error);
 check(pt.native_hc_pending&&pt.hc_residual==expected_pending&&pt.hc_pending_input==original&&pt.hc_previous_output==bo&&pt.hc_previous_inject==inj,"pending tuple/candidate trace binding");
 check(original==f.input(0)&&bo==saved_bo&&inj==saved_inj,"caller pending tuple changed");
 const auto pending_first=pending->last_residual(),first_state=pending->recurrent_state(),first_history=pending->convolution_history();
 check(pending->reset(error),error);const auto before_refusal=pending->recurrent_state(),before_refusal_history=pending->convolution_history();const auto before_refusal_tokens=pending->token_history();
 check(!pending->step_pending(original,bo,inj,no,route,error),"pending provider refusal accepted");
 check(pending->tokens()==0&&pending->last_residual().empty()&&pending->recurrent_state()==before_refusal&&pending->convolution_history()==before_refusal_history&&pending->token_history()==before_refusal_tokens,"pending provider refusal committed public state/history/token metadata");
 check(pending->native_hc_pending_calls()==4&&pending->native_hc_calls()==3&&pending->native_hc_mix_calls()==2,"pending paid work lost after provider refusal");
 check(pending->step_pending(original,bo,inj,experts(),route,error),error);
 check(pending->last_residual()==pending_first&&pending->recurrent_state()==first_state&&pending->convolution_history()==first_history,"pending retry/reset state differs");
 check(pending->native_hc_pending_calls()==5,"pending cumulative counter changed");
 bool tuple_receipt=false;for(const auto&e:std::filesystem::directory_iterator(root))if(std::filesystem::exists(e.path()/"first-previous-output.f32le"))tuple_receipt=true;check(tuple_receipt,"pending first tuple receipt missing");
 check(pending->reset(error),error);auto next_bo=bo;for(float&v:next_bo)v=-v;std::vector<float>next_inj{-8.f,-1.f,0.f,4.f};PortableGdnTrace next_pending;check(pending->step_pending(f.input(1),next_bo,next_inj,experts(),route,error,&next_pending),error);check(next_pending.hc_pending_input==f.input(1)&&next_pending.hc_previous_output==next_bo&&next_pending.hc_previous_inject==next_inj,"stale pending tuple upload");
 pending.reset();check(!pending_owners&&!owners&&!mix_owners,"pending owner release failed");
 // Sequential same-backend control avoids duplicate resident banks/scratch.
 o.native_hc_pending=false;auto materialized=PortableGdnLayer::create(0,f.views,o,error);check(bool(materialized),error);check(materialized->reset(error),error);
 check(materialized->step(expected_pending,experts(),route,error),error);
 check(materialized->last_residual()==pending_first&&materialized->recurrent_state()==first_state&&materialized->convolution_history()==first_history,"graph hidden consumed old R rather than R-prime");
 check(!materialized->native_hc_pending_calls(),"disabled pending path invoked component");
 check(!materialized->step_pending(original,bo,inj,experts(),route,error),"pending tuple accepted without opt-in");materialized.reset();
 }
 o.resident_state=false;
 o.native_hc_mix=false;
 o.native_hc_norm_inject=false;o.scratch.reset();auto off=PortableGdnLayer::create(0,f.views,o,error);check(bool(off),error);check(off->reset(error),error);check(off->step(f.input(0),experts(),route,error),error);check(off->last_residual()==disabled,"default-disabled equivalence lost");
 // Real GDN projected-output -> half-1 FFN HC boundary; no synthetic BO substitution.
 o.native_hc_norm_inject=true;o.native_hc_mix=true;o.native_hc_pending=false;
 PortableModelOptions model;model.native_hc_norm_inject_layer0=true;model.native_hc_mix_layer0=true;
 model.native_hc_ffn_read_layer0=true;model.layer0_resident_weights=true;model.native_hc_receipt_prefix=o.native_hc_receipt_prefix;
 portable_detail::model_gdn_hc_options(model,0,o);
 o.execution.resident_dense_weights=portable_detail::model_resident_dense(model,0,false);
 check(!o.native_hc_ffn_write&&o.native_hc_ffn_read&&o.native_hc_mix&&o.native_hc_norm_inject&&o.execution.resident_dense_weights,"actual model-to-layer FFN wiring");
 check(!portable_detail::model_resident_dense(model,1,false)&&!portable_detail::model_resident_dense(model,0,true),"layer0 residency broadened to another layer/head");
 PortableGdnOptions other;portable_detail::model_gdn_hc_options(model,1,other);
 check(!other.native_hc_ffn_read&&!other.native_hc_mix&&!other.native_hc_norm_inject,"native model options broadened beyond layer0");
 check(portable_detail::model_hc_host_bound(model,0)==(16u<<20)&&portable_detail::model_hc_host_bound(model,1)==0,"model FFN/attention host pre-admission omitted or double counted");
 std::vector<PortableTensorView> fw;for(const auto&v:f.views)if(v.name.find("hc_ffn_")!=std::string::npos||v.name.find(".ffn_")!=std::string::npos)fw.push_back(v);
 for(bool resident:{false,true}) {
  o.resident_state=resident;o.scratch=resident?PortableGdnScratch::create(o.execution,error):nullptr;
  for(int bad=0;bad<4;++bad){auto opts=o;if(bad==0)opts.native_hc_mix=false;if(bad==1)opts.native_hc_pending=true;if(bad==2)opts.execution.resident_dense_weights=false;if(bad==3)opts.execution.graph_budget=1;
   check(!PortableGdnLayer::create(0,f.views,opts,error),"invalid actual FFN boundary accepted");}
  {auto short_room=o;short_room.execution.weight_budget=baseline->weight_bytes()+2*6182400-1;
   check(!PortableGdnLayer::create(0,f.views,short_room,error),"model FFN converted weight budget refusal missing");}
  auto product=PortableGdnLayer::create(0,f.views,o,error);check(bool(product),error);
  check(product->weight_bytes()==baseline->weight_bytes()+2*6182400,"FFN roles duplicated or replacement ledger wrong");
  check(product->memory_usage().device_graph>0&&product->memory_usage().host_extra_bound>=(8u<<20),"FFN scratch/preflight ledger omitted");
  check(!product->prepare_chunk(1,error),"actual FFN pending accepted chunk");check(product->reset(error),error);
  inspect_producer=true;actual_projected.clear();actual_attention_inject.clear();PortableGdnTrace pt;
  const auto original=f.input(0);check(product->step(original,experts(),route,error,&pt),error);
  check(actual_projected.size()==2560&&actual_attention_inject.size()==4,"actual producer was not executed/captured");
  check(pt.ffn_hc.original==original&&pt.ffn_hc.previous_output==actual_projected&&pt.ffn_hc.previous_injection==actual_attention_inject,"FFN consumed a substituted/stale tuple");
  // Independent scalar native source order: quarter, negate, exp, add, divide,
  // then FMA into original R BEFORE the half-1 norm. Not the shared helper.
  std::vector<float> expected(10240);for(size_t c=0;c<4;++c){const float a=actual_attention_inject[c]*0.25f;const float e=std::exp(-a);const float den=1.0f+e;const float gate=2.0f/den;
   for(size_t h=0;h<2560;++h)expected[c*2560+h]=std::fma(actual_projected[h],gate,original[c*2560+h]);}
  check(pt.ffn_hc.residual==expected,"FFN did not use native FMA-before-normalization order");
  check(pt.ffn_hc.normalized==expected_xn,"FFN norm did not consume candidate");
  const auto completed=product->last_residual(),state=product->recurrent_state(),history=product->convolution_history();const auto activation=route.activation;
  check(product->native_ffn_hc_pending_calls()==1&&product->native_hc_pending_calls()==0,"half-specific paid counters mixed");
  check(product->reset(error),error);check(product->native_ffn_hc_pending_calls()==1,"reset erased paid FFN component work");const auto clean_state=product->recurrent_state(),clean_history=product->convolution_history();
  PortableExpertRows refuse=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="provider rollback";return false;};
  check(!product->step(original,refuse,route,error,&pt),"FFN provider refusal accepted");
  check(product->tokens()==0&&product->last_residual().empty()&&product->recurrent_state()==clean_state&&product->convolution_history()==clean_history&&pt.ffn_hc.residual.empty(),"FFN refusal published candidate/state/trace");
  check(product->native_ffn_hc_pending_calls()==2,"provider rollback hid paid FFN component work");
  fail_pending_run=true;check(!product->step(original,experts(),route,error),"half-1 component refusal accepted");fail_pending_run=false;
  check(product->native_ffn_hc_pending_calls()==2,"failed FFN component counted as completed work");
  check(product->tokens()==0&&product->recurrent_state()==clean_state,"half-1 refusal committed GDN state");
  check(product->step(original,experts(),route,error,&pt),error);check(product->last_residual()==completed&&product->recurrent_state()==state&&product->convolution_history()==history,"FFN reset/retry changed transaction");
  check(product->step(f.input(1),experts(),route,error,&pt),error);check(pt.ffn_hc.original==f.input(1)&&pt.ffn_hc.previous_output==actual_projected,"next token used stale producer");
  inspect_producer=false;product.reset();
  // Materialize the independently computed candidate, then run the same FFN
  // read with zero BO. This independently catches old-R post-consumer bugs.
  auto opts=o.execution;opts.native_hc_pending_read=true;opts.native_hc_receipt_prefix=(root/"ffn-control").string();
  auto control=PortableFfn::create({2560,4,320,640,512,10,1e-6f},0,fw,opts,error);check(bool(control),error);
  check(control->reset(expected,error),error);check(!control->step(experts(),route,error),"pending FFN accepted implicit tuple");
  check(!control->create_chunk_execution(2,error),"pending FFN accepted chunk owner");
  {auto bo=std::vector<float>(2560,0);auto inj=std::vector<float>(4,0);bo.pop_back();check(!control->step_pending(bo,inj,experts(),route,error),"short FFN producer accepted");bo.resize(2560);bo[0]=std::numeric_limits<float>::quiet_NaN();check(!control->step_pending(bo,inj,experts(),route,error),"nonfinite FFN producer accepted");check(control->residual()==expected,"invalid producer changed public residual");}
  {std::vector<PortableRoute> routes;check(!control->step_batch([](int64_t,const std::vector<PortableRoute>&,std::vector<float>&,std::string&){return true;},routes,error),"pending FFN accepted batch bypass");}
  check(control->step_pending(std::vector<float>(2560,0),std::vector<float>(4,0),experts(),route,error),error);
  check(control->residual()==completed&&route.activation==activation,"FFN downstream used original R instead of candidate");control.reset();
 }
 check(ffn_mixed_consumers>=6,"actual FFN router consumer checks missing");
 // Actual final FFN producer -> retained FFN R/injection -> disjoint native write.
 o.native_hc_ffn_write=true;
 fail_write_create=true;check(!PortableGdnLayer::create(0,f.views,o,error),"write creation refusal ignored");fail_write_create=false;check(!write_owners,"write owner leaked on refusal");
 {auto final=PortableGdnLayer::create(0,f.views,o,error);check(bool(final),error);check(final->reset(error),error);PortableGdnTrace trace;
 check(final->step(f.input(0),experts(),route,error,&trace),error);const auto& w=trace.ffn_hc;
 check(w.write_block==write_seen_block&&w.write_block.size()==2560&&w.write_candidate==final->last_residual(),"actual combined block/materialized consumer mismatch");
 for(size_t c=0;c<4;++c)for(size_t d=0;d<2560;++d)check(w.write_candidate[c*2560+d]==std::fma(w.write_block[d],w.write_gate[c],w.residual[c*2560+d]),"independent final FMA/source stream order");
 const auto first=final->last_residual(),state=final->recurrent_state(),history=final->convolution_history();check(final->native_ffn_hc_write_calls()==1,"write work not credited");
 check(final->reset(error),error);check(final->native_ffn_hc_write_calls()==1,"reset erased paid write");
 fail_write_run=true;const auto clean_state=final->recurrent_state(),clean_history=final->convolution_history();const auto clean_tokens=final->token_history();check(!final->step(f.input(0),experts(),route,error),"write refusal accepted");fail_write_run=false;
 check(route.ids.empty()&&final->recurrent_state()==clean_state&&final->convolution_history()==clean_history&&final->token_history()==clean_tokens&&final->tokens()==0&&final->last_residual().empty(),"write refusal committed layer state/route");check(final->native_ffn_hc_write_calls()==1,"failed write credited");
 check(final->step(f.input(0),experts(),route,error),error);check(final->last_residual()==first&&final->recurrent_state()==state&&final->convolution_history()==history,"write reset/retry changed result/state");
 check(final->native_ffn_hc_write_calls()==2,"write successful work count");}
 check(!write_owners,"write owner leaked");
 {std::vector<PortableTensorView> fw;for(const auto&v:f.views)if(v.name.find(".hc_ffn_")!=std::string::npos||v.name.find(".ffn_")!=std::string::npos)fw.push_back(v);
  auto wo=o.execution;wo.native_hc_pending_read=true;wo.native_hc_final_write=true;wo.native_hc_receipt_prefix=o.native_hc_receipt_prefix+".write-budget";
  auto final=PortableFfn::create({2560,4,320,640,512,10,1e-6f},0,fw,wo,error);check(bool(final),error);const auto bytes=final->graph_bytes(),weights=final->weight_bytes();
  check(final->reset(f.input(0),error),error);check(final->step_pending(std::vector<float>(2560,0),std::vector<float>(4,0),experts(),route,error),error);
  check(final->work().post_graph_calls==0&&final->native_hc_write_calls()==1,"native write credited/ran GGML post graph");
  auto short_room=wo;short_room.graph_budget=bytes-1;check(!PortableFfn::create({2560,4,320,640,512,10,1e-6f},0,fw,short_room,error),"one-byte-short write graph admitted");
  wo.native_hc_final_write=false;auto read=PortableFfn::create({2560,4,320,640,512,10,1e-6f},0,fw,wo,error);check(bool(read),error);check(read->weight_bytes()==weights,"final write duplicated/replaced weight banks");}
 {auto write_model=model;write_model.native_hc_ffn_write_layer0=true;PortableGdnOptions w0,w1;model_gdn_hc_options(write_model,0,w0);model_gdn_hc_options(write_model,1,w1);
 check(w0.native_hc_ffn_write&&!w1.native_hc_ffn_write,"write escaped layer0");check(model_hc_host_bound(write_model,0)==model_hc_host_bound(model,0)+PortableFfnOptions::native_hc_write_host_bound&&model_hc_host_bound(write_model,1)==0,"write host preadmission not conservative/scoped");}
 o.native_hc_ffn_write=false;
 o.native_hc_ffn_read=false;o.native_hc_mix=false;o.native_hc_norm_inject=false;o.execution.resident_dense_weights=false;o.resident_state=false;o.scratch.reset();
 auto disabled_again=PortableGdnLayer::create(0,f.views,o,error);check(bool(disabled_again),error);check(disabled_again->reset(error),error);check(disabled_again->step(f.input(0),experts(),route,error),error);check(disabled_again->last_residual()==disabled,"new boundary changed default-disabled output");
 cpu->iface.graph_compute=original_compute;std::filesystem::remove_all(root);
 puts("serial layer0 HC seam upload/consumption/liveness/reset/accounting/refusal pass; mocked component, no GPU qualification");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
