#include "../kernels/native_hc_write_fixture.hpp"
// Physical seam probe: approved caller-supplied HC inputs plus existing synthetic
// full-GDN fixture. No model, new reference kernel or full-HC parity claim.
#include "portable_gdn_fixture.hpp"
#include "../kernels/native_hc_pending_fixture.hpp"
#include "ggml-vulkan.h"
#include <filesystem>
#include <fstream>
using namespace strata::core;
using namespace gdn_fixture;
namespace {
std::vector<uint8_t> load(const std::filesystem::path&p,size_t n){check(std::filesystem::file_size(p)==n,"bounded operand size");std::ifstream f(p,std::ios::binary);std::vector<uint8_t>x(n);check(bool(f.read(reinterpret_cast<char*>(x.data()),n)),"operand read");return x;}
void save(const std::filesystem::path&p,const std::vector<float>&x){check(!std::filesystem::exists(p),"fresh output");native_hc_fixture::floats(p.string(),x);}
void export_ffn(const std::filesystem::path& out,const PortableFfnHcTrace& t,const Fixture& f) {
 check(std::filesystem::create_directory(out),"fresh FFN observations");
 for(auto pair:{std::pair{"unapplied-residual",&t.original},std::pair{"previous-output",&t.previous_output},std::pair{"previous-injection",&t.previous_injection},
   std::pair{"residual",&t.residual},std::pair{"pending-gate",&t.gate},std::pair{"pending-exp",&t.exponential},std::pair{"normalized",&t.normalized},std::pair{"rs",&t.rs},std::pair{"inject",&t.inject},
   std::pair{"low-pre",&t.low_pre},std::pair{"low",&t.low},std::pair{"low-exp",&t.low_exp},std::pair{"up-pre",&t.up_pre},std::pair{"gates",&t.gates},std::pair{"up-exp",&t.up_exp},std::pair{"mixed",&t.mixed}})
  save(out/(std::string(pair.first)+".f32le"),*pair.second);
 native_hc_pending_fixture::Pending fixture;fixture.norm.t=1;fixture.norm.x=t.original;fixture.bo=t.previous_output;fixture.inj=t.previous_injection;
 for(const auto&v:f.views){
  if(v.name=="blk.0.hc_ffn_norm.weight"){fixture.norm.w.resize(10240);memcpy(fixture.norm.w.data(),v.data,40960);}
  if(v.name=="blk.0.hc_ffn_inject.weight")fixture.norm.wi=strata::core::portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);
  if(v.name=="blk.0.hc_ffn_down.weight")fixture.down=strata::core::portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
  if(v.name=="blk.0.hc_ffn_up.weight")fixture.up=strata::core::portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
 }
 native_hc_pending_fixture::write((out/"fixture.bin").string(),fixture);
}
void export_write(const std::filesystem::path& out,const PortableFfnHcTrace&t){
 check(std::filesystem::create_directory(out),"fresh final write observations");
 native_hc_write_fixture::Fixture f{t.residual,t.write_block,t.inject};native_hc_write_fixture::write((out/"fixture.bin").string(),f);
 save(out/"residual.f32le",t.write_candidate);save(out/"gate.f32le",t.write_gate);save(out/"exp.f32le",t.write_exp);
}
struct Result{std::vector<float> residual,state,history;size_t weights=0;std::vector<float> hc_input;PortableFfnHcTrace ffn;};
Result run(const Fixture&f,ggml_backend_t backend,bool enabled,bool composed,const std::vector<float>&input,const std::filesystem::path&out,bool pending=false,bool ffn=false,bool final_write=false){
 PortableGdnOptions o;o.execution.threads=1;o.execution.accelerator=backend;o.execution.request_accelerator=true;o.execution.allow_cpu_fallback=false;o.resident_attention_weights=true;o.resident_state=true;o.execution.resident_dense_weights=ffn;
 std::string error;o.scratch=PortableGdnScratch::create(o.execution,error);check(bool(o.scratch),error);
 // Warm the reusable scheduler with a normal graph before borrowing it for HC.
 auto views=f.views;for(auto&v:views)v.name.replace(0,6,"blk.4.");
 auto warm=PortableGdnLayer::create(4,views,o,error);check(bool(warm),error);
 o.native_hc_ffn_read=enabled&&ffn;
 o.native_hc_ffn_write=enabled&&final_write;
 o.native_hc_pending=enabled&&pending;
 o.native_hc_mix=enabled&&composed;
 o.native_hc_norm_inject=enabled;o.native_hc_receipt_prefix=(out/"hc").string();
 auto layer=PortableGdnLayer::create(0,f.views,o,error);check(bool(layer),error);check(layer->reset(error),error);
 std::vector<float>bo(2560),previous_injection{-8.f,-1.f,0.f,4.f};for(size_t d=0;d<bo.size();++d)bo[d]=float(int(d%11)-5)*0.001953125f;
 PortableRoute route;PortableGdnTrace a;
 auto step=[&](const PortableExpertRows&dispatch,PortableGdnTrace*trace=nullptr){return enabled&&pending?layer->step_pending(input,bo,previous_injection,dispatch,route,error,trace):layer->step(input,dispatch,route,error,trace);};
 check(step(experts(),&a),error);
 check(layer->native_hc_calls()==(enabled?1u:0u),"actual HC participation");
 auto first=layer->last_residual(),state=layer->recurrent_state(),history=layer->convolution_history();
 if(enabled&&ffn){check(a.ffn_hc.original==input&&a.ffn_hc.previous_output.size()==2560&&a.ffn_hc.previous_injection==a.hc_inject,"actual GDN-to-FFN tuple binding");export_ffn(out/"ffn",a.ffn_hc,f);}
 if(enabled&&final_write){
  check(layer->native_ffn_hc_write_calls()==1&&a.ffn_hc.write_candidate==first,"actual final materialization participation/consumer");
  export_write(out/"write",a.ffn_hc);
 }
 if(enabled){
  check(a.native_hc_norm_inject&&(pending?a.native_hc_pending&&a.hc_pending_input==input&&a.hc_previous_output==bo&&a.hc_previous_inject==previous_injection:a.hc_residual==input),"post-graph input/tuple preservation/upload");
  if(pending){save(out/"residual.f32le",a.hc_residual);save(out/"pending-gate.f32le",a.hc_pending_gate);save(out/"pending-exp.f32le",a.hc_pending_exp);save(out/"unapplied-residual.f32le",a.hc_pending_input);save(out/"previous-output.f32le",a.hc_previous_output);save(out/"previous-injection.f32le",a.hc_previous_inject);}

  save(out/"normalized.f32le",a.hc_normalized);save(out/"rs.f32le",a.hc_rs);save(out/"inject.f32le",a.hc_inject);
  if(composed) {
   check(a.native_hc_mix&&layer->native_hc_mix_calls()==1,"composed participation");check(std::filesystem::create_directory(out/"mix"),"fresh mix observations");
   for(auto pair:{std::pair{"low-pre",&a.hc_low_pre},std::pair{"low",&a.hc_low},std::pair{"low-exp",&a.hc_low_exp},std::pair{"up-pre",&a.hc_up_pre},std::pair{"gates",&a.hc_gates},std::pair{"up-exp",&a.hc_up_exp},std::pair{"mixed",&a.hc_mixed}}){save(out/"mix"/(std::string(pair.first)+".f32le"),*pair.second);if(pending)save(out/(std::string(pair.first)+".f32le"),*pair.second);}
   std::ofstream fixture(out/"mix-fixture.bin",std::ios::binary);fixture.write("HCMIX01\0",8);fixture.write(reinterpret_cast<const char*>(a.hc_normalized.data()),40960);
   for(const char*role:{"blk.0.hc_attn_down.weight","blk.0.hc_attn_up.weight"})for(const auto&v:f.views)if(v.name==role){const auto bf16=strata::core::portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);fixture.write(reinterpret_cast<const char*>(bf16.data()),bf16.size()*2);}
   check(bool(fixture),"composed fixture write");
  }
 }
 check(layer->reset(error),error);PortableGdnTrace b;
 check(step(experts(),&b),error);
 check(first==layer->last_residual()&&state==layer->recurrent_state()&&history==layer->convolution_history(),"reset/replay state/residual differs");
 if(enabled&&ffn)check(a.ffn_hc.residual==b.ffn_hc.residual&&a.ffn_hc.normalized==b.ffn_hc.normalized&&a.ffn_hc.mixed==b.ffn_hc.mixed,"FFN reset/repeat differs");
 if(enabled)check(a.hc_normalized==b.hc_normalized&&a.hc_rs==b.hc_rs&&a.hc_inject==b.hc_inject&&a.hc_mixed==b.hc_mixed&&a.hc_low==b.hc_low,"repeat HC differs");
 check(layer->reset(error),error);const auto before=layer->recurrent_state(),before_history=layer->convolution_history();const auto before_tokens=layer->token_history();
 PortableExpertRows refusal=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional fixture provider refusal";return false;};
 check(!step(refusal),"provider failure accepted");check(layer->tokens()==0&&layer->last_residual().empty()&&layer->recurrent_state()==before&&layer->convolution_history()==before_history&&layer->token_history()==before_tokens,"provider failure committed state/history/token metadata");
 check(layer->reset(error),error);
 check(layer->native_ffn_hc_write_calls()==(enabled&&final_write?2u:0u),"final write paid work/provider rollback count");
 check(layer->native_ffn_hc_pending_calls()==(enabled&&ffn?3u:0u),"actual half-1 work counter");
 check(layer->native_hc_mix_calls()==(enabled&&composed?3u:0u),"actual mixed work counter");
 check(layer->native_hc_pending_calls()==(enabled&&pending?3u:0u),"actual pending work counter");
 printf("{\"native_layer0\":%s,\"actual_component_calls\":%llu,\"pending_calls\":%llu,\"weight_bytes\":%zu,\"graph_bytes\":%zu,\"reset_repeat_exact\":true,\"provider_rollback\":true}\n",enabled?"true":"false",(unsigned long long)layer->native_hc_calls(),(unsigned long long)layer->native_hc_pending_calls(),layer->weight_bytes(),layer->graph_bytes());
 if(ffn)printf("{\"actual_ffn_pending_calls\":%llu,\"actual_ffn_write_calls\":%llu}\n",(unsigned long long)layer->native_ffn_hc_pending_calls(),(unsigned long long)layer->native_ffn_hc_write_calls());
 return {first,state,history,layer->weight_bytes(),a.hc_residual,std::move(a.ffn_hc)};
}
void boundary(const Fixture&f,ggml_backend_t backend,const std::vector<float>&input,const std::filesystem::path&out){
 auto renamed=[&](int l){auto v=f.views;for(auto&t:v)t.name.replace(0,6,"blk."+std::to_string(l)+".");return v;};
 auto dispatch=[](int64_t l,const PortableRoute&r,std::vector<float>&rows,std::string&e){check(l==0||l==1||l==2,"boundary source layer");return experts()(0,r,rows,e);};
 PortableGdnOptions a;a.execution.threads=1;a.execution.accelerator=backend;a.execution.request_accelerator=true;a.defer_ffn_hc_write=true;
 std::string error;auto producer=PortableGdnLayer::create(1,renamed(1),a,error);check(bool(producer),error);check(producer->reset(error),error);
 PortableGdnOptions b;b.execution=a.execution;b.resident_attention_weights=true;b.native_hc_norm_inject=true;b.native_hc_mix=true;b.native_hc_pending=true;b.native_hc_receipt_prefix=(out/"consumer").string();
 auto consumer=PortableGdnLayer::create(2,renamed(2),b,error);check(bool(consumer),error);check(consumer->reset(error),error);PortableRoute route;PortableGdnTrace t;
 check(producer->step(input,dispatch,route,error),error);const auto tuple=producer->pending_write();
 check(consumer->step_pending(tuple.residual,tuple.block,tuple.injection,dispatch,route,error,&t),error);
 check(t.hc_pending_input==tuple.residual&&t.hc_previous_output==tuple.block&&t.hc_previous_inject==tuple.injection,"actual two-layer tuple binding");
 const auto result=consumer->last_residual(),state=consumer->recurrent_state(),history=consumer->convolution_history();
 native_hc_pending_fixture::Pending fixture;fixture.norm.t=1;fixture.norm.x=tuple.residual;fixture.bo=tuple.block;fixture.inj=tuple.injection;
 for(const auto&v:f.views){if(v.name=="blk.0.hc_attn_norm.weight"){fixture.norm.w.resize(10240);memcpy(fixture.norm.w.data(),v.data,40960);}
 if(v.name=="blk.0.hc_attn_inject.weight")fixture.norm.wi=strata::core::portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);
 if(v.name=="blk.0.hc_attn_down.weight")fixture.down=strata::core::portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
 if(v.name=="blk.0.hc_attn_up.weight")fixture.up=strata::core::portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);}
 native_hc_pending_fixture::write((out/"fixture.bin").string(),fixture);
 for(auto pair:{std::pair{"residual",&t.hc_residual},std::pair{"pending-gate",&t.hc_pending_gate},std::pair{"pending-exp",&t.hc_pending_exp},std::pair{"rs",&t.hc_rs},std::pair{"normalized",&t.hc_normalized},std::pair{"inject",&t.hc_inject},std::pair{"low-pre",&t.hc_low_pre},std::pair{"low",&t.hc_low},std::pair{"low-exp",&t.hc_low_exp},std::pair{"up-pre",&t.hc_up_pre},std::pair{"gates",&t.hc_gates},std::pair{"up-exp",&t.hc_up_exp},std::pair{"mixed",&t.hc_mixed}})save(out/(std::string(pair.first)+".f32le"),*pair.second);
 // Layer number is a test binding, not arithmetic: independent materialized
 // candidate feeds the unchanged ordinary layer0 native read control.
 auto c=b;c.native_hc_pending=false;c.native_hc_receipt_prefix=(out/"control").string();auto control=PortableGdnLayer::create(0,f.views,c,error);check(bool(control),error);check(control->reset(error),error);
 check(control->step(t.hc_residual,dispatch,route,error),error);check(control->last_residual()==result&&control->recurrent_state()==state&&control->convolution_history()==history,"same-backend materialized candidate differs");
 check(producer->reset(error)&&consumer->reset(error),error);check(producer->pending_write().empty(),"reset retained prior token tuple");
 check(producer->step(input,dispatch,route,error),error);check(producer->pending_write().residual==tuple.residual&&producer->pending_write().block==tuple.block&&producer->pending_write().injection==tuple.injection,"producer repeat differs");
 check(consumer->step_pending(tuple.residual,tuple.block,tuple.injection,dispatch,route,error),error);check(consumer->last_residual()==result&&consumer->recurrent_state()==state&&consumer->convolution_history()==history,"consumer reset/replay differs");
 auto fail=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional boundary refusal";return false;};
 check(!producer->step(input,fail,route,error),"producer refusal accepted");check(producer->tokens()==1&&producer->pending_write().block==tuple.block,"producer refusal changed committed tuple");
 check(consumer->reset(error),error);auto clean=consumer->recurrent_state();check(!consumer->step_pending(tuple.residual,tuple.block,tuple.injection,fail,route,error),"consumer refusal accepted");check(consumer->tokens()==0&&consumer->last_residual().empty()&&consumer->recurrent_state()==clean,"consumer refusal committed state");
 check(consumer->step_pending(tuple.residual,tuple.block,tuple.injection,dispatch,route,error),error);check(consumer->last_residual()==result,"consumer retry differs");
 check(producer->deferred_hc_outputs()==2&&consumer->native_hc_pending_calls()==4,"actual producer/consumer paid work");
 // A disabled producer still takes the original materialized FFN path.
 a.defer_ffn_hc_write=false;auto disabled=PortableGdnLayer::create(1,renamed(1),a,error);check(bool(disabled),error);check(disabled->reset(error)&&disabled->step(input,dispatch,route,error),error);auto d=disabled->last_residual();check(disabled->reset(error)&&disabled->step(input,dispatch,route,error),error);check(disabled->last_residual()==d&&disabled->pending_write().empty()&&disabled->deferred_hc_outputs()==0,"disabled replay/carry regression");check(disabled->weight_bytes()==producer->weight_bytes(),"deferred producer duplicated weights");
 std::ofstream(out/"report.json")<<"{\"ordinary_layer1_to_layer2\":true,\"actual_producer_tuple\":true,\"deferred_productions\":2,\"pending_consumer_calls\":4,\"materialized_control_exact\":true,\"reset_rollback_retry\":true,\"disabled_repeat_exact\":true,\"synthetic_two_layer_fixture\":true,\"actual_layer1_PLE_executed\":false,\"qualified\":false,\"full_hc_parity\":false}\n";
}

}
int main(int argc,char**argv){try{
 check(argc==4||(argc==5&&(std::string(argv[4])=="read"||std::string(argv[4])=="pending"||std::string(argv[4])=="ffn"||std::string(argv[4])=="write"||std::string(argv[4])=="boundary")),"usage: portable_gdn_native_hc_probe OPERANDS FRESH_OUTPUT EXPECT_DEVICE [read|pending|ffn|write|boundary]");const bool composed=argc==5;const bool pending=composed&&std::string(argv[4])=="pending";const bool final_write=composed&&std::string(argv[4])=="write";const bool ffn=composed&&(std::string(argv[4])=="ffn"||final_write);const std::filesystem::path d=argv[1],out=argv[2];check(std::filesystem::create_directory(out),"fresh output directory");
 auto norm=load(d/"portable-hc_attn_norm.weight.bin",40960),inj=load(d/"portable-hc_attn_inject.weight.bin",43520),raw=load(d/"portable-embedding.f32le",40960);
 std::vector<uint8_t>down,up;if(composed){down=load(d/"portable-hc_attn_down.weight.bin",3481600);up=load(d/"portable-hc_attn_up.weight.bin",3481600);}
 std::vector<float>input(10240);memcpy(input.data(),raw.data(),40960);Fixture f;
 for(auto&v:f.views){if(v.name=="blk.0.hc_attn_norm.weight")v.data=norm.data();if(v.name=="blk.0.hc_attn_inject.weight")v.data=inj.data();if(composed&&v.name=="blk.0.hc_attn_down.weight")v.data=down.data();if(composed&&v.name=="blk.0.hc_attn_up.weight")v.data=up.data();}
 check(ggml_backend_vk_get_device_count()>0,"real Vulkan required");char desc[256]{};ggml_backend_vk_get_device_description(0,desc,sizeof(desc));check(std::string(desc).find(argv[3])!=std::string::npos,"device identity");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_vk_init(0),ggml_backend_free);check(bool(backend),"Vulkan initialization");
 if(argc==5&&std::string(argv[4])=="boundary"){boundary(f,backend.get(),input,out);return 0;}
 const auto active=run(f,backend.get(),true,composed,input,out,pending,ffn,final_write);
 if(ffn){
  const auto control_dir=out/"materialized-ffn";check(std::filesystem::create_directory(control_dir),"fresh FFN control");
  PortableFfnOptions o;o.threads=1;o.accelerator=backend.get();o.request_accelerator=true;o.resident_dense_weights=true;o.native_hc_pending_read=true;o.native_hc_final_write=final_write;o.native_hc_receipt_prefix=(control_dir/"hc").string();
  std::vector<PortableTensorView> weights;for(const auto&v:f.views)if(v.name.find("hc_ffn_")!=std::string::npos||v.name.find(".ffn_")!=std::string::npos)weights.push_back(v);
  std::string error;auto control=PortableFfn::create({2560,4,320,640,512,10,1e-6f},0,weights,o,error);check(bool(control),error);
  check(control->reset(active.ffn.residual,error),error);PortableRoute route;PortableFfnHcTrace trace;
  check(control->step_pending(std::vector<float>(2560,0),std::vector<float>(4,0),experts(),route,error,&trace),error);
  check(control->residual()==active.residual&&trace.normalized==active.ffn.normalized&&trace.rs==active.ffn.rs&&trace.inject==active.ffn.inject&&trace.mixed==active.ffn.mixed,"actual FFN candidate downstream control differs");
  export_ffn(control_dir/"ffn",trace,f);
  if(final_write){check(control->native_hc_write_calls()==1&&control->work().post_graph_calls==0,"control did not use final component replacement");export_write(control_dir/"write",trace);}
 }
 if(pending){
  check(std::filesystem::create_directory(out/"materialized-control"),"fresh materialized control");
  const auto control=run(f,backend.get(),true,true,active.hc_input,out/"materialized-control");
  check(control.residual==active.residual&&control.state==active.state&&control.history==active.history,"pending graph hidden/state differs from same-backend materialized R-prime");
 }
 std::filesystem::create_directory(out/"disabled-a");std::filesystem::create_directory(out/"disabled-b");
 const auto a=run(f,backend.get(),false,composed,input,out/"disabled-a",false,ffn),b=run(f,backend.get(),false,composed,input,out/"disabled-b",false,ffn);
 check(a.residual==b.residual&&a.state==b.state&&a.history==b.history,"disabled repeated path changed");
 check(active.weights==a.weights+(ffn?12364800u:(composed?6182400u:38400u)),"converted weights must replace original banks exactly");
 native_hc_fixture::Fixture fixture;fixture.t=1;fixture.x=pending?active.hc_input:input;fixture.w.resize(10240);memcpy(fixture.w.data(),norm.data(),40960);fixture.wi=strata::core::portable_detail::native_hc_q8_bf16(inj.data(),inj.size(),40960);
 native_hc_fixture::write_fixture((out/"fixture.bin").string(),fixture);
 if(pending){
  native_hc_pending_fixture::Pending pf;pf.norm=fixture;pf.norm.x=input;pf.bo.resize(2560);for(size_t d=0;d<2560;++d)pf.bo[d]=float(int(d%11)-5)*0.001953125f;pf.inj={-8.f,-1.f,0.f,4.f};
  pf.down=strata::core::portable_detail::native_hc_q8_bf16(down.data(),down.size(),3276800);pf.up=strata::core::portable_detail::native_hc_q8_bf16(up.data(),up.size(),3276800);
  native_hc_pending_fixture::write((out/"pending-fixture.bin").string(),pf);
 }
 std::ofstream report(out/"report.json");if(ffn)report<<"{\"native_final_write\":"<<(final_write?"true":"false")<<",\"actual_gdn_to_ffn\":"<<(ffn?"true":"false")<<",\"ffn_materialized_control_exact\":"<<(ffn?"true":"false")<<",\"metadata\":";report<<"{\"tokens\":1,\"guards\":true,\"finite_normal_domain\":true,\"repeat_exact\":true,\"qualified\":false,\"serial_layer0_integration\":true,\"composed_read\":"<<(composed?"true":"false")<<",\"pending_read\":"<<(pending?"true":"false")<<",\"materialized_candidate_control_exact\":"<<(pending?"true":"false")<<",\"full_hc_parity\":false,\"reset_repeat_exact\":true,\"provider_rollback\":true,\"disabled_repeat_exact\":true}";if(ffn)report<<"}";report<<"\n";
 return 0;
}catch(const std::exception&e){fprintf(stderr,"STOP: %s\n",e.what());return 1;}}
