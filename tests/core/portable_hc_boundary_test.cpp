// Actual two-layer producer/consumer, mocked device arithmetic; no GPU proof.
#include "portable_native_hc_test_backend.hpp"
#ifdef STRATA_BOUNDARY_QSA
#include "portable_qsa_fixture.hpp"
#endif
int main(){try{
 Fixture f;std::string error;auto root=std::filesystem::temp_directory_path()/("hc-boundary-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));check(std::filesystem::create_directory(root),"fresh root");
 for(const auto&v:f.views){
 if(v.name=="blk.0.hc_attn_inject.weight")expected_weights=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);
 if(v.name=="blk.0.hc_attn_down.weight")expected_down=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
 if(v.name=="blk.0.hc_attn_up.weight")expected_up=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
 }
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>cpu(ggml_backend_cpu_init(),ggml_backend_free);
 auto named=[&](int layer){auto views=f.views;for(auto&v:views)v.name.replace(0,6,"blk."+std::to_string(layer)+".");return views;};
 auto dispatch=[](int64_t l,const PortableRoute&r,std::vector<float>&rows,std::string&e){check(l==1||l==2||l==0,"unexpected source layer");return experts()(0,r,rows,e);};
 PortableGdnOptions a;a.execution.threads=1;a.execution.accelerator=cpu.get();a.execution.request_accelerator=true;a.defer_ffn_hc_write=true;
 a.scratch=PortableGdnScratch::create(a.execution,error);check(bool(a.scratch),error);
 auto producer=PortableGdnLayer::create(1,named(1),a,error);check(bool(producer),error);check(producer->reset(error),error);PortableRoute route;
 check(!producer->pending_snapshot(error),"snapshot before commit");
 check(producer->step(f.input(0),dispatch,route,error),error);const auto tuple=producer->pending_write();
 auto snap0=producer->pending_snapshot(error);check(bool(snap0),error);check(snap0->layer()==1&&snap0->position()==0&&snap0->tuple().residual==tuple.residual,"snapshot metadata/tuple");
 check(tuple.residual==producer->last_residual()&&tuple.block.size()==2560&&tuple.injection.size()==4,"incomplete actual producer tuple");check(producer->deferred_hc_outputs()==1,"producer count");
 std::vector<float> applied(10240);
 for(size_t c=0;c<4;++c){const float scale=std::fma(0.25f,tuple.injection[c],0.0f);const float gate=std::fma(2.0f,1.0f/(1.0f+std::exp(-scale)),0.0f);for(size_t d=0;d<2560;++d)applied[c*2560+d]=std::fma(tuple.block[d],gate,tuple.residual[c*2560+d]);}
 PortableGdnOptions b;b.execution=a.execution;b.scratch=a.scratch;b.resident_attention_weights=true;b.native_hc_norm_inject=true;b.native_hc_mix=true;b.native_hc_pending=true;b.native_hc_receipt_prefix=(root/"consumer").string();
 auto consumer=PortableGdnLayer::create(2,named(2),b,error);check(bool(consumer),error);check(consumer->reset(error),error);PortableGdnTrace trace;
 auto wrong_owner=PortableGdnLayer::create(1,named(1),a,error);check(bool(wrong_owner),error);check(wrong_owner->reset(error),error);
 check(!consumer->step_pending(*wrong_owner,*snap0,dispatch,route,error)&&consumer->tokens()==0,"wrong producer accepted");
 check(consumer->step_pending(*producer,*snap0,dispatch,route,error,&trace),error);
 check(!consumer->step_pending(*producer,*snap0,dispatch,route,error)&&consumer->tokens()==1,"wrong position accepted");
 check(trace.hc_pending_input==tuple.residual&&trace.hc_previous_output==tuple.block&&trace.hc_previous_inject==tuple.injection,"actual inter-layer tuple replaced/stale");
 check(trace.hc_residual==applied&&uploaded==applied,"native FMA-before-normalization/source stream order");
 const auto result=consumer->last_residual(),state=consumer->recurrent_state(),history=consumer->convolution_history();
 auto c=b;c.native_hc_pending=false;c.native_hc_receipt_prefix=(root/"materialized-control").string();auto control=PortableGdnLayer::create(0,f.views,c,error);check(bool(control),error);check(control->reset(error),error);
 check(!control->step_pending(*producer,*snap0,dispatch,route,error)&&control->tokens()==0,"wrong layer accepted");
 check(control->step(applied,dispatch,route,error),error);check(control->last_residual()==result&&control->recurrent_state()==state&&control->convolution_history()==history,"materialized-candidate control differs");
 check(consumer->reset(error),error);const auto clean=consumer->recurrent_state(),clean_history=consumer->convolution_history();
 auto fail=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional boundary provider refusal";return false;};
 check(!consumer->step_pending(tuple.residual,tuple.block,tuple.injection,fail,route,error),"failed consumer accepted");check(consumer->tokens()==0&&consumer->last_residual().empty()&&consumer->recurrent_state()==clean&&consumer->convolution_history()==clean_history,"consumer failure committed state");
 check(producer->pending_write().block==tuple.block&&producer->pending_write().residual==tuple.residual,"consumer mutated upstream tuple");
 check(consumer->step_pending(tuple.residual,tuple.block,tuple.injection,dispatch,route,error),error);check(consumer->last_residual()==result&&consumer->recurrent_state()==state,"consumer retry differs");
 check(!producer->step(f.input(1),fail,route,error),"producer failure accepted");check(producer->tokens()==1&&producer->pending_write().block==tuple.block&&producer->pending_write().injection==tuple.injection,"producer failure replaced committed tuple");
 check(!producer->pending_snapshot(error),"failure published snapshot");
 check(producer->step(f.input(1),dispatch,route,error),error);auto snap1=producer->pending_snapshot(error);check(bool(snap1)&&snap1->position()==1,error);
 check(snap0->tuple().residual==tuple.residual&&snap0->position()==0,"successive commit mutated owned snapshot");
 check(consumer->reset(error),error);check(consumer->step_pending(*producer,*snap0,dispatch,route,error),error);
 check(consumer->step_pending(*producer,*snap1,dispatch,route,error),error);
 check(producer->reset(error)&&producer->pending_write().empty()&&producer->deferred_hc_outputs()==2,"reset tuple lifetime/paid count");
 check(consumer->reset(error),error);check(!consumer->step_pending(*producer,*snap0,dispatch,route,error)&&consumer->tokens()==0,"reset accepted stale snapshot");
 check(producer->step(f.input(0),dispatch,route,error),error);check(producer->pending_write().residual==tuple.residual&&producer->pending_write().block==tuple.block,"producer reset/replay differs");
 // Disabled producer still materializes; deferred weights are identical and post scheduler is omitted.
 auto ordinary=a;ordinary.defer_ffn_hc_write=false;auto disabled=PortableGdnLayer::create(1,named(1),ordinary,error);check(bool(disabled),error);check(disabled->reset(error),error);check(disabled->step(f.input(0),dispatch,route,error),error);check(disabled->pending_write().empty()&&disabled->deferred_hc_outputs()==0,"default published carry");check(disabled->weight_bytes()==producer->weight_bytes(),"carry duplicated weights");check(!producer->prepare_chunk(2,error),"deferred producer accepted chunk");
 {
 std::vector<PortableTensorView> fw;for(auto v:named(1))if(v.name.find(".ffn_")!=std::string::npos||v.name.find(".hc_ffn_")!=std::string::npos)fw.push_back(v);
 auto options=a.execution;options.defer_hc_write=true;auto owner=PortableFfn::create({2560,4,320,640,512,10,1e-6f},1,fw,options,error);check(bool(owner),error);
 check(owner->reset(f.input(0),error)&&owner->step(dispatch,route,error),error);check(owner->work().post_graph_calls==0&&!owner->pending_write().empty(),"defer executed unused post graph");
 check(!owner->step(dispatch,route,error),"deferred FFN silently reused an unconsumed tuple");
 const auto graph=owner->graph_bytes(),weights=owner->weight_bytes();auto short_budget=options;short_budget.graph_budget=graph-1;check(!PortableFfn::create({2560,4,320,640,512,10,1e-6f},1,fw,short_budget,error),"one-byte-short deferred graph accepted");
 check(!owner->create_chunk_execution(2,error),"deferred owner accepted chunk bypass");
 options.defer_hc_write=false;auto materialized=PortableFfn::create({2560,4,320,640,512,10,1e-6f},1,fw,options,error);check(bool(materialized),error);check(materialized->weight_bytes()==weights,"defer duplicated weight bank");
 check(materialized->reset(f.input(0),error)&&materialized->step(dispatch,route,error),error);check(materialized->work().post_graph_calls==1,"control post graph missing");
 }
 #ifdef STRATA_BOUNDARY_QSA
 {
 qsa_fixture::Fixture qf; PortableQsaOptions qo; qo.capacity=4; qo.execution=a.execution; qo.defer_ffn_hc_write=true;
 auto q=PortableQsaLayer::create(3,qf.views,qo,error);check(bool(q),error);check(q->reset(error),error);
 check(!q->pending_snapshot(error),"QSA snapshot before commit");
 check(q->step(0,qf.input(0),qsa_fixture::experts(),route,error),error);
 auto s0=q->pending_snapshot(error);check(bool(s0),error);auto saved=s0->tuple();
 check(s0->layer()==3&&s0->position()==0&&s0->valid_for(*q,4,0)&&!s0->valid_for(*q,5,0)&&!s0->valid_for(*q,4,1),"QSA snapshot metadata validation");
 auto other=PortableQsaLayer::create(3,qf.views,qo,error);check(bool(other),error);check(!s0->valid_for(*other,4,0),"QSA wrong owner accepted");
 check(!q->step(1,qf.input(1),fail,route,error)&&q->tokens()==1&&!q->pending_snapshot(error),"QSA failure published snapshot");
 check(q->step(1,qf.input(1),qsa_fixture::experts(),route,error),error);auto s1=q->pending_snapshot(error);check(bool(s1)&&s1->position()==1&&s1->valid_for(*q,4,1),"QSA second commit");
 check(s0->tuple().residual==saved.residual&&s0->tuple().block==saved.block&&s0->tuple().injection==saved.injection&&s0->valid_for(*q,4,0),"QSA retained snapshot changed");
 check(!q->prepare_chunk(2,error),"QSA deferred chunk accepted");
 check(q->reset(error)&&!s0->valid_for(*q,4,0)&&!s1->valid_for(*q,4,1),"QSA reset accepted stale snapshot");
 check(s0->tuple().block==saved.block,"QSA reset changed retained values");
 check(q->step(0,qf.input(0),qsa_fixture::experts(),route,error),error);auto after_reset=q->pending_snapshot(error);check(bool(after_reset)&&after_reset->generation()!=s0->generation(),"QSA reset generation");
 q.reset();check(!after_reset->valid_for(*other,4,0)&&after_reset->tuple().block==saved.block,"QSA destruction retained values/owner");
 }
 #endif
 PortableModelOptions opts;opts.native_hc_pending_layer1_2=true;opts.gdn_resident_weights=true;opts.native_hc_pending_receipt_prefix=(root/"model").string();validate_model_modes(opts,1);
 PortableGdnOptions o0,o1,o2,o3;model_gdn_hc_options(opts,0,o0);model_gdn_hc_options(opts,1,o1);model_gdn_hc_options(opts,2,o2);model_gdn_hc_options(opts,3,o3);
 check(!o0.defer_ffn_hc_write&&!o0.native_hc_pending&&!o1.native_hc_norm_inject&&o1.defer_ffn_hc_write&&o2.native_hc_pending&&o2.native_hc_mix&&!o3.native_hc_norm_inject,"model widened scope or bypassed layer1 PLE");
 check(model_hc_host_bound(opts,1)==PortableFfnOptions::deferred_hc_host_bound&&model_hc_host_bound(opts,2)==PortableGdnOptions::native_hc_pending_host_bound&&model_hc_host_bound(opts,0)==0,"scoped complete host bounds");
 std::filesystem::remove_all(root);puts("layer1 producer -> layer2 native pending read, independent FMA/materialized control, reset/rollback/scope pass; mocked device only");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
