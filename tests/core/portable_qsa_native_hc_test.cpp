// Actual GDN producer/QSA consumer, shared attention arena, mocked HC device only.
#include "portable_native_hc_test_backend.hpp"
#include "portable_qsa_fixture.hpp"
namespace {
bool inspect_qsa=false;size_t qsa_consumers=0;
ggml_status qsa_compute(ggml_backend_t b,ggml_cgraph*g){
 if(inspect_qsa)for(int n=0;n<ggml_graph_n_nodes(g);++n){auto*t=ggml_graph_node(g,n);if(t->op!=GGML_OP_MUL_MAT||!t->src[0])continue;const std::string name=ggml_get_name(t->src[0]);
 if(name=="blk.3.attn_q.weight"||name=="blk.3.attn_k.weight"||name=="blk.3.attn_v.weight"||name=="blk.3.indexer.q_proj.weight"||name=="blk.3.indexer.k_proj.weight"){
 auto*root=t->src[1];while(root->view_src)root=root->view_src;check(root==live_mixed,"QSA projection/indexer ignored actual native mixed leaf");++qsa_consumers;}}
 return original_compute(b,g);
}
}
int main(){try{
 qsa_fixture::Fixture qf;auto&f=qf.base;std::string error;auto root=std::filesystem::temp_directory_path()/("qsa-pending-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));check(std::filesystem::create_directory(root),"fresh root");
 for(auto&v:qf.views){if(v.name=="blk.3.hc_attn_inject.weight")expected_weights=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);if(v.name=="blk.3.hc_attn_down.weight")expected_down=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);if(v.name=="blk.3.hc_attn_up.weight")expected_up=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);}
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>cpu(ggml_backend_cpu_init(),ggml_backend_free);original_compute=cpu->iface.graph_compute;cpu->iface.graph_compute=qsa_compute;
 PortableFfnOptions exec;exec.threads=1;exec.accelerator=cpu.get();exec.request_accelerator=true;
 PortableGdnOptions p;p.execution=exec;p.defer_ffn_hc_write=true;p.scratch=PortableGdnScratch::create(exec,error);check(bool(p.scratch),error);
 auto gv=f.views;for(auto&v:gv)v.name.replace(0,6,"blk.2.");auto producer=PortableGdnLayer::create(2,gv,p,error);check(bool(producer),error);check(producer->reset(error),error);
 PortableQsaOptions o;o.execution=exec;o.capacity=8;o.reuse_attention=true;o.attention=PortableQsaAttention::create(exec,error);check(bool(o.attention),error);
 auto warmv=qf.views;for(auto&v:warmv)v.name.replace(0,6,"blk.11.");auto warm=PortableQsaLayer::create(11,warmv,o,error);check(bool(warm),error);check(warm->reset(error),error);
 auto disabled=PortableQsaLayer::create(3,qf.views,o,error);check(bool(disabled),error);

 // Native QSA model seam: actual completed projection outputs, not graph-time
 // placeholders. The component primitive alone is mocked on this CPU backend.
 auto nr=o;nr.native_norm_rope_weights=true;nr.native_norm_rope_receipt_prefix=(root/"nr-control").string();
 auto matched=PortableQsaLayer::create(3,qf.views,nr,error);check(bool(matched),error);
 nr.native_norm_rope_t1=true;nr.native_norm_rope_record_first=true;nr.native_norm_rope_receipt_prefix=(root/"nr-on").string();
 auto native_qsa=PortableQsaLayer::create(3,qf.views,nr,error);check(bool(native_qsa),error);
 check(matched->weight_bytes()==native_qsa->weight_bytes(),"QSA norm roles not identically resident");
 check(matched->weight_bytes()<=disabled->weight_bytes()+4096,"QSA norm bank duplicated source weights");
 check(native_qsa->memory_usage().host_extra_bound>=PortableQsaOptions::native_norm_rope_host_bound,"QSA norm host allowance omitted");
 PortableRoute nr_route;auto nr_dispatch=[](int64_t,const PortableRoute&r,std::vector<float>&rows,std::string&e){return gdn_fixture::experts()(0,r,rows,e);};
 check(disabled->reset(error)&&matched->reset(error)&&native_qsa->reset(error),error);std::vector<float>nr_first;
 for(size_t pos=0;pos<4;++pos){
  check(disabled->step(pos,f.input(pos),nr_dispatch,nr_route,error),error);check(matched->step(pos,f.input(pos),nr_dispatch,nr_route,error),error);
  check(disabled->last_residual()==matched->last_residual()&&disabled->keys()==matched->keys()&&disabled->pooled_keys()==matched->pooled_keys(),"explicit residency changed disabled arithmetic");
  PortableQsaTrace tr;check(native_qsa->step(pos,f.input(pos),nr_dispatch,nr_route,error,&tr),error);
  check(tr.query.size()==6144&&tr.key.size()==512&&tr.index_query.size()==512,"actual native candidate consumers lost geometry");
  check(native_qsa->native_norm_rope_calls()==3*(pos+1)&&matched->native_norm_rope_calls()==0,"native QSA paid counters wrong");
  if(pos==0)nr_first=native_qsa->last_residual();
 }
 auto nr_keys=native_qsa->keys();auto nr_values=native_qsa->values();auto nr_pool=native_qsa->pooled_keys();auto nr_tail=native_qsa->indexer_tail();auto nr_last=native_qsa->last_residual();
 fail_qsa_norm=true;check(!native_qsa->step(4,f.input(4),nr_dispatch,nr_route,error),"native QSA domain refusal accepted");fail_qsa_norm=false;
 check(native_qsa->tokens()==4&&native_qsa->keys()==nr_keys&&native_qsa->values()==nr_values&&native_qsa->pooled_keys()==nr_pool&&native_qsa->indexer_tail()==nr_tail&&native_qsa->last_residual()==nr_last,"native QSA failure committed state");
 check(native_qsa->native_norm_rope_calls()==13,"refused completed QSA work hidden");
 check(std::filesystem::exists(root/"nr-on/failure-q-4/fixture.bin")&&!std::filesystem::exists(root/"nr-on/failure-q-4/rotated.f32le"),"failure published stale/uninitialized candidate values");
 check(native_qsa->step(4,f.input(4),nr_dispatch,nr_route,error),error);check(native_qsa->reset(error)&&native_qsa->step(0,f.input(0),nr_dispatch,nr_route,error),error);check(native_qsa->last_residual()==nr_first,"native QSA reset/repeat changed state/output");
 auto nr_bad=nr;nr_bad.execution.weight_budget=native_qsa->weight_bytes()-1;nr_bad.native_norm_rope_receipt_prefix=(root/"nr-budget").string();check(!PortableQsaLayer::create(3,qf.views,nr_bad,error),"QSA one-byte-short weights admitted");
 nr_bad=nr;nr_bad.production=true;nr_bad.native_norm_rope_receipt_prefix=(root/"nr-production").string();check(!PortableQsaLayer::create(3,qf.views,nr_bad,error),"unsupported native QSA production mode accepted");
 // Actual raw-key graph producer, native retained-tail candidate owner and
 // existing FFN-success transaction. Arithmetic is the declared host mock.
 auto ix=o;ix.native_indexer_weights=true;ix.native_indexer_receipt_prefix=(root/"idx-control").string();
 auto idx_control=PortableQsaLayer::create(3,qf.views,ix,error);check(bool(idx_control),error);
 ix.native_indexer_t1=true;ix.native_indexer_record_first=true;ix.native_indexer_receipt_prefix=(root/"idx-on").string();
 auto idx=PortableQsaLayer::create(3,qf.views,ix,error);check(bool(idx),error);
 check(idx_control->weight_bytes()==idx->weight_bytes()&&idx->weight_bytes()<=disabled->weight_bytes()+4096,"indexer resident F32 role duplicated/unequal");
 check(idx->memory_usage().host_weights<disabled->memory_usage().host_weights&&idx->memory_usage().device_weights>disabled->memory_usage().device_weights,"resident indexer role did not replace host storage");
 check(idx->memory_usage().host_extra_bound>=PortableQsaOptions::native_indexer_host_bound,"native indexer host allowance missing");
 check(disabled->reset(error)&&idx_control->reset(error)&&idx->reset(error),error);std::vector<float>expected_tail(384),spare,idx_first;
 for(size_t pos=0;pos<4;++pos){
  check(disabled->step(pos,f.input(pos),nr_dispatch,nr_route,error),error);check(idx_control->step(pos,f.input(pos),nr_dispatch,nr_route,error),error);
  check(disabled->last_residual()==idx_control->last_residual()&&disabled->pooled_keys()==idx_control->pooled_keys()&&disabled->indexer_tail()==idx_control->indexer_tail(),"matched indexer residency changed disabled arithmetic/state");
  PortableQsaTrace tr;check(idx->step(pos,f.input(pos),nr_dispatch,nr_route,error,&tr),error);
  if(pos<3)for(size_t d=0;d<128;++d)expected_tail[pos*128+d]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(tr.index_raw_key[d]));
  check(idx->indexer_tail()==expected_tail,"native three-row half-rounded tail differs or was cleared on completion");
  check(idx->native_indexer_calls()==pos+1&&idx_control->native_indexer_calls()==0,"indexer work counters are not actual calls");
  if(pos==0){spare=idx->pooled_keys();idx_first=idx->last_residual();}
 }
 check(std::vector<float>(idx->pooled_keys().begin()+128,idx->pooled_keys().end())==spare,"completed block lost persistent cell-zero spare");
 for(size_t pos=0;pos<4;++pos){auto dir=root/("idx-on/first-"+std::to_string(pos));auto fixture=native_qsa_idx::read((dir/"fixture.bin").string());native_qsa_idx::check(fixture,dir);check(native_qsa_idx::words(dir/"state.u32le",fixture.before.size())==native_qsa_idx::words(dir/"accepted-state.u32le",fixture.before.size()),"serialized candidate/FFN commit link differs");}
 auto idx_state=idx->native_indexer_state();auto idx_keys=idx->keys();auto idx_pool=idx->pooled_keys();auto idx_last=idx->last_residual();
 auto idx_fail=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional post-indexer FFN refusal";return false;};
 check(!idx->step(4,f.input(4),idx_fail,nr_route,error),"failed FFN published indexer candidate");
 check(idx->native_indexer_calls()==5&&idx->tokens()==4&&idx->native_indexer_state()==idx_state&&idx->keys()==idx_keys&&idx->pooled_keys()==idx_pool&&idx->indexer_tail()==expected_tail&&idx->last_residual()==idx_last,"post-component failure changed committed native state/counters");
 fail_qsa_indexer=true;check(!idx->step(4,f.input(4),nr_dispatch,nr_route,error),"indexer domain refusal accepted");fail_qsa_indexer=false;
 check(std::filesystem::exists(root/"idx-on/failure-4-call-6-attempt-6/fixture.bin"),"current failed indexer input not retained");
 check(idx->native_indexer_calls()==6&&idx->native_indexer_state()==idx_state&&!std::filesystem::exists(root/"idx-on/failure-4-call-6-attempt-6/state.u32le"),"failed component committed state or published stale output");
 check(idx->step(4,f.input(4),nr_dispatch,nr_route,error),error);check(idx->reset(error)&&idx->step(0,f.input(0),nr_dispatch,nr_route,error),error);check(idx->last_residual()==idx_first&&idx->indexer_tail().size()==384,"indexer reset/retry changed result or native geometry");
 auto ix_bad=ix;ix_bad.execution.graph_budget=1;ix_bad.native_indexer_receipt_prefix=(root/"idx-budget").string();check(!PortableQsaLayer::create(3,qf.views,ix_bad,error),"indexer banks bypassed graph admission");
 ix_bad=ix;ix_bad.execution.weight_budget=idx->weight_bytes()-1;ix_bad.native_indexer_receipt_prefix=(root/"idx-weight-budget").string();check(!PortableQsaLayer::create(3,qf.views,ix_bad,error),"one-byte-short indexer weight allocation admitted");
 ix_bad=ix;ix_bad.state_budget=idx->state_bytes()-1;ix_bad.native_indexer_receipt_prefix=(root/"idx-state-budget").string();check(!PortableQsaLayer::create(3,qf.views,ix_bad,error),"indexer host state allowance bypassed state budget");
 ix_bad=ix;ix_bad.production=true;ix_bad.native_indexer_receipt_prefix=(root/"idx-production").string();check(!PortableQsaLayer::create(3,qf.views,ix_bad,error),"unsupported native indexer production accepted");
 check(disabled->reset(error),error);
 o.native_hc_read=true;o.native_hc_pending=true;o.native_hc_receipt_prefix=(root/"pending").string();
 auto active=PortableQsaLayer::create(3,qf.views,o,error);check(bool(active),error);check(active->reset(error),error);
 auto co=o;co.native_hc_pending=false;co.native_hc_receipt_prefix=(root/"control").string();auto control=PortableQsaLayer::create(3,qf.views,co,error);check(bool(control),error);check(control->reset(error),error);
 check(active->weight_bytes()==disabled->weight_bytes()+6182400u,"HC conversion did not replace exactly one Q8 bank");
 const auto memory=active->memory_usage();check(memory.device_weights>=6182400u&&memory.host_extra_bound>=PortableQsaOptions::native_hc_host_bound,"resident HC allocation/host envelope missing from memory ledger");
 auto dispatch=[](int64_t l,const PortableRoute&r,std::vector<float>&rows,std::string&e){check(l==2||l==3||l==11,"actual source layer mismatch");return gdn_fixture::experts()(0,r,rows,e);};
 PortableRoute route;std::vector<float> first;PortableHcPendingWrite last;
 for(size_t pos=0;pos<4;++pos){
 check(producer->step(f.input(pos),dispatch,route,error),error);auto tuple=producer->pending_write();last=tuple;
 check(warm->step(pos,f.input(pos),dispatch,route,error),error); // different layer reuses the shared arena first
 std::vector<float> applied(10240);for(size_t c=0;c<4;++c){const float gate=std::fma(2.f,1.f/(1.f+std::exp(-std::fma(.25f,tuple.injection[c],0.f))),0.f);for(size_t d=0;d<2560;++d)applied[c*2560+d]=std::fma(tuple.block[d],gate,tuple.residual[c*2560+d]);}
 PortableQsaTrace trace;inspect_qsa=true;check(active->step_pending(pos,tuple,dispatch,route,error,&trace),error);inspect_qsa=false;
 check(trace.hc.original==tuple.residual&&trace.hc.previous_output==tuple.block&&trace.hc.previous_injection==tuple.injection&&trace.hc.residual==applied,"QSA changed tuple/operation order");
 inspect_qsa=true;check(control->step(pos,applied,dispatch,route,error),error);inspect_qsa=false;
 check(active->last_residual()==control->last_residual()&&active->keys()==control->keys()&&active->values()==control->values()&&active->pooled_keys()==control->pooled_keys()&&active->indexer_tail()==control->indexer_tail(),"QSA HC old-R consumer/KV/indexer/control differs");
 if(pos==0)first=active->last_residual();
 }
 check(active->tokens()==4&&active->pooled_keys().size()==256&&active->indexer_tail()==std::vector<float>(512,0),"QSA block completion/spare/tail semantics changed");
 auto keys=active->keys(),values=active->values();auto pool=active->pooled_keys(),tail=active->indexer_tail(),res=active->last_residual();const auto before=active->native_hc_pending_calls();
 auto fail=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional QSA provider refusal";return false;};
 inspect_qsa=true;check(!active->step_pending(4,last,fail,route,error),"QSA failed provider accepted");inspect_qsa=false;
 check(active->tokens()==4&&active->keys()==keys&&active->values()==values&&active->pooled_keys()==pool&&active->indexer_tail()==tail&&active->last_residual()==res&&!active->poisoned(),"QSA failure committed candidate or changed eager semantics");
 check(active->native_hc_pending_calls()==before+1,"paid consumer work lost on later provider refusal");
 inspect_qsa=true;check(active->step_pending(4,last,dispatch,route,error),error);check(control->step(4,native_hc_pending_control(last.residual,last.block,last.injection).residual,dispatch,route,error),error);inspect_qsa=false;check(active->keys()==control->keys()&&active->pooled_keys()==control->pooled_keys()&&active->last_residual()==control->last_residual(),"QSA retry diverged");
 const auto paid=active->native_hc_pending_calls();check(active->reset(error)&&producer->reset(error),error);check(active->native_hc_pending_calls()==paid&&active->keys().empty()&&active->indexer_tail()==std::vector<float>(512,0),"QSA reset lost paid work/state");
 check(producer->step(f.input(0),dispatch,route,error),error);inspect_qsa=true;check(active->step_pending(0,producer->pending_write(),dispatch,route,error),error);inspect_qsa=false;check(active->last_residual()==first,"QSA reset/replay differs");
 check(!active->step(1,f.input(1),dispatch,route,error),"QSA accepted implicit missing tuple");check(!active->prepare_chunk(2,error),"QSA native HC chunk bypass");
 auto invalid=o;invalid.production=true;check(!PortableQsaLayer::create(3,qf.views,invalid,error),"production mode admitted");invalid=o;invalid.execution.allow_cpu_fallback=true;check(!PortableQsaLayer::create(3,qf.views,invalid,error),"fallback mode admitted");
 invalid=o;invalid.execution.weight_budget=active->weight_bytes()-1;check(!PortableQsaLayer::create(3,qf.views,invalid,error),"one-byte-short aligned HC weight budget admitted");
 PortableModelOptions model;model.native_hc_pending_layer2_3=true;model.gdn_resident_weights=true;model.native_hc_qsa_receipt_prefix=(root/"model").string();validate_model_modes(model,1);PortableGdnOptions g0,g2;PortableQsaOptions q3,q7;model_gdn_hc_options(model,0,g0);model_gdn_hc_options(model,2,g2);model_qsa_hc_options(model,3,q3);model_qsa_hc_options(model,7,q7);
 check(!g0.defer_ffn_hc_write&&g2.defer_ffn_hc_write&&q3.native_hc_pending&&!q7.native_hc_read,"scope/PLE/other QSA changed");check(!model_resident_dense(model,3,false),"native HC silently enabled all QSA dense residency");
 check(model_hc_host_bound(model,2)==PortableFfnOptions::deferred_hc_host_bound&&model_hc_host_bound(model,3)==PortableQsaOptions::native_hc_host_bound,"new host envelope unpriced");model.native_hc_pending_layer1_2=true;model.native_hc_pending_receipt_prefix=(root/"both").string();model_gdn_hc_options(model,2,g2);check(g2.native_hc_pending&&g2.defer_ffn_hc_write,"adjacent native boundaries failed composition");
 check(qsa_consumers>=50,"actual QSA projection/indexer consumers not checked");cpu->iface.graph_compute=original_compute;std::filesystem::remove_all(root);puts("actual layer2 FFN to QSA3 pending read/shared scratch/independent FMA/KV-indexer rollback/reset/accounting pass; mocked HC only");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
