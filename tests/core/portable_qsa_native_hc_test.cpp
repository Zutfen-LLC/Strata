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
