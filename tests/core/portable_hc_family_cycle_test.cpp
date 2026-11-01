// Real family coordinators; existing HC arithmetic test backend, not GPU proof.
#include "portable_native_hc_test_backend.hpp"
#include "portable_qsa_fixture.hpp"
#include "native_hc_boundary.hpp"
namespace {
std::vector<float> projected;size_t projections=0;
ggml_status inspect(ggml_backend_t b,ggml_cgraph*g){
 auto result=original_compute(b,g);
 for(int n=0;n<ggml_graph_n_nodes(g);++n){auto*t=ggml_graph_node(g,n);
  if(t->op==GGML_OP_MUL_MAT&&t->src[0]&&std::string(ggml_get_name(t->src[0]))=="blk.3.attn_output.weight"){
   check(t->flags&GGML_TENSOR_FLAG_OUTPUT,"post-projection producer not retained");projected=get(t);++projections;
  }}return result;
}
std::vector<float> apply(const PortableHcPendingWrite&t){
 std::vector<float> out(10240);for(size_t c=0;c<4;++c){float gate=std::fma(2.f,1.f/(1.f+std::exp(-std::fma(.25f,t.injection[c],0.f))),0.f);for(size_t d=0;d<2560;++d)out[c*2560+d]=std::fma(t.block[d],gate,t.residual[c*2560+d]);}return out;
}
}
int main(){try{
 for(int l=0;l<48;++l){auto b=native_hc_boundary(l);check(b.pending_attention==(l>1),"native PLE/first read predicate");check(b.defer_ffn==(l>0&&l<47),"native PLE/head write predicate");check(!native_hc_boundary(l,48,true,false,true).defer_ffn&&!native_hc_boundary(l,48,true,false,false,true).defer_ffn,"control/split materialization lost");check(!native_hc_boundary(l,48,true,true).pending_attention,"previous control must clear pending");}
 check(native_hc_boundary(1,48,false).pending_attention&&native_hc_boundary(0,48,false).defer_ffn,"PLE-disabled native rule");
 Fixture f;qsa_fixture::Fixture qf;std::string error;
 // Match HC role bytes across both halves/families, leaving real QSA projection
 // data nonzero so gated and post-output-projection values cannot be confused.
 auto rebind=[&](std::vector<PortableTensorView>&views){for(auto&v:views)if(v.name.find("hc_")!=std::string::npos){for(const auto&src:f.views)if(src.name.find("hc_attn_")!=std::string::npos && v.name.substr(v.name.find("_",v.name.find("hc_")+3)+1)==src.name.substr(src.name.find("_",src.name.find("hc_")+3)+1)){v.data=src.data;v.bytes=src.bytes;}}};
 rebind(qf.views);auto gv=f.views;rebind(gv);
 for(auto&v:f.views){if(v.name=="blk.0.hc_attn_inject.weight")expected_weights=native_hc_q8_bf16((const uint8_t*)v.data,v.bytes,40960);if(v.name=="blk.0.hc_attn_down.weight")expected_down=native_hc_q8_bf16((const uint8_t*)v.data,v.bytes,3276800);if(v.name=="blk.0.hc_attn_up.weight")expected_up=native_hc_q8_bf16((const uint8_t*)v.data,v.bytes,3276800);}
 auto root=std::filesystem::temp_directory_path()/("hc-family-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));check(std::filesystem::create_directory(root),"fresh root");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>cpu(ggml_backend_cpu_init(),ggml_backend_free);original_compute=cpu->iface.graph_compute;cpu->iface.graph_compute=inspect;
 PortableFfnOptions exec;exec.threads=1;exec.accelerator=cpu.get();exec.request_accelerator=true;exec.resident_dense_weights=false;
 auto renamed=[&](int l){auto v=gv;for(auto&t:v)t.name.replace(0,6,"blk."+std::to_string(l)+".");return v;};
 auto scratch=PortableGdnScratch::create(exec,error);check(bool(scratch),error);
 PortableGdnOptions go;go.execution=exec;go.scratch=scratch;go.resident_ffn_weights=true;go.resident_attention_weights=true;go.native_hc_norm_inject=go.native_hc_mix=go.native_hc_ffn_read=true;go.defer_ffn_hc_write=true;go.native_hc_receipt_prefix=(root/"gdn2").string();
 auto g2=PortableGdnLayer::create(2,renamed(2),go,error);check(bool(g2),error);check(g2->reset(error),error);
 auto end=go;end.resident_ffn_weights=false;end.native_hc_pending=true;end.native_hc_ffn_read=false;end.defer_ffn_hc_write=false;end.native_hc_receipt_prefix=(root/"gdn4").string();auto g4=PortableGdnLayer::create(4,renamed(4),end,error);check(bool(g4),error);check(g4->reset(error),error);
 auto ec=end;ec.native_hc_pending=false;ec.native_hc_receipt_prefix=(root/"gdn4-control").string();auto control=PortableGdnLayer::create(4,renamed(4),ec,error);check(bool(control),error);check(control->reset(error),error);
 PortableQsaOptions qo;qo.execution=exec;qo.resident_ffn_weights=true;qo.capacity=8;qo.reuse_attention=true;qo.attention=PortableQsaAttention::create(exec,error);check(bool(qo.attention),error);qo.native_hc_read=qo.native_hc_pending=qo.native_hc_ffn_read=qo.defer_ffn_hc_write=true;qo.native_hc_receipt_prefix=(root/"qsa3").string();auto q3=PortableQsaLayer::create(3,qf.views,qo,error);check(bool(q3),error);check(q3->reset(error),error);
 check(q3->memory_usage().host_weights>0&&!qo.execution.resident_dense_weights,"FFN residency moved unrelated QSA weights");
 check(q3->memory_usage().host_extra_bound>=PortableQsaOptions::native_hc_host_bound+PortableFfnOptions::native_hc_pending_host_bound+PortableFfnOptions::deferred_hc_host_bound,"family bounds missing");auto short_weight=qo;short_weight.execution.weight_budget=q3->weight_bytes()-1;check(!PortableQsaLayer::create(3,qf.views,short_weight,error),"short weight admission");
 auto short_graph=qo;short_graph.execution.graph_budget=q3->graph_bytes()-1;const auto prior_owners=owners,prior_mix=mix_owners,prior_pending=pending_owners;check(!PortableQsaLayer::create(3,qf.views,short_graph,error),"short graph admission");check(owners==prior_owners&&mix_owners==prior_mix&&pending_owners==prior_pending,"refused setup leaked HC owners");
 auto dispatch=[](int64_t,const PortableRoute&r,std::vector<float>&rows,std::string&e){return experts()(0,r,rows,e);};PortableRoute route;std::vector<float> first;PortableHcPendingWrite last;
 for(size_t pos=0;pos<4;++pos){check(g2->step(f.input(pos),dispatch,route,error),error);auto p=g2->pending_write();PortableQsaTrace t;check(q3->step_pending(pos,p,dispatch,route,error,&t),error);
 check(t.projected_output==projected&&projected.size()==2560&&t.gated_attention.size()==6144,"wrong QSA projected producer");check(t.ffn_hc.previous_output==projected&&t.ffn_hc.previous_injection==t.hc.inject&&t.ffn_hc.original==t.hc.residual,"actual attention-to-FFN tuple/lifetime");check(t.ffn_hc.residual==apply({t.hc.residual,projected,t.hc.inject}),"independent half1 FMA order");
 auto tuple=q3->pending_write();check(tuple.residual==t.ffn_hc.residual&&tuple.injection==t.ffn_hc.inject,"FFN outgoing tuple not actual owner");last=tuple;
 check(g4->step_pending(tuple.residual,tuple.block,tuple.injection,dispatch,route,error),error);check(control->step(apply(tuple),dispatch,route,error),error);check(g4->last_residual()==control->last_residual()&&g4->recurrent_state()==control->recurrent_state()&&g4->convolution_history()==control->convolution_history(),"QSA to next GDN materialized control");if(!pos)first=g4->last_residual();}
 // The same GDN implementation composes incoming pending plus half1 read;
 // it is not restricted to the initial materialized input of this fixture.
 auto pg=go;pg.native_hc_pending=true;pg.native_hc_receipt_prefix=(root/"gdn-composed").string();auto pm=go;pm.native_hc_receipt_prefix=(root/"gdn-materialized").string();auto pgo=PortableGdnLayer::create(2,renamed(2),pg,error);auto pmo=PortableGdnLayer::create(2,renamed(2),pm,error);check(bool(pgo)&&bool(pmo),error);check(pgo->reset(error)&&pmo->reset(error),error);PortableGdnTrace pt;check(pgo->step_pending(last.residual,last.block,last.injection,dispatch,route,error,&pt),error);check(pmo->step(apply(last),dispatch,route,error),error);check(pt.ffn_hc.original==pt.hc_residual&&pgo->last_residual()==pmo->last_residual()&&pgo->recurrent_state()==pmo->recurrent_state(),"GDN incoming pending plus half1 native read failed composition");
 auto keys=q3->keys(),values=q3->values();auto pool=q3->pooled_keys(),tail=q3->indexer_tail(),res=q3->last_residual();auto committed=q3->pending_write();auto count=q3->deferred_hc_outputs();auto fail=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional provider refusal";return false;};
 check(!q3->step_pending(4,g2->pending_write(),fail,route,error),"provider accepted");check(q3->tokens()==4&&q3->keys()==keys&&q3->values()==values&&q3->pooled_keys()==pool&&q3->indexer_tail()==tail&&q3->last_residual()==res&&q3->pending_write().block==committed.block&&q3->deferred_hc_outputs()==count,"failed family candidate committed");
 check(q3->step_pending(4,g2->pending_write(),dispatch,route,error),error);check(q3->deferred_hc_outputs()==count+1,"retry missing actual producer");
 check(g2->reset(error)&&q3->reset(error)&&g4->reset(error),error);check(q3->pending_write().empty()&&q3->keys().empty()&&q3->indexer_tail()==std::vector<float>(512,0),"QSA reset retained candidate");check(g2->step(f.input(0),dispatch,route,error),error);check(q3->step_pending(0,g2->pending_write(),dispatch,route,error),error);last=q3->pending_write();check(g4->step_pending(last.residual,last.block,last.injection,dispatch,route,error),error);check(g4->last_residual()==first,"family reset/replay");
 auto invalid=qo;invalid.native_hc_ffn_read=false;invalid.native_hc_ffn_write=true;check(!PortableQsaLayer::create(3,qf.views,invalid,error),"missing half1 method accepted");auto finalviews=qf.views;for(auto&v:finalviews)v.name.replace(0,6,"blk.47.");check(!PortableQsaLayer::create(47,finalviews,qo,error),"head deferred incorrectly");auto gv0=renamed(0);check(!PortableGdnLayer::create(0,gv0,go,error),"pre-PLE defer accepted");check(!q3->prepare_chunk(2,error),"family chunk bypass");
 PortableModelOptions mo;mo.native_hc_serial_cycle=mo.hc_cycle_resident_weights=mo.native_hc_pending_layer1_2=mo.native_hc_pending_layer2_3=mo.gdn_resident_weights=true;mo.native_hc_pending_receipt_prefix=(root/"entry").string();mo.native_hc_qsa_receipt_prefix=(root/"qsa").string();mo.native_hc_cycle_receipt_prefix=(root/"cycle").string();validate_model_modes(mo,1);
 bool carry=false;for(size_t l=0;l<48;++l){check(model_pending_consumer(mo,l)==carry,"model cycle stale/missing carry");carry=model_deferred_producer(mo,l);if(l%4==3){PortableQsaOptions q;model_qsa_hc_options(mo,l,q);check(q.resident_ffn_weights==(l==3)&&q.native_hc_ffn_read==(l==3)&&q.defer_ffn_hc_write==(l==3),"QSA product cycle scope");}else{PortableGdnOptions g;model_gdn_hc_options(mo,l,g);check(g.resident_ffn_weights==(l==2)&&g.native_hc_ffn_read==(l==2),"GDN product cycle scope");check(g.defer_ffn_hc_write==(l==1||l==2),"GDN outgoing product scope");}}
 check(!carry&&!model_resident_dense(mo,3,false),"model final/head carry or unrelated residency");auto off=mo;off.native_hc_serial_cycle=false;PortableGdnOptions off2;PortableQsaOptions off3;model_gdn_hc_options(off,2,off2);model_qsa_hc_options(off,3,off3);check(off2.resident_ffn_weights&&off3.resident_ffn_weights&&!off2.native_hc_ffn_read&&!off3.native_hc_ffn_read&&!off3.defer_ffn_hc_write,"matched control residency changed numerical policy");check(model_hc_host_bound(mo,2)-model_hc_host_bound(off,2)==PortableFfnOptions::native_hc_pending_host_bound&&model_hc_host_bound(mo,3)-model_hc_host_bound(off,3)==PortableFfnOptions::native_hc_pending_host_bound+PortableFfnOptions::deferred_hc_host_bound&&model_hc_host_bound(mo,4)==PortableGdnOptions::native_hc_pending_host_bound,"model host envelope incomplete");

 // Full ordinary scope selects families by native geometry, never a copied
 // numbered-layer selector. PLE layer1 remains in its original graph.
 auto ordinary=mo;ordinary.native_hc_serial_cycle=false;ordinary.native_hc_serial=true;
 ordinary.hc_serial_resident_weights=true;ordinary.native_hc_ffn_write_layer0=true;
 ordinary.native_hc_ffn_read_layer0=ordinary.native_hc_mix_layer0=ordinary.native_hc_norm_inject_layer0=true;
 ordinary.layer0_resident_weights=true;ordinary.dense_topology=PortableDenseTopology::primary;
 ordinary.native_hc_receipt_prefix=(root/"zero").string();ordinary.native_hc_serial_receipt_prefix=(root/"ordinary").string();
 validate_model_modes(ordinary,1);carry=false;
 size_t selected=0;
 for(size_t l=0;l<48;++l){check(model_pending_consumer(ordinary,l)==carry,"ordinary pending chain");carry=model_deferred_producer(ordinary,l);
  if(l%4==3){PortableQsaOptions q;model_qsa_hc_options(ordinary,l,q);check(q.native_hc_read&&q.native_hc_ffn_read&&q.resident_ffn_weights,"ordinary QSA scope");check(q.defer_ffn_hc_write==(l<47)&&q.native_hc_ffn_write==(l==47),"ordinary QSA final head boundary");}
  else {PortableGdnOptions g;model_gdn_hc_options(ordinary,l,g);check(g.native_hc_ffn_read==(l!=1),"ordinary GDN PLE exception");check(g.native_hc_pending==(l>=2),"ordinary GDN pending predicate");check(g.defer_ffn_hc_write==(l>0),"ordinary GDN post predicate");}
  if(l>=2){++selected;check(model_ffn_resident(ordinary,l),"ordinary bank admission selector");check(model_hc_host_bound(ordinary,l)>=PortableFfnOptions::native_hc_pending_host_bound,"ordinary host charge");}
 }
 check(selected==46&&!carry,"ordinary head must materialize");
 auto matched=ordinary;matched.native_hc_serial=false;matched.native_hc_serial_cycle=true;
 for(size_t l=2;l<48;++l)check(model_ffn_resident(matched,l),"matched control residency missing");
 // Actual QSA final endpoint executes the same qualified write and refuses
 // outgoing carry, including rollback and reset, rather than deferring to head.
 auto fq=qo;fq.defer_ffn_hc_write=false;fq.native_hc_ffn_write=true;fq.native_hc_receipt_prefix=(root/"final47").string();
 auto final=PortableQsaLayer::create(47,finalviews,fq,error);check(bool(final),error);check(final->reset(error),error);
 check(final->step_pending(0,last,dispatch,route,error),error);auto finalres=final->last_residual();check(final->pending_write().empty()&&final->native_ffn_hc_write_calls()==1,"head materialization/counter");
 check(!final->step_pending(1,last,fail,route,error)&&final->last_residual()==finalres&&final->tokens()==1,"final endpoint rollback");
 check(final->reset(error)&&final->step_pending(0,last,dispatch,route,error)&&final->last_residual()==finalres,"final endpoint reset");

 // Failure-only diagnostics read the completed failing call, not initial success.
 auto diagnostic=go;diagnostic.native_hc_ffn_read=false;diagnostic.defer_ffn_hc_write=false;diagnostic.native_hc_receipt_prefix=(root/"diagnostic38").string();
 auto d38=PortableGdnLayer::create(38,renamed(38),diagnostic,error);check(bool(d38)&&d38->reset(error),error);
 check(d38->step(f.input(0),dispatch,route,error)&&d38->step(f.input(1),dispatch,route,error),error);
 auto committed_state=d38->recurrent_state(),committed_history=d38->convolution_history(),committed_residual=d38->last_residual();
 fail_mix_domain=true;check(!d38->step(f.input(2),dispatch,route,error)&&error.find("status=2048")!=std::string::npos,"domain failure accepted/lost status");fail_mix_domain=false;
 check(d38->tokens()==2&&d38->recurrent_state()==committed_state&&d38->convolution_history()==committed_history&&d38->last_residual()==committed_residual,"diagnostic refusal committed state");
 std::filesystem::path diagnostic_owner;for(const auto&e:std::filesystem::directory_iterator(root))if(e.path().filename().string().starts_with("diagnostic38.owner-"))diagnostic_owner=e.path();
 auto read_words=[](const std::filesystem::path&p){std::ifstream f(p,std::ios::binary);check(bool(f),"diagnostic file absent");std::vector<float> v(std::filesystem::file_size(p)/4);f.read((char*)v.data(),v.size()*4);check(bool(f),"diagnostic short read");return v;};
 check(read_words(diagnostic_owner/"refusal/normalized.f32le")==expected_xn,"refusal captured stale normalized input");
 check(read_words(diagnostic_owner/"first-normalized.f32le")!=expected_xn,"diagnostic test did not distinguish first successful input");
 std::ifstream metadata(diagnostic_owner/"refusal/metadata.json");std::string text((std::istreambuf_iterator<char>(metadata)),{});
 check(text.find("\"norm_call\":3")!=std::string::npos&&text.find("\"position\":2")!=std::string::npos&&text.find("\"layer\":38")!=std::string::npos&&text.find("\"mix_successes\":2")!=std::string::npos&&text.find("[0,8,0]")!=std::string::npos,"exact refusal identity/stage/current-call metadata");
 size_t diagnostic_bytes=0;for(const auto&e:std::filesystem::directory_iterator(diagnostic_owner/"refusal"))if(e.path().extension()==".f32le")diagnostic_bytes+=e.file_size();check(diagnostic_bytes==177920,"bounded diagnostic payload");
 check(d38->step(f.input(2),dispatch,route,error)&&d38->tokens()==3,"diagnostic rollback retry");
 diagnostic.native_hc_receipt_prefix=(root/"generic37").string();auto d37=PortableGdnLayer::create(37,renamed(37),diagnostic,error);check(bool(d37)&&d37->reset(error),error);fail_mix=true;check(!d37->step(f.input(0),dispatch,route,error),"generic failure accepted");fail_mix=false;
 for(const auto&e:std::filesystem::directory_iterator(root))if(e.path().filename().string().starts_with("generic37.owner-"))check(!std::filesystem::exists(e.path()/"refusal"),"pre-dispatch failure exposed stale/uninitialized outputs");

 // The actual GDN->FFN call forwards logical position independently of FFN reset.
 auto fd=diagnostic;fd.native_hc_ffn_read=true;fd.native_hc_receipt_prefix=(root/"diagnostic-ffn38").string();auto f38=PortableGdnLayer::create(38,renamed(38),fd,error);check(bool(f38)&&f38->reset(error),error);
 check(f38->step(f.input(0),dispatch,route,error)&&f38->step(f.input(1),dispatch,route,error),error);auto fs=f38->recurrent_state(),fr=f38->last_residual();fail_mix_domain_ffn=true;check(!f38->step(f.input(2),dispatch,route,error),"FFN domain refusal accepted");fail_mix_domain_ffn=false;
 check(f38->tokens()==2&&f38->recurrent_state()==fs&&f38->last_residual()==fr,"FFN diagnostic refusal committed GDN state");
 for(const auto&e:std::filesystem::directory_iterator(root))if(e.path().filename().string().starts_with("diagnostic-ffn38.ffn.owner-")){
  std::ifstream stream(e.path()/"refusal/metadata.json");std::string m((std::istreambuf_iterator<char>(stream)),{});check(m.find("blk.38.hc_ffn_norm.weight")!=std::string::npos&&m.find("\"position\":2")!=std::string::npos,"actual FFN logical position/role lost");check(read_words(e.path()/"refusal/normalized.f32le")==expected_xn,"FFN failure captured previous call");
 }
 check(f38->step(f.input(2),dispatch,route,error),error);

 // Non-mix failures save only current immutable inputs, never stale outputs.
 auto find_input_packet=[&](const std::string&prefix,const std::string&component){std::filesystem::path out;for(const auto&e:std::filesystem::directory_iterator(root))if(e.path().filename().string().starts_with(prefix+".owner-"))out=e.path()/"input-refusal";std::ifstream stream(out/"metadata.json");std::string m((std::istreambuf_iterator<char>(stream)),{});check(m.find("\"component\":\""+component+"\"")!=std::string::npos&&m.find("\"outputs_captured\":false")!=std::string::npos,"input-only diagnostic stage/completion");return out;};
 check(read_words(find_input_packet("generic37","mix")/"normalized.f32le").size()==10240,"generic mix current input missing");
 auto nd=diagnostic;nd.native_hc_receipt_prefix=(root/"norm36").string();auto n36=PortableGdnLayer::create(36,renamed(36),nd,error);check(bool(n36)&&n36->reset(error),error);fail_run=true;check(!n36->step(f.input(0),dispatch,route,error),"norm refusal accepted");fail_run=false;check(n36->tokens()==0&&read_words(find_input_packet("norm36","norm")/"input.f32le")==f.input(0),"norm failure input/rollback");
 auto pd=diagnostic;pd.native_hc_pending=true;pd.native_hc_receipt_prefix=(root/"pending34").string();auto p34=PortableGdnLayer::create(34,renamed(34),pd,error);check(bool(p34)&&p34->reset(error),error);fail_pending_run=true;check(!p34->step_pending(last.residual,last.block,last.injection,dispatch,route,error),"pending refusal accepted");fail_pending_run=false;auto pi=find_input_packet("pending34","pending");check(p34->tokens()==0&&read_words(pi/"input.f32le")==last.residual&&read_words(pi/"block.f32le")==last.block&&read_words(pi/"inject.f32le")==last.injection,"pending immutable input/rollback");
 auto oldkeys=final->keys();fail_write_run=true;check(!final->step_pending(1,last,dispatch,route,error),"write refusal accepted");fail_write_run=false;check(final->tokens()==1&&final->keys()==oldkeys,"write input snapshot committed candidate");
 bool write_packet=false;for(const auto&e:std::filesystem::directory_iterator(root))if(e.path().filename().string().starts_with("final47.ffn.owner-")){std::ifstream file(e.path()/"input-refusal/metadata.json");std::string m((std::istreambuf_iterator<char>(file)),{});write_packet=m.find("\"component\":\"write\"")!=std::string::npos&&m.find("\"payload_bytes\":51216")!=std::string::npos;}check(write_packet,"write current inputs missing");
 check(projections>=6&&g2->native_ffn_hc_pending_calls()==5&&q3->native_ffn_hc_pending_calls()==7&&q3->deferred_hc_outputs()==6,"paid family work counters");cpu->iface.graph_compute=original_compute;std::filesystem::remove_all(root);puts("reusable GDN/QSA HC family cycle / post-projection tuple / independent FMA / shared scratch / admission / rollback reset / native boundary predicates pass; mocked HC only");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
