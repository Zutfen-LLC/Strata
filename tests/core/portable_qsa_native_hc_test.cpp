// Actual GDN producer/QSA consumer, shared attention arena, mocked HC device only.
#include "portable_native_hc_test_backend.hpp"
#include "portable_qsa_fixture.hpp"
#include "portable_native_qsa_indexer_bf16.hpp"
namespace {
bool inspect_qsa=false,inspect_native_selection=false;size_t qsa_consumers=0,selection_consumers=0;bool inspect_native_attention=false;size_t output_consumers=0;
ggml_status qsa_compute(ggml_backend_t b,ggml_cgraph*g){
 if(inspect_native_selection)for(int n=0;n<ggml_graph_n_nodes(g);++n){auto*t=ggml_graph_node(g,n);if(t->op==GGML_OP_LOG&&t->src[0]&&std::string(ggml_get_name(t->src[0]))=="qsa-selected-mask"){check(t->src[0]->op==GGML_OP_NONE,"attention recomputed a portable selection instead of consuming native IDs");++selection_consumers;}}
 if(inspect_native_attention)for(int n=0;n<ggml_graph_n_nodes(g);++n){auto*t=ggml_graph_node(g,n);if(t->op==GGML_OP_MUL_MAT&&t->src[0]&&std::string(ggml_get_name(t->src[0]))=="blk.3.attn_output.weight"){check(t->src[1]->op==GGML_OP_NONE,"output projection recomputed gate");check(get(t->src[1])==actual_native_gated,"actual projection ignored native attention gated output");++output_consumers;}}
 if(inspect_qsa)for(int n=0;n<ggml_graph_n_nodes(g);++n){auto*t=ggml_graph_node(g,n);if(t->op!=GGML_OP_MUL_MAT||!t->src[0])continue;const std::string name=ggml_get_name(t->src[0]);
 if(name=="blk.3.attn_q.weight"||name=="blk.3.attn_k.weight"||name=="blk.3.attn_v.weight"||name=="blk.3.indexer.q_proj.weight"||name=="blk.3.indexer.k_proj.weight"){
 auto*root=t->src[1];while(root->view_src)root=root->view_src;check(root==live_mixed,"QSA projection/indexer ignored actual native mixed leaf");++qsa_consumers;}}
 return original_compute(b,g);
}
}
int main(){try{
 qsa_fixture::Fixture qf;auto&f=qf.base;std::string error;auto root=std::filesystem::temp_directory_path()/("qsa-pending-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));check(std::filesystem::create_directory(root),"fresh root");
 for(auto&v:qf.views){if(v.name=="blk.3.hc_attn_inject.weight")expected_weights=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);if(v.name=="blk.3.hc_attn_down.weight")expected_down=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);if(v.name=="blk.3.hc_attn_up.weight")expected_up=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);}
 for(auto&v:qf.views){if(v.name=="blk.3.hc_ffn_inject.weight")expected_ffn_inject=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);if(v.name=="blk.3.hc_ffn_down.weight")expected_ffn_down=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);if(v.name=="blk.3.hc_ffn_up.weight")expected_ffn_up=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);}
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
 // Score/selection consumes the current native candidate, not committed state.
 auto ss=ix;ss.native_select_banks=true;ss.native_indexer_receipt_prefix=(root/"ss-control-indexer").string();ss.native_select_receipt_prefix=(root/"ss-control").string();
 auto score_control=PortableQsaLayer::create(3,qf.views,ss,error);check(bool(score_control),error);
 ss.native_select_t1=true;ss.native_select_record_first=true;ss.native_indexer_receipt_prefix=(root/"ss-on-indexer").string();ss.native_select_receipt_prefix=(root/"ss-on").string();
 auto score_on=PortableQsaLayer::create(3,qf.views,ss,error);check(bool(score_on),error);check(score_on->weight_bytes()==score_control->weight_bytes(),"score banks changed weight residency");check(score_on->graph_bytes()>=idx->graph_bytes(),"score aligned banks omitted");check(score_on->state_bytes()>=idx->state_bytes()+PortableQsaOptions::native_select_host_bound,"score host accounting missing");
 check(score_control->reset(error)&&score_on->reset(error),error);std::vector<float>ss_first;
 for(size_t pos=0;pos<4;++pos){check(score_control->step(pos,f.input(pos),nr_dispatch,nr_route,error),error);PortableQsaTrace tr;inspect_native_selection=true;check(score_on->step(pos,f.input(pos),nr_dispatch,nr_route,error,&tr),error);inspect_native_selection=false;check(score_on->native_score_calls()==pos+1&&score_on->native_select_calls()==pos+1&&score_control->native_score_calls()==0&&score_control->native_select_calls()==0,"score counters not actual work");check(tr.selected==std::vector<float>(pos+1,1),"native identity IDs not consumed by real attention");check(score_on->last_residual()==score_control->last_residual()&&score_on->keys()==score_control->keys()&&score_on->native_indexer_state()==score_control->native_indexer_state(),"short-context native score/identity changed attention/state");if(pos==0)ss_first=score_on->last_residual();
 auto dir=root/("ss-on/first-"+std::to_string(pos));native_qsa_ss::check(native_qsa_ss::read((dir/"fixture.bin").string()),dir);}
 check(selection_consumers==4,"actual native selection attention consumer missing");
 auto ss_state=score_on->native_indexer_state();auto ss_keys=score_on->keys();auto ss_last=score_on->last_residual();
 fail_qsa_score=true;check(!score_on->step(4,f.input(4),nr_dispatch,nr_route,error),"score refusal accepted");fail_qsa_score=false;check(score_on->native_score_calls()==5&&score_on->native_select_calls()==4&&score_on->native_indexer_state()==ss_state&&score_on->keys()==ss_keys&&score_on->tokens()==4,"score refusal committed candidate or hid work");
 fail_qsa_select=true;check(!score_on->step(4,f.input(4),nr_dispatch,nr_route,error),"selector refusal accepted");fail_qsa_select=false;check(score_on->native_score_calls()==6&&score_on->native_select_calls()==5&&score_on->native_indexer_state()==ss_state&&score_on->last_residual()==ss_last,"selection refusal committed state");
 check(!score_on->step(4,f.input(4),[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="injected FFN failure";return false;},nr_route,error),"post-selection FFN failure accepted");check(score_on->native_score_calls()==7&&score_on->native_select_calls()==6&&score_on->native_indexer_state()==ss_state&&score_on->keys()==ss_keys,"FFN failure published scored candidate");
 check(score_on->step(4,f.input(4),nr_dispatch,nr_route,error),error);check(score_on->reset(error)&&score_on->step(0,f.input(0),nr_dispatch,nr_route,error),error);check(score_on->last_residual()==ss_first,"score reset/retry changed output");
 // Actual attention/gate output feeds WO, with candidate-only state until FFN.
 auto at=ss;at.native_norm_rope_weights=at.native_norm_rope_t1=true;at.native_attention_banks=true;at.native_attention_record_first=true;
 auto fresh=[&](PortableQsaOptions&v,const std::string&name){v.native_norm_rope_receipt_prefix=(root/(name+"-norm")).string();v.native_indexer_receipt_prefix=(root/(name+"-indexer")).string();v.native_select_receipt_prefix=(root/(name+"-select")).string();v.native_attention_receipt_prefix=(root/name).string();};
 auto no_bank=at;no_bank.native_attention_banks=false;fresh(no_bank,"attn-no-bank");auto attn_no_bank=PortableQsaLayer::create(3,qf.views,no_bank,error);check(bool(attn_no_bank),error);check(attn_no_bank->reset(error),error);
 fresh(at,"attn-control");auto attn_control=PortableQsaLayer::create(3,qf.views,at,error);check(bool(attn_control),error);
 at.native_attention_t1=true;fresh(at,"attn-on");auto attn_on=PortableQsaLayer::create(3,qf.views,at,error);check(bool(attn_on),error);
 check(attn_control->weight_bytes()==attn_on->weight_bytes(),"attention bank changed resident weight policy");check(attn_on->state_bytes()>=score_on->state_bytes()+PortableQsaOptions::native_attention_host_bound && attn_on->memory_usage().host_extra_bound>=PortableQsaOptions::native_attention_host_bound,"attention host envelope omitted");
 check(attn_on->reset(error)&&attn_control->reset(error),error);std::vector<float>attn_first;
 for(size_t pos=0;pos<2;++pos){check(attn_control->step(pos,f.input(pos),nr_dispatch,nr_route,error),error);check(attn_no_bank->step(pos,f.input(pos),nr_dispatch,nr_route,error),error);check(attn_no_bank->last_residual()==attn_control->last_residual()&&attn_no_bank->keys()==attn_control->keys()&&attn_no_bank->native_indexer_state()==attn_control->native_indexer_state(),"matched attention banks changed disabled arithmetic/state");inspect_native_attention=true;PortableQsaTrace tr;check(attn_on->step(pos,f.input(pos),nr_dispatch,nr_route,error,&tr),error);inspect_native_attention=false;check(tr.gated_attention==actual_native_gated,"trace lost actual native gated candidate");for(unsigned stage=0;stage<3;++stage)check(attn_on->native_attention_calls(stage)==pos+1&&attn_control->native_attention_calls(stage)==0,"attention counters not actual stages");auto fi=native_qsa_attn::read((root/("attn-on/first-"+std::to_string(pos)+"/fixture.bin")).string());for(size_t cell=0;cell<=pos;++cell)for(size_t h=0;h<2;++h)for(size_t d=0;d<256;++d){size_t slot=((cell/4*2+h)*4+cell%4)*256+d;check(native_qsa_attn::half(fi.in[0],slot)==ggml_fp16_to_fp32(attn_on->keys()[cell*512+h*256+d]),"native paged key transport changed token/head order");check(native_qsa_attn::half(fi.in[1],slot)==ggml_fp16_to_fp32(attn_on->values()[cell*512+h*256+d]),"native paged value transport changed token/head order");}if(!pos)attn_first=attn_on->last_residual();}
 check(output_consumers==2,"actual WO consumer not inspected");auto ak=attn_on->keys();auto ai=attn_on->native_indexer_state();auto al=attn_on->last_residual();
 fail_qsa_attention=true;check(!attn_on->step(2,f.input(2),nr_dispatch,nr_route,error),"attention refusal accepted");fail_qsa_attention=false;check(attn_on->tokens()==2&&attn_on->keys()==ak&&attn_on->native_indexer_state()==ai&&attn_on->last_residual()==al,"attention failure committed candidate");check(attn_on->native_attention_calls(0)==3&&attn_on->native_attention_calls(1)==2,"attention failure hidden paid stages");check(!std::filesystem::exists(root/"attn-on/failure-2-attempt-3/gated.f32le"),"failed attention exported stale gated output");
 check(!attn_on->step(2,f.input(2),[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="FFN failure";return false;},nr_route,error),"post-attention FFN failure accepted");check(attn_on->keys()==ak&&attn_on->native_indexer_state()==ai&&attn_on->tokens()==2,"FFN failure committed attention candidate");check(attn_on->step(2,f.input(2),nr_dispatch,nr_route,error),error);check(attn_on->reset(error)&&attn_on->step(0,f.input(0),nr_dispatch,nr_route,error),error);check(attn_on->last_residual()==attn_first,"attention reset changed output/state");
 auto ab=at;ab.execution.graph_budget=1;fresh(ab,"attn-budget");check(!PortableQsaLayer::create(3,qf.views,ab,error),"attention budget refusal missing");ab=at;ab.native_select_t1=false;fresh(ab,"attn-no-ids");check(!PortableQsaLayer::create(3,qf.views,ab,error),"attention accepted missing native IDs");
 auto composed=at;composed.native_hc_read=true;composed.native_hc_ffn_read=true;composed.defer_ffn_hc_write=true;composed.resident_ffn_weights=true;composed.native_hc_receipt_prefix=(root/"attn-composed-hc").string();fresh(composed,"attn-composed");auto actual_composed=PortableQsaLayer::create(3,qf.views,composed,error);check(bool(actual_composed),error);check(actual_composed->reset(error),error);
 check(actual_composed->graph_bytes()>0&&actual_composed->native_attention_calls(0)==0,"actual composed on-arm allocation dispatched work or omitted graph storage");
 // The default-off WO path feeds the actual producer image directly to the
 // consumer; the device primitive is a host mock and is not GPU parity proof.
 auto wc=composed;wc.execution.resident_dense_weights=true;wc.native_hc_receipt_prefix=(root/"wo-control-hc").string();fresh(wc,"wo-control");
 auto wo_control=PortableQsaLayer::create(3,qf.views,wc,error);check(bool(wo_control),error);
 auto wn=wc;wn.native_wo_t1=true;wn.native_wo_record_first=true;wn.native_wo_receipt_prefix=(root/"wo-on").string();wn.native_hc_receipt_prefix=(root/"wo-on-hc").string();fresh(wn,"wo-on-attention");
 auto wo_on=PortableQsaLayer::create(3,qf.views,wn,error);check(bool(wo_on),error);
 check(wo_on->weight_bytes()==wo_control->weight_bytes(),"WO duplicated resident weights");
 check(wo_on->state_bytes()>=wo_control->state_bytes()+PortableQsaOptions::native_wo_host_bound,"WO host admission omitted");
 check(wo_on->graph_bytes()>wo_control->graph_bytes(),"WO output bank not accounted");
 check(wo_on->reset(error)&&wo_control->reset(error)&&actual_composed->reset(error),error);
 check(wo_control->step(0,f.input(0),nr_dispatch,nr_route,error)&&actual_composed->step(0,f.input(0),nr_dispatch,nr_route,error),error);
 check(wo_control->last_residual()==actual_composed->last_residual()&&wo_control->keys()==actual_composed->keys()&&wo_control->pending_write().block==actual_composed->pending_write().block,"disabled WO or matched residency changed bytes");
 check(wo_control->reset(error),error);PortableQsaTrace wt;observe_wo_pending=true;
 check(wo_on->step(0,f.input(0),nr_dispatch,nr_route,error,&wt),error);observe_wo_pending=false;
 check(wt.projected_output==actual_native_wo&&wo_pending_links>0,"native WO output did not reach FFN pending consumer");
 for(unsigned i=0;i<4;++i)check(wo_on->native_wo_calls(i)==1&&wo_control->native_wo_calls(i)==0,"native WO paid-work counters wrong");
 auto wk=wo_on->keys(),wv=wo_on->values();auto wp=wo_on->pooled_keys(),wl=wo_on->last_residual();auto first_wo=wl;
 fail_qsa_wo=true;check(!wo_on->step(1,f.input(1),nr_dispatch,nr_route,error),"WO failure accepted");fail_qsa_wo=false;
 check(wo_on->tokens()==1&&wo_on->keys()==wk&&wo_on->values()==wv&&wo_on->pooled_keys()==wp&&wo_on->last_residual()==wl,"WO failure committed state");
 check(wo_on->native_wo_calls(0)==2&&wo_on->native_wo_calls(1)==2&&wo_on->native_wo_calls(2)==1&&wo_on->native_wo_calls(3)==1&&wo_on->native_wo_calls(5)==1,"WO failure did not record uncertain submission");
 check(!wo_on->step(1,f.input(1),[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="FFN failure";return false;},nr_route,error),"post-WO FFN failure accepted");
 check(wo_on->tokens()==1&&wo_on->keys()==wk&&wo_on->values()==wv&&wo_on->pooled_keys()==wp&&wo_on->last_residual()==wl,"post-WO FFN failure committed state");
 check(wo_on->native_wo_calls(0)==3&&wo_on->native_wo_calls(1)==3&&wo_on->native_wo_calls(2)==2&&wo_on->native_wo_calls(3)==2&&wo_on->native_wo_calls(5)==1,"post-WO failure counters wrong");
 check(wo_on->step(1,f.input(1),nr_dispatch,nr_route,error),error);check(wo_on->reset(error)&&wo_on->step(0,f.input(0),nr_dispatch,nr_route,error),error);
 check(wo_on->last_residual()==first_wo,"WO reset/retry changed output");
 check(std::filesystem::file_size(root/"wo-on/first-0/qx.bin")==6912&&std::filesystem::file_size(root/"wo-on/first-0/projected.f32le")==10240,"WO receipt spans differ");
 auto bad_wo=wn;bad_wo.native_attention_t1=false;bad_wo.native_wo_receipt_prefix=(root/"wo-invalid").string();check(!PortableQsaLayer::create(3,qf.views,bad_wo,error),"WO accepted nonnative attention");
 // Q/K/V consume one actual HC mixed producer and one packed image. Only the
 // Vulkan primitives are mocked; the layer graph and transaction are real.
 auto qn=wn;qn.native_qkv_t1=true;qn.native_qkv_record_first=true;qn.native_qkv_receipt_prefix=(root/"qkv-on").string();qn.native_wo_receipt_prefix=(root/"qkv-wo").string();qn.native_hc_receipt_prefix=(root/"qkv-hc").string();fresh(qn,"qkv-attention");
 auto qkv_on=PortableQsaLayer::create(3,qf.views,qn,error);check(bool(qkv_on),error);check(qkv_on->reset(error),error);
 PortableQsaTrace qt;observe_qkv_norm=true;check(qkv_on->step(0,f.input(0),nr_dispatch,nr_route,error,&qt),error);observe_qkv_norm=false;check(qkv_norm_links==2,"QKV query/key not consumed by norm/RoPE");
 check(qkv_on->native_qkv_calls(0)==1&&qkv_on->native_qkv_calls(1)==1&&qkv_on->native_qkv_calls(2)==3&&qkv_on->native_qkv_calls(3)==3,"QKV work counters differ");
 check(qt.value==actual_qkv[2]&&qt.gate.size()==6144,"actual native value or gate lost");
 for(size_t h=0;h<24;++h)for(size_t d=0;d<256;++d)check(qt.gate[h*256+d]==actual_qkv[0][h*512+256+d],"interleaved gate row lost");
 auto qk=qkv_on->keys(),qv=qkv_on->values();auto qp=qkv_on->pooled_keys(),ql=qkv_on->last_residual();
 fail_qsa_qkv=true;check(!qkv_on->step(1,f.input(1),nr_dispatch,nr_route,error),"QKV refusal accepted");fail_qsa_qkv=false;
 check(qkv_on->tokens()==1&&qkv_on->keys()==qk&&qkv_on->values()==qv&&qkv_on->pooled_keys()==qp&&qkv_on->last_residual()==ql,"QKV failure committed state");
 check(qkv_on->native_qkv_calls(0)==2&&qkv_on->native_qkv_calls(1)==2&&qkv_on->native_qkv_calls(2)==3&&qkv_on->native_qkv_calls(5)==1,"QKV failure accounting differs");
 check(!qkv_on->step(1,f.input(1),[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="FFN failure";return false;},nr_route,error),"post-QKV failure accepted");
 check(qkv_on->tokens()==1&&qkv_on->keys()==qk&&qkv_on->values()==qv&&qkv_on->pooled_keys()==qp&&qkv_on->last_residual()==ql,"post-QKV failure committed state");
 check(qkv_on->step(1,f.input(1),nr_dispatch,nr_route,error),error);check(qkv_on->reset(error)&&qkv_on->step(0,f.input(0),nr_dispatch,nr_route,error),error);
 check(qkv_on->last_residual()==ql&&std::filesystem::file_size(root/"qkv-on/first-0/qx.bin")==2880,"QKV reset/retry or packed receipt differs");
 auto bad_qkv=qn;bad_qkv.native_wo_t1=false;bad_qkv.native_qkv_receipt_prefix=(root/"qkv-invalid").string();check(!PortableQsaLayer::create(3,qf.views,bad_qkv,error),"QKV accepted missing native WO");
 // Raw BF16 indexer projections replace the exact-widened control storage.
 std::vector<uint16_t> bfkey(2560*128,0),bfquery(2560*512,0);bfkey[0]=0x3f80;bfkey.back()=0xbf80;bfquery[0]=0x3f80;bfquery.back()=0x3f00;
 // Exercise owner admission through the same aggregate dispatch accounting.
 {using Owner=strata::core::portable_detail::PortableNativeQsaIndexerBf16;
  std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({32768,nullptr,true}),ggml_free);
  auto* k=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,bfkey.size()/2);auto* q=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,bfquery.size()/2);auto* m=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2560);
  std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> b(ggml_backend_alloc_ctx_tensors(c.get(),cpu.get()),ggml_backend_buffer_free);
  check(bool(b),"BF16 direct owner bank");ggml_backend_tensor_set(k,bfkey.data(),0,bfkey.size()*2);ggml_backend_tensor_set(q,bfquery.data(),0,bfquery.size()*2);
  std::vector<float> zeros(2560,0.f);ggml_backend_tensor_set(m,zeros.data(),0,zeros.size()*4);auto* old_mixed=live_mixed;live_mixed=m;
  const std::array<ggml_tensor*,2> ws{{k,q}};const std::array<const void*,2> raw{{bfkey.data(),bfquery.data()}};const std::array<size_t,2> sizes{{bfkey.size()*2,bfquery.size()*2}};
  const size_t exact=Owner::estimate(cpu.get());Owner owner(cpu.get(),ws,m,raw,sizes,(root/"bf-exact-budget").string(),exact,false);
  owner.run(m);check(owner.calls(1)==2,"exact BF16 owner budget did not dispatch both roles");
  bool short_rejected=false;try{Owner short_owner(cpu.get(),ws,m,raw,sizes,(root/"bf-short-budget").string(),exact-1,false);}catch(const std::invalid_argument&){short_rejected=true;}
  check(short_rejected,"one-byte-short BF16 owner budget accepted");live_mixed=old_mixed;
 }
 auto bfviews=qf.views;for(auto&v:bfviews){if(v.name=="blk.3.indexer.k_proj.weight"){v.type=30;v.data=bfkey.data();v.bytes=bfkey.size()*2;}if(v.name=="blk.3.indexer.q_proj.weight"){v.type=30;v.data=bfquery.data();v.bytes=bfquery.size()*2;}}
 auto bfc=composed;bfc.execution.resident_dense_weights=true;bfc.native_hc_receipt_prefix=(root/"bf-control-hc").string();fresh(bfc,"bf-control");
 auto bfcontrol=PortableQsaLayer::create(3,bfviews,bfc,error);check(bool(bfcontrol),error);
 auto bfo=bfc;bfo.native_indexer_projection_t1=true;bfo.native_indexer_projection_record_first=true;bfo.native_indexer_projection_receipt_prefix=(root/"bf-projection").string();bfo.native_hc_receipt_prefix=(root/"bf-on-hc").string();fresh(bfo,"bf-on");
 auto bf=PortableQsaLayer::create(3,bfviews,bfo,error);check(bool(bf),error);
 check(bfcontrol->weight_bytes()==bf->weight_bytes()+3276800,"BF16 raw bank duplicated or widened");
 check(bf->state_bytes()>=bfcontrol->state_bytes()+PortableQsaOptions::native_indexer_projection_host_bound&&bf->graph_bytes()>bfcontrol->graph_bytes(),"BF16 graph or host scratch not admitted");
 check(bf->reset(error),error);observe_indexer_projection=true;check(bf->step(0,f.input(0),nr_dispatch,nr_route,error),error);observe_indexer_projection=false;
 check(indexer_projection_links==2&&bf->native_indexer_projection_calls(0)==2&&bf->native_indexer_projection_calls(1)==2,"BF16 producers or consumers differ");
 auto bfkeys=bf->keys();auto bfstate=bf->native_indexer_state();auto bfres=bf->last_residual();
 for(int fail:{1,2}){fail_indexer_projection=fail;check(!bf->step(1,f.input(1),nr_dispatch,nr_route,error),"BF16 projection failure accepted");fail_indexer_projection=0;check(bf->tokens()==1&&bf->keys()==bfkeys&&bf->native_indexer_state()==bfstate&&bf->last_residual()==bfres,"BF16 projection failure committed state");}
 check(bf->native_indexer_projection_calls(2)==2,"BF16 uncertain submissions lost");
 check(!bf->step(1,f.input(1),[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="FFN failure";return false;},nr_route,error),"post-BF16 FFN failure accepted");
 check(bf->tokens()==1&&bf->keys()==bfkeys&&bf->native_indexer_state()==bfstate,"post-BF16 failure committed state");
 check(bf->step(1,f.input(1),nr_dispatch,nr_route,error),error);check(bf->reset(error)&&bf->step(0,f.input(0),nr_dispatch,nr_route,error),error);check(bf->last_residual()==bfres,"BF16 reset/retry changed result");
 // The integrated flags must preserve all three norm/RoPE producer banks in
 // one forward; testing each flag alone misses an overwritten local bank list.
 auto combined=qn;combined.native_indexer_projection_t1=true;
 combined.native_indexer_projection_receipt_prefix=(root/"combined-bf").string();
 combined.native_qkv_receipt_prefix=(root/"combined-qkv").string();combined.native_wo_receipt_prefix=(root/"combined-wo").string();
 combined.native_hc_receipt_prefix=(root/"combined-hc").string();fresh(combined,"combined-attention");
 auto all=PortableQsaLayer::create(3,bfviews,combined,error);check(bool(all),error);check(all->reset(error),error);
 qkv_norm_links=0;indexer_projection_links=0;observe_qkv_norm=true;observe_indexer_projection=true;
 check(all->step(0,f.input(0),nr_dispatch,nr_route,error),error);observe_qkv_norm=false;observe_indexer_projection=false;
 check(qkv_norm_links==2&&indexer_projection_links==2,"combined QKV/indexer producer banks were replaced");
 auto all_keys=all->keys();auto all_state=all->native_indexer_state();auto all_residual=all->last_residual();
 for(int failure:{0,1,2}){
  fail_qsa_qkv=failure==0;fail_indexer_projection=failure;
  check(!all->step(1,f.input(1),nr_dispatch,nr_route,error),"combined projection refusal accepted");
  fail_qsa_qkv=false;fail_indexer_projection=0;
  check(all->tokens()==1&&all->keys()==all_keys&&all->native_indexer_state()==all_state&&all->last_residual()==all_residual,"combined projection failure committed state");
 }
 check(all->step(1,f.input(1),nr_dispatch,nr_route,error),error);
 check(all->reset(error)&&all->step(0,f.input(0),nr_dispatch,nr_route,error),error);
 check(all->last_residual()==all_residual&&all->keys()==all_keys&&all->native_indexer_state()==all_state,"combined producer reset/retry changed state");
 auto bf_bad=bfo;bf_bad.native_indexer_projection_receipt_prefix=(root/"bf-invalid").string();check(!PortableQsaLayer::create(3,qf.views,bf_bad,error),"BF16 projection accepted non-BF16 weights");
 auto sb=ss;sb.native_indexer_t1=false;sb.native_select_receipt_prefix=(root/"ss-invalid").string();check(!PortableQsaLayer::create(3,qf.views,sb,error),"scores accepted nonnative indexer state");
 sb=ss;sb.execution.graph_budget=1;sb.native_select_receipt_prefix=(root/"ss-budget").string();check(!PortableQsaLayer::create(3,qf.views,sb,error),"score banks bypassed graph budget");
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
