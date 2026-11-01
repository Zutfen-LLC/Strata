#pragma once
#include "portable_native_qsa_indexer.hpp"
#include "native_qsa_select_layout.hpp"
#include "strata_vk_native_qsa_select.hpp"
namespace strata::core::portable_detail {
// A detached score/ID candidate. Session/indexer authority stays with the
// existing FFN-success transaction; this owner never publishes persistent state.
class PortableNativeQsaSelect {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeQsaSelect,decltype(&strata_vk_native_qsa_select_destroy)> kernel_{nullptr,strata_vk_native_qsa_select_destroy};
 ggml_tensor *pool_=nullptr,*dead_=nullptr,*step_=nullptr,*scores_=nullptr,*ids_=nullptr;
 uint32_t cells_;size_t bytes_=0;std::filesystem::path dir_;bool record_;std::array<bool,8>recorded_{};uint64_t attempts_=0;
 static Context context(){Context c(ggml_init({32768,nullptr,true}),ggml_free);native_hc_need(bool(c),"native QSA selection metadata");return c;}
 static void leaves(ggml_context*c,uint32_t cells,ggml_tensor*&pool,ggml_tensor*&dead,ggml_tensor*&step,ggml_tensor*&score,ggml_tensor*&ids){pool=ggml_new_tensor_1d(c,GGML_TYPE_F32,size_t(qsa_select_blocks(cells))*128);dead=ggml_new_tensor_1d(c,GGML_TYPE_F32,128);step=ggml_new_tensor_1d(c,GGML_TYPE_I32,4);score=ggml_new_tensor_1d(c,GGML_TYPE_F32,qsa_select_blocks(cells));ids=ggml_new_tensor_1d(c,GGML_TYPE_I32,qsa_select_cap);}
 static void word(std::ostream&o,uint32_t v){for(unsigned j=0;j<4;++j)o.put(char(v>>(j*8)));}
 static void tensor(std::ostream&o,ggml_tensor*t){std::vector<uint32_t>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);for(auto x:v)word(o,x);native_hc_need(bool(o),"native QSA selection receipt write");}
 void capture(ggml_tensor*query,uint64_t position,const char*prefix,bool completed,uint32_t status){
  auto d=dir_/(std::string(prefix)+std::to_string(position)+(std::string(prefix)=="failure-"?"-attempt-"+std::to_string(attempts_):""));native_hc_need(std::filesystem::create_directory(d),"native QSA selection fresh receipt");
  std::ofstream f(d/"fixture.bin",std::ios::binary);for(uint32_t v:{0x3153514eu,1u,cells_,1u,qsa_select_cap,uint32_t(position),uint32_t(position+1),uint32_t((position+1)/4),uint32_t(std::min<uint64_t>(position+1,qsa_select_cap))})word(f,v);tensor(f,pool_);tensor(f,dead_);tensor(f,query);for(uint32_t i=0;i<qsa_select_blocks(cells_);++i)word(f,0);f.close();native_hc_need(bool(f),"native QSA selection fixture close");
  if(completed){std::ofstream s(d/"scores.f32le",std::ios::binary);tensor(s,scores_);std::ofstream ids(d/"selected.i32le",std::ios::binary);tensor(ids,ids_);}
  std::ofstream(d/"status.json")<<"{\"position\":"<<position<<",\"status\":"<<status<<",\"completed_outputs\":"<<(completed?"true":"false")<<",\"qualified\":false}\n";
 }
public:
 struct Candidate{std::vector<float>cell_scores,mask;};
 static size_t estimate(ggml_backend_t b,uint32_t cells){native_hc_need(cells>=1&&cells<=256,"native QSA selection eager capacity");auto c=context();ggml_tensor *p,*d,*t,*s,*i;leaves(c.get(),cells,p,d,t,s,i);size_t n=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b));auto m=context();ggml_new_tensor_1d(m.get(),GGML_TYPE_I32,128);ggml_new_tensor_1d(m.get(),GGML_TYPE_I32,128);size_t meta=ggml_backend_alloc_ctx_tensors_from_buft_size(m.get(),ggml_backend_get_default_buffer_type(b));native_hc_need(n<=SIZE_MAX-meta,"native QSA selection estimate overflow");return n+meta;}
 PortableNativeQsaSelect(ggml_backend_t b,uint32_t cells,const std::string&directory,size_t budget,bool record):cells_(cells),dir_(directory),record_(record){
  native_hc_need(b&&!directory.empty()&&estimate(b,cells)<=budget,"native QSA selection aligned admission");native_hc_need(std::filesystem::create_directory(dir_),"native QSA selection fresh namespace");ctx_=context();leaves(ctx_.get(),cells,pool_,dead_,step_,scores_,ids_);size_t n=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(b));native_hc_need(n<budget,"native QSA selection metadata room");kernel_.reset(strata_vk_native_qsa_select_create(b,(dir_/"scores-effective.spv").c_str(),(dir_/"select-effective.spv").c_str(),budget-n));buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),b));native_hc_need(bool(buffer_),"native QSA selection banks");bytes_=ggml_backend_buffer_get_size(buffer_.get())+strata_vk_native_qsa_select_bytes(kernel_.get());native_hc_need(bytes_<=estimate(b,cells)&&bytes_<=budget,"native QSA selection actual allocation exceeds bound");
 }
 size_t bytes()const{return bytes_;}uint64_t score_calls()const{return strata_vk_native_qsa_score_calls(kernel_.get());}uint64_t select_calls()const{return strata_vk_native_qsa_select_calls(kernel_.get());}
 Candidate run(ggml_tensor*query,const std::vector<uint32_t>&state,uint64_t position){
  ++attempts_;native_hc_need(position<cells_&&state.size()==qsa_idx_words(cells_),"native QSA selection requires actual complete indexer candidate");std::vector<uint32_t>pool(size_t(qsa_select_blocks(cells_))*128,0);std::copy(state.begin()+qsa_idx_pool,state.end()-1,pool.begin());ggml_backend_tensor_set(pool_,pool.data(),0,pool.size()*4);ggml_backend_tensor_set(dead_,state.data()+384,0,128*4);std::array<int32_t,4>step{{int32_t(position),int32_t(position+1),int32_t((position+1)/4),int32_t(std::min<uint64_t>(position+1,qsa_select_cap))}};ggml_backend_tensor_set(step_,step.data(),0,16);ggml_backend_tensor_memset(scores_,0,0,ggml_nbytes(scores_));std::vector<uint32_t>inactive(qsa_select_cap,0x4acaf00d);ggml_backend_tensor_set(ids_,inactive.data(),0,inactive.size()*4);
  uint32_t status=UINT32_MAX;bool completed=false;
  try{std::array<ggml_tensor*,5>s{{pool_,dead_,query,step_,scores_}};status=strata_vk_native_qsa_scores_run(kernel_.get(),s.data(),cells_);native_hc_need(status==0,"native QSA scores refused");std::array<ggml_tensor*,3>t{{scores_,step_,ids_}};status=strata_vk_native_qsa_select_run(kernel_.get(),t.data(),cells_);native_hc_need(status==0,"native QSA selection refused");completed=true;}
  catch(...){try{capture(query,position,"failure-",completed,status);}catch(const std::exception&e){std::fprintf(stderr,"native QSA selection receipt: %s\n",e.what());}throw;}
  std::vector<float>scores(qsa_select_blocks(cells_));std::vector<int32_t>ids(qsa_select_cap);ggml_backend_tensor_get(scores_,scores.data(),0,scores.size()*4);ggml_backend_tensor_get(ids_,ids.data(),0,ids.size()*4);Candidate out;out.cell_scores.resize(position+1);out.mask.assign(position+1,0);for(size_t i=0;i<out.cell_scores.size();++i)out.cell_scores[i]=scores[i/4];for(int32_t i=0;i<step[3];++i){native_hc_need(ids[i]>=0&&ids[i]<=int32_t(position)&&(i==0||ids[i]>ids[i-1]),"native QSA invalid selected IDs");out.mask[ids[i]]=1;}
  if(record_&&position<8&&!recorded_[position]){capture(query,position,"first-",true,0);recorded_[position]=true;}return out;
 }
 void consumed(uint64_t position,const Candidate&candidate,const std::vector<float>&scores,const std::vector<float>&mask){native_hc_need(scores==candidate.cell_scores&&mask==candidate.mask,"native QSA actual attention consumer ignored score/ID candidate");if(record_&&position<8){auto d=dir_/("first-"+std::to_string(position));for(auto entry:{std::make_pair("consumed-scores.f32le",&scores),std::make_pair("consumed-mask.f32le",&mask)}){std::ofstream f(d/entry.first,std::ios::binary);for(float v:*entry.second){uint32_t w;std::memcpy(&w,&v,4);word(f,w);}native_hc_need(bool(f),"native QSA consumer receipt");}}}
};
}
