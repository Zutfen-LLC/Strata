#pragma once
#include "portable_native_qsa_norm_rope.hpp"
#include "native_qsa_indexer_layout.hpp"
#include "strata_vk_native_qsa_indexer.hpp"
namespace strata::core::portable_detail {
// Eager session host authority, like existing QSA. The component receives a
// disjoint before/candidate bank; only successful FFN may publish its candidate.
class PortableNativeQsaIndexer {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeQsaIndexer,decltype(&strata_vk_native_qsa_indexer_destroy)> component_{nullptr,strata_vk_native_qsa_indexer_destroy};
 ggml_tensor *before_=nullptr,*candidate_=nullptr,*diag_=nullptr,*gamma_=nullptr;
 uint32_t cells_;std::vector<uint32_t>state_;std::array<bool,8>recorded_{};
 std::filesystem::path directory_;size_t bytes_=0;uint64_t attempts_=0;bool record_;
 static Context context(){Context c(ggml_init({32768,nullptr,true}),ggml_free);native_hc_need(bool(c),"native QSA indexer metadata");return c;}
 static void leaves(ggml_context*c,uint32_t cells,ggml_tensor*&b,ggml_tensor*&out,ggml_tensor*&diag){b=ggml_new_tensor_1d(c,GGML_TYPE_I32,qsa_idx_words(cells));out=ggml_new_tensor_1d(c,GGML_TYPE_I32,qsa_idx_words(cells));diag=ggml_new_tensor_1d(c,GGML_TYPE_F32,386);}
 static void word(std::ostream&o,uint32_t v){for(int s=0;s<32;s+=8)o.put(char(v>>s));}
 static void tensor(std::ostream&o,ggml_tensor*t){std::vector<uint32_t>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);for(auto x:v)word(o,x);native_hc_need(bool(o),"native QSA indexer receipt write");}
 void capture(ggml_tensor*raw,uint64_t position,const char*prefix,bool completed,uint32_t status){
  auto dir=directory_/(std::string(prefix)+std::to_string(position)+(std::string(prefix)=="failure-"?"-call-"+std::to_string(calls())+"-attempt-"+std::to_string(attempts_):""));native_hc_need(std::filesystem::create_directory(dir),"native QSA indexer fresh receipt");
  std::ofstream f(dir/"fixture.bin",std::ios::binary);for(uint32_t x:{0x3149514eu,1u,cells_,0u,uint32_t(position),1u,native_qsa_theta_scale()})word(f,x);tensor(f,raw);tensor(f,gamma_);tensor(f,before_);f.close();native_hc_need(bool(f),"native QSA indexer fixture close");
  if(completed){std::ofstream o(dir/"state.u32le",std::ios::binary);tensor(o,candidate_);std::ofstream d(dir/"diag.f32le",std::ios::binary);tensor(d,diag_);}
  std::ofstream(dir/"status.json")<<"{\"position\":"<<position<<",\"status\":"<<status<<",\"completed_outputs\":"<<(completed?"true":"false")<<",\"qualified\":false}\n";
 }
public:
 static size_t estimate(ggml_backend_t backend,uint32_t cells){native_qsa_indexer_geometry(cells,0,0,native_qsa_theta_scale());auto c=context();ggml_tensor *b,*o,*d;leaves(c.get(),cells,b,o,d);size_t bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(backend));auto meta=context();ggml_new_tensor_1d(meta.get(),GGML_TYPE_I32,128);size_t extra=ggml_backend_alloc_ctx_tensors_from_buft_size(meta.get(),ggml_backend_get_default_buffer_type(backend));native_hc_need(bytes<=SIZE_MAX-extra,"native QSA indexer estimate overflow");return bytes+extra;}
 PortableNativeQsaIndexer(ggml_backend_t backend,ggml_tensor*gamma,uint32_t cells,const std::string&directory,size_t budget,bool record):gamma_(gamma),cells_(cells),state_(qsa_idx_words(cells),0),directory_(directory),record_(record){
  native_hc_need(backend&&gamma&&!directory.empty()&&estimate(backend,cells)<=budget,"native QSA indexer complete aligned admission");native_hc_need(std::filesystem::create_directory(directory_),"native QSA indexer fresh namespace");ctx_=context();leaves(ctx_.get(),cells,before_,candidate_,diag_);size_t b=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));native_hc_need(b<budget,"native QSA indexer metadata room");component_.reset(strata_vk_native_qsa_indexer_create(backend,(directory_/"effective.spv").c_str(),budget-b));buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));native_hc_need(bool(buffer_),"native QSA indexer banks");bytes_=ggml_backend_buffer_get_size(buffer_.get())+strata_vk_native_qsa_indexer_bytes(component_.get());native_hc_need(bytes_<=estimate(backend,cells)&&bytes_<=budget,"native QSA indexer actual allocation exceeds bound");
 }
 uint32_t capacity()const{return cells_;}
 size_t bytes()const{return bytes_;}uint64_t calls()const{return strata_vk_native_qsa_indexer_calls(component_.get());}
 const std::vector<uint32_t>& state()const{return state_;}
 void reset(){std::fill(state_.begin(),state_.end(),0);}
 // Detached append: the caller owns the before image and may chain private
 // candidates without publishing intermediate rows. This object and its scratch
 // tensors are not reentrant; the returned vector owns its contents.
 std::vector<uint32_t> run_before(ggml_tensor*raw,uint64_t position,const std::vector<uint32_t>&before){++attempts_;native_hc_need(position<cells_,"native QSA indexer position exceeds capacity");native_hc_need(before.size()==state_.size(),"native QSA indexer complete before image");native_qsa_indexer_geometry(cells_,int32_t(position),0,native_qsa_theta_scale());ggml_backend_tensor_set(before_,before.data(),0,before.size()*4);std::array<ggml_tensor*,5>t{{raw,gamma_,before_,candidate_,diag_}};const auto paid=calls();uint32_t status=UINT32_MAX;
  try{status=strata_vk_native_qsa_indexer_run(component_.get(),t.data(),cells_,position,0,native_qsa_theta_scale());native_hc_need(status==0,"native QSA indexer actual numerical domain refused");}
  catch(...){try{capture(raw,position,"failure-",calls()!=paid&&status==0,status);}catch(const std::exception&e){std::fprintf(stderr,"native QSA indexer receipt: %s\n",e.what());}throw;}
  if(record_&&position<8&&!recorded_[position]){capture(raw,position,"first-",true,0);recorded_[position]=true;}
  std::vector<uint32_t>out(state_.size());ggml_backend_tensor_get(candidate_,out.data(),0,out.size()*4);return out;
 }
 std::vector<uint32_t> run(ggml_tensor*raw,uint64_t position){return run_before(raw,position,state_);}
 void commit(std::vector<uint32_t>&candidate,uint64_t position){native_hc_need(candidate.size()==state_.size(),"native QSA indexer complete commit candidate");if(record_&&position<8){std::string bytes(candidate.size()*4,'\0');for(size_t i=0;i<candidate.size();++i)for(unsigned j=0;j<4;++j)bytes[i*4+j]=char(candidate[i]>>(8*j));auto file=directory_/("first-"+std::to_string(position))/"accepted-state.u32le";
   if(std::filesystem::exists(file)){std::ifstream f(file,std::ios::binary);std::string old((std::istreambuf_iterator<char>(f)),{});native_hc_need(old==bytes,"native QSA indexer reset committed state differs");}else{std::ofstream f(file,std::ios::binary);f.write(bytes.data(),bytes.size());native_hc_need(bool(f),"native QSA indexer accepted state receipt");}}
  state_.swap(candidate);
 }
};
}
