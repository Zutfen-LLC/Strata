#pragma once
#include <array>
#include <filesystem>
#include <fstream>
#include <vector>
#include "portable_native_hc.hpp"
#include "native_qsa_norm_rope_layout.hpp"
#include "strata_vk_native_qsa_norm_rope.hpp"
namespace strata::core::portable_detail {
// Serial candidate bank. Inputs are the actual completed projections; no KV or
// indexer state is written here. Scratch is reused only after each fenced call.
class PortableNativeQsaNormRope {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free}; Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeQsaNormRope,decltype(&strata_vk_native_qsa_norm_rope_destroy)> component_{nullptr,strata_vk_native_qsa_norm_rope_destroy};
 std::array<ggml_tensor*,3> norm_{},diag_{},pos_{},gamma_{};
 size_t bytes_=0;std::filesystem::path directory_;bool record_,recorded_=false;
 static constexpr std::array<uint32_t,3> widths{{256,256,128}},rows{{24,2,4}},strides{{512,256,128}};
 static Context context(){Context c(ggml_init({16384,nullptr,true}),ggml_free);native_hc_need(bool(c),"native QSA norm/RoPE metadata");return c;}
 static void leaves(ggml_context*c,std::array<ggml_tensor*,3>&out,std::array<ggml_tensor*,3>&norm,std::array<ggml_tensor*,3>&diag,std::array<ggml_tensor*,3>&pos){
  auto*n=ggml_new_tensor_1d(c,GGML_TYPE_F32,6144);auto*d=ggml_new_tensor_1d(c,GGML_TYPE_F32,3120);auto*p=ggml_new_tensor_1d(c,GGML_TYPE_I32,24);
  for(size_t i=0;i<3;++i){out[i]=ggml_new_tensor_1d(c,GGML_TYPE_F32,widths[i]*rows[i]);norm[i]=ggml_view_1d(c,n,widths[i]*rows[i],0);diag[i]=ggml_view_1d(c,d,rows[i]*130,0);pos[i]=ggml_view_1d(c,p,rows[i],0);ggml_set_input(out[i]);}
 }
 static void word(std::ostream&o,uint32_t v){for(int s=0;s<32;s+=8)o.put(char(v>>s));}
 static void tensor(std::ostream&o,ggml_tensor*t){std::vector<uint32_t>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);for(auto x:v)word(o,x);native_hc_need(bool(o),"native QSA norm/RoPE receipt write");}
 void capture(size_t i,ggml_tensor*x,uint64_t position,bool outputs,const char*prefix,uint32_t status){
  auto dir=directory_/(std::string(prefix)+std::array<const char*,3>{{"q","k","iq"}}[i]+"-"+std::to_string(position));native_hc_need(std::filesystem::create_directory(dir),"native QSA norm/RoPE preserve receipt");
  std::ofstream f(dir/"fixture.bin",std::ios::binary);for(uint32_t v:{0x3152514eu,1u,widths[i],rows[i],strides[i],native_qsa_theta_scale()})word(f,v);tensor(f,x);tensor(f,gamma_[i]);tensor(f,pos_[i]);f.close();native_hc_need(bool(f),"native QSA fixture close");
  if(outputs){for(auto pair:{std::pair<const char*,ggml_tensor*>{"normalized.f32le",norm_[i]},{"rotated.f32le",output[i]},{"diag.f32le",diag_[i]}}){std::ofstream o(dir/pair.first,std::ios::binary);tensor(o,pair.second);}}
  std::ofstream meta(dir/"status.json");meta<<"{\"position\":"<<position<<",\"status\":"<<status<<",\"current_outputs\":"<<(outputs?"true":"false")<<",\"qualified\":false}\n";
 }
public:
 std::array<ggml_tensor*,3> output{};
 static size_t estimate(ggml_backend_t backend){auto c=context();std::array<ggml_tensor*,3>o{},n{},d{},p{};leaves(c.get(),o,n,d,p);size_t b=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(backend));auto m=context();ggml_new_tensor_1d(m.get(),GGML_TYPE_I32,128);size_t meta=ggml_backend_alloc_ctx_tensors_from_buft_size(m.get(),ggml_backend_get_default_buffer_type(backend));native_hc_need(b<=SIZE_MAX-meta,"native QSA norm/RoPE estimate overflow");return b+meta;}
 PortableNativeQsaNormRope(ggml_backend_t backend,std::array<ggml_tensor*,3> gamma,const std::string&directory,size_t budget,bool record):gamma_(gamma),directory_(directory),record_(record){
  native_hc_need(backend&&!directory.empty(),"native QSA norm/RoPE backend/receipt");native_hc_need(std::filesystem::create_directory(directory_),"native QSA norm/RoPE fresh namespace");ctx_=context();leaves(ctx_.get(),output,norm_,diag_,pos_);size_t b=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));native_hc_need(estimate(backend)<=budget&&b<budget,"native QSA norm/RoPE complete aligned admission");component_.reset(strata_vk_native_qsa_norm_rope_create(backend,(directory_/"effective.spv").c_str(),budget-b));buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));native_hc_need(bool(buffer_),"native QSA norm/RoPE candidate allocation");bytes_=ggml_backend_buffer_get_size(buffer_.get())+strata_vk_native_qsa_norm_rope_bytes(component_.get());native_hc_need(bytes_<=estimate(backend)&&bytes_<=budget,"native QSA norm/RoPE actual allocation exceeds bound");
 }
 void record_commit(const std::vector<uint16_t>&key,uint64_t position){
  if(!record_||position!=0)return;native_hc_need(key.size()==512,"native QSA accepted key geometry");auto file=directory_/"accepted-key.f16le";std::string raw;raw.reserve(key.size()*2);for(uint16_t x:key){raw.push_back(char(x));raw.push_back(char(x>>8));}
  if(std::filesystem::exists(file)){std::ifstream i(file,std::ios::binary);std::string prior((std::istreambuf_iterator<char>(i)),{});native_hc_need(prior==raw,"native QSA reset accepted key differs");return;}
  std::ofstream o(file,std::ios::binary);o.write(raw.data(),raw.size());native_hc_need(bool(o),"native QSA accepted key receipt write");
 }
 size_t bytes()const{return bytes_;}uint64_t calls()const{return strata_vk_native_qsa_norm_rope_calls(component_.get());}
 void run(std::array<ggml_tensor*,3> raw,uint64_t position){native_hc_need(position<4096,"native QSA norm/RoPE text context");
  for(size_t i=0;i<3;++i){std::vector<int32_t>p(rows[i],int32_t(position));ggml_backend_tensor_set(pos_[i],p.data(),0,p.size()*4);std::array<ggml_tensor*,6>t{{raw[i],gamma_[i],pos_[i],norm_[i],output[i],diag_[i]}};auto before=calls();uint32_t status=0;
   try{status=strata_vk_native_qsa_norm_rope_run(component_.get(),t.data(),widths[i],rows[i],strides[i],native_qsa_theta_scale());native_hc_need(status==0,"native QSA norm/RoPE actual domain refused");}
   catch(...){try{capture(i,raw[i],position,calls()!=before&&status==0,"failure-",status);}catch(const std::exception&e){std::fprintf(stderr,"native QSA receipt: %s\n",e.what());}throw;}
   if(record_&&!recorded_)capture(i,raw[i],position,true,"first-",0);
  }recorded_=true;
 }
};
}
