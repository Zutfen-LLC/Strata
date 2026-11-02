#pragma once
#include "native_qsa_indexer_bf16.hpp"
#include "strata_vk_native_shared.hpp"
#include "portable_native_qsa_norm_rope.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
namespace strata::core::portable_detail {
class PortableNativeQsaIndexerBf16 {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeShared,decltype(&strata_vk_native_shared_destroy)> component_{nullptr,strata_vk_native_shared_destroy};
 std::array<ggml_tensor*,2> weights_{},outputs_{};ggml_tensor*aux_=nullptr;size_t bytes_=0;
 std::filesystem::path directory_;bool record_;std::array<bool,8> recorded_{};
 static constexpr std::array<size_t,2> rows_{{128,512}};
 static Context context(){Context c(ggml_init({32768,nullptr,true}),ggml_free);native_shared_need(bool(c),"QSA indexer BF16: metadata allocation");return c;}
 static void leaves(ggml_context*c,std::array<ggml_tensor*,2>&o,ggml_tensor*&a){for(size_t i=0;i<2;++i)o[i]=ggml_new_tensor_1d(c,GGML_TYPE_F32,rows_[i]);a=ggml_new_tensor_1d(c,GGML_TYPE_F32,1);}
 static size_t metadata(ggml_backend_t b){auto c=context();ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,128);return ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b));}
 static void save(const std::filesystem::path&p,const void*data,size_t n){std::ofstream o(p,std::ios::binary);o.write(static_cast<const char*>(data),n);native_shared_need(bool(o),"QSA indexer BF16: receipt write");}
public:
 struct Candidate{std::array<std::vector<float>,2> values;};
 static size_t estimate(ggml_backend_t b){auto c=context();std::array<ggml_tensor*,2>o{};ggml_tensor*a=nullptr;leaves(c.get(),o,a);size_t n=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b));size_t m=metadata(b);native_shared_need(n<=SIZE_MAX-m,"QSA indexer BF16: allocation overflow");return n+m;}
 PortableNativeQsaIndexerBf16(ggml_backend_t b,const std::array<ggml_tensor*,2>& weights,ggml_tensor*mixed,const std::array<const void*,2>& raw,const std::array<size_t,2>& sizes,const std::string&dir,size_t budget,bool record):weights_(weights),directory_(dir),record_(record){
  native_shared_need(b && mixed && mixed->buffer && !dir.empty() && estimate(b)<=budget,"QSA indexer BF16: complete aligned admission");
  native_shared_need(std::filesystem::create_directory(directory_),"QSA indexer BF16: fresh receipt directory");
  size_t borrowed=0;std::array<ggml_backend_buffer_t,3> seen{{mixed->buffer,nullptr,nullptr}};borrowed=ggml_backend_buffer_get_size(mixed->buffer);
  for(size_t i=0;i<2;++i){auto*w=weights_[i];size_t n=rows_[i]*2560*2;native_shared_need(w && w->type==GGML_TYPE_I32 && w->buffer && ggml_is_contiguous(w) && ggml_nbytes(w)==n && raw[i] && sizes[i]==n,"QSA indexer BF16: raw role span");
   std::vector<uint16_t> check(n/2);ggml_backend_tensor_get(w,check.data(),0,n);native_shared_need(std::memcmp(check.data(),raw[i],n)==0,"QSA indexer BF16: resident role differs");qsa_indexer_bf16_weights(check.data(),check.size());
   bool duplicate=false;for(size_t j=0;j<=i;++j)duplicate|=seen[j]==w->buffer;seen[i+1]=w->buffer;if(!duplicate){size_t amount=ggml_backend_buffer_get_size(w->buffer);native_shared_need(amount<=SIZE_MAX-borrowed,"QSA indexer BF16: borrowed buffer overflow");borrowed+=amount;}}
  ctx_=context();leaves(ctx_.get(),outputs_,aux_);size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(b));
  native_shared_need(bank<budget && borrowed<=SIZE_MAX-(budget-bank),"QSA indexer BF16: owner budget");
  component_.reset(strata_vk_qsa_indexer_bf16_create(b,(directory_/"effective.spv").c_str(),budget-bank+borrowed));
  buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),b));native_shared_need(bool(buffer_),"QSA indexer BF16: output allocation");
  bytes_=ggml_backend_buffer_get_size(buffer_.get())+strata_vk_native_shared_bytes(component_.get());native_shared_need(bytes_<=estimate(b)&&bytes_<=budget,"QSA indexer BF16: allocation exceeds admission");
 }
 size_t bytes()const{return bytes_;}
 uint64_t calls(unsigned stage)const{switch(stage){case 0:return strata_vk_native_shared_submitted(component_.get());case 1:return strata_vk_native_shared_completed(component_.get());case 2:return strata_vk_native_shared_uncertain(component_.get());default:return 0;}}
 Candidate run(ggml_tensor*mixed){native_shared_need(mixed && ggml_nbytes(mixed)==2560*4,"QSA indexer BF16: mixed input span");std::vector<float> input(2560);ggml_backend_tensor_get(mixed,input.data(),0,input.size()*4);Candidate result;
  for(size_t i=0;i<2;++i){std::vector<uint16_t>weights(rows_[i]*2560);ggml_backend_tensor_get(weights_[i],weights.data(),0,weights.size()*2);auto expected=qsa_indexer_bf16_control(input.data(),weights.data(),rows_[i]);
   strata_vk_native_shared_run(component_.get(),1,weights_[i],mixed,outputs_[i],aux_,2560,uint32_t(rows_[i]),1);result.values[i].resize(rows_[i]);ggml_backend_tensor_get(outputs_[i],result.values[i].data(),0,rows_[i]*4);
   native_shared_need(std::memcmp(result.values[i].data(),expected.data(),rows_[i]*4)==0,"QSA indexer BF16: device output differs from ordered host control");}
  return result;
 }
 void consumed(uint64_t position,ggml_tensor*mixed,const std::array<ggml_tensor*,2>&leaves,const Candidate&result){
  for(size_t i=0;i<2;++i){std::vector<float>actual(rows_[i]);ggml_backend_tensor_get(leaves[i],actual.data(),0,rows_[i]*4);native_shared_need(std::memcmp(actual.data(),result.values[i].data(),rows_[i]*4)==0,"QSA indexer BF16: downstream input differs");}
  if(record_&&position<8&&!recorded_[position]){auto dir=directory_/("first-"+std::to_string(position));native_shared_need(std::filesystem::create_directory(dir),"QSA indexer BF16: receipt exists");std::vector<float>input(2560);ggml_backend_tensor_get(mixed,input.data(),0,input.size()*4);save(dir/"mixed.f32le",input.data(),input.size()*4);save(dir/"key.f32le",result.values[0].data(),128*4);save(dir/"query.f32le",result.values[1].data(),512*4);std::ofstream(dir/"role.json")<<"{\"type\":\"BF16\",\"input\":2560,\"rows\":[128,512],\"one_mixed_input\":true,\"qualified\":false}\n";recorded_[position]=true;}
 }
};
}
