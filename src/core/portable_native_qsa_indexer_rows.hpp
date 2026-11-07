#pragma once
// Standalone QSA BF16 indexer projections. Backend and roles are borrowed.
#include "native_qsa_indexer_bf16.hpp"
#include "portable_native_ple_host_bound.hpp"
#include "strata_vk_native_shared.hpp"
#include "ggml-alloc.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>
namespace strata::core::portable_detail {
class PortableNativeQsaIndexerRows {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeShared,decltype(&strata_vk_native_shared_destroy)> dot_{nullptr,strata_vk_native_shared_destroy};
 std::array<ggml_tensor*,2> weights_{};ggml_tensor *input_[2]{},*output_[2][2]{},*aux_=nullptr;
 size_t bytes_=0,rows_=0;bool valid_=false;
 static constexpr std::array<size_t,2> sizes_{{128,512}};
 static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
 static Context context(){Context c(ggml_init({32768,nullptr,true}),ggml_free);need(bool(c),"QSA indexer rows: metadata");return c;}
public:
 // Excludes caller-owned role/input storage and backend internals.
 static size_t host_bound(){return 32768+512*2560*2+2*2560*4+2*512*8+512*4;}
 static size_t estimate(ggml_backend_t b){need(b,"QSA indexer rows: backend");auto c=context();ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2*2560);for(auto n:sizes_)ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2*n);ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,1);return ple_host_add(ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b)),3u<<20);}
 PortableNativeQsaIndexerRows(ggml_backend_t b,const std::array<ggml_tensor*,2>&weights,const std::array<const void*,2>&roles,const std::array<size_t,2>&role_sizes,const std::string&dir,size_t budget):weights_(weights){
  need(b&&estimate(b)<=budget,"QSA indexer rows: budget");size_t borrowed=0;ggml_backend_buffer_t seen[2]{};
  for(size_t i=0;i<2;++i){auto*w=weights[i];size_t n=sizes_[i]*2560*2;need(w&&w->type==GGML_TYPE_I32&&w->buffer&&ggml_is_contiguous(w)&&ggml_nbytes(w)==n&&roles[i]&&role_sizes[i]==n,"QSA indexer rows: role");
   std::vector<uint16_t> copy(n/2);ggml_backend_tensor_get(w,copy.data(),0,n);need(!std::memcmp(copy.data(),roles[i],n),"QSA indexer rows: role mismatch");qsa_indexer_bf16_weights(copy.data(),copy.size());
   bool duplicate=false;for(size_t j=0;j<i;++j)duplicate|=seen[j]==w->buffer;seen[i]=w->buffer;if(!duplicate)borrowed=ple_host_add(borrowed,ggml_backend_buffer_get_size(w->buffer));}
  need(std::filesystem::create_directory(dir),"QSA indexer rows: fresh receipt");ctx_=context();input_[1]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,2*2560);input_[0]=ggml_view_1d(ctx_.get(),input_[1],2560,0);
  for(size_t i=0;i<2;++i){output_[i][1]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,2*sizes_[i]);output_[i][0]=ggml_view_1d(ctx_.get(),output_[i][1],sizes_[i],0);ggml_set_output(output_[i][0]);ggml_set_output(output_[i][1]);}aux_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,1);
  size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(b));need(bank<=budget,"QSA indexer rows: bank budget");
  dot_.reset(strata_vk_qsa_indexer_bf16_rows_create(b,(std::filesystem::path(dir)/"effective.spv").string().c_str(),ple_host_add(budget-bank,borrowed)));
  buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),b));need(bool(buffer_),"QSA indexer rows: allocation");bytes_=ple_host_add(ggml_backend_buffer_get_size(buffer_.get()),strata_vk_native_shared_bytes(dot_.get()));need(bytes_<=budget&&bytes_<=estimate(b),"QSA indexer rows: actual budget");
 }
 PortableNativeQsaIndexerRows(const PortableNativeQsaIndexerRows&)=delete;PortableNativeQsaIndexerRows& operator=(const PortableNativeQsaIndexerRows&)=delete;
 size_t bytes()const{return bytes_;}bool valid()const{return valid_;}
 ggml_tensor* output(size_t role,size_t rows)const{need(valid_&&role<2&&rows==rows_,"QSA indexer rows: invalid output");return output_[role][rows-1];}
 void run(const std::vector<float>&input,size_t rows){valid_=false;need((rows==1||rows==2)&&input.size()==rows*2560,"QSA indexer rows: input span");
  ggml_backend_tensor_set(input_[rows-1],input.data(),0,input.size()*4);
  for(size_t i=0;i<2;++i){std::vector<uint16_t>w(sizes_[i]*2560);ggml_backend_tensor_get(weights_[i],w.data(),0,w.size()*2);std::vector<float> expected;expected.reserve(rows*sizes_[i]);
   for(size_t r=0;r<rows;++r){auto row=qsa_indexer_bf16_control(input.data()+r*2560,w.data(),sizes_[i]);expected.insert(expected.end(),row.begin(),row.end());}
   strata_vk_native_shared_run(dot_.get(),1,weights_[i],input_[rows-1],output_[i][rows-1],aux_,2560,uint32_t(sizes_[i]),uint32_t(rows));std::vector<float> actual(expected.size());ggml_backend_tensor_get(output_[i][rows-1],actual.data(),0,actual.size()*4);need(!std::memcmp(actual.data(),expected.data(),actual.size()*4),"QSA indexer rows: ordered control mismatch");}
  rows_=rows;valid_=true;
 }
};
}
