#pragma once
// Standalone native GDN output row projection; backend and weight are borrowed.
#include "native_qsa_wo_domain.hpp"
#include "portable_native_ple_host_bound.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
#include "ggml-alloc.h"
#include <cstring>
#include <filesystem>
#include <memory>
namespace strata::core::portable_detail {
class PortableNativeGdnOutputRows {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> quant_{nullptr,strata_vk_native_stages_destroy};
 std::unique_ptr<StrataVkQ8Wo,decltype(&strata_vk_q8_wo_destroy)> dot_{nullptr,strata_vk_q8_wo_destroy};
 ggml_tensor *input_[2]{},*packed_[2]{},*output_[2]{},*weight_=nullptr;size_t bytes_=0,rows_=0;bool valid_=false;
 static Context context(){Context c(ggml_init({32768,nullptr,true}),ggml_free);if(!c)throw std::runtime_error("GDN output rows: metadata");return c;}
 static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
public:
 // Construction peaks at either the role-comparison buffer or the two
 // simultaneous metadata contexts in the final estimate, not both. Runtime
 // retains one context while packed rows and output readback are live.
 // Requested payload only; excludes caller inputs, roles, allocator overhead,
 // factory internals and backend/driver allocations. Not an RSS bound.
 static constexpr size_t construction_host_bound(){return 65536;}
 static constexpr size_t runtime_host_bound(){return 32768+2*6912+2*2560*sizeof(float);}
 static constexpr size_t host_bound(){return runtime_host_bound()>construction_host_bound()?runtime_host_bound():construction_host_bound();}
 static size_t estimate(ggml_backend_t b){need(b,"GDN output rows: backend");auto c=context();ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2*6144);ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,2*6912/4);ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2*2560);return ple_host_add(ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b)),3u<<20);}
 PortableNativeGdnOutputRows(ggml_backend_t b,ggml_tensor*w,const void*role,size_t role_size,const std::string&directory,size_t budget){
  need(b&&w&&w->type==GGML_TYPE_Q8_0&&w->ne[0]==6144&&w->ne[1]==2560&&w->buffer&&ggml_is_contiguous(w)&&ggml_nbytes(w)==16711680&&estimate(b)<=budget,"GDN output rows: role or budget");
  qsa_wo_weight_domain(role,role_size,"GDN output");
  {std::vector<uint8_t>check(65536);for(size_t at=0;at<role_size;at+=check.size()){size_t n=std::min(check.size(),role_size-at);ggml_backend_tensor_get(w,check.data(),at,n);need(!std::memcmp(check.data(),static_cast<const uint8_t*>(role)+at,n),"GDN output rows: role mismatch");}}
  need(std::filesystem::create_directory(directory),"GDN output rows: fresh receipt");ctx_=context();weight_=ggml_view_1d(ctx_.get(),w,6144*2560,0);weight_->type=GGML_TYPE_I32;weight_->ne[0]=role_size/4;weight_->nb[0]=4;weight_->nb[1]=weight_->nb[2]=weight_->nb[3]=role_size;need(ggml_backend_view_init(weight_)==GGML_STATUS_SUCCESS,"GDN output rows: weight view");
  input_[1]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,2*6144);packed_[1]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_I32,2*6912/4);output_[1]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,2*2560);
  input_[0]=ggml_view_1d(ctx_.get(),input_[1],6144,0);packed_[0]=ggml_view_1d(ctx_.get(),packed_[1],6912/4,0);output_[0]=ggml_view_1d(ctx_.get(),output_[1],2560,0);
  const size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(b));need(bank<=budget,"GDN output rows: bank budget");
  quant_.reset(strata_vk_native_stages_create_quant_rcp_observation(b,(std::filesystem::path(directory)/"quant-effective.spv").string().c_str(),budget-bank));size_t qb=strata_vk_native_stages_bytes(quant_.get());need(qb<=budget-bank,"GDN output rows: quant budget");
  dot_.reset(strata_vk_gdn_output_rows_create(b,(std::filesystem::path(directory)/"dot-effective.spv").string().c_str(),ple_host_add(budget-bank-qb,ggml_backend_buffer_get_size(w->buffer))));
  buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),b));need(bool(buffer_),"GDN output rows: allocation");bytes_=ple_host_add(ple_host_add(ggml_backend_buffer_get_size(buffer_.get()),qb),strata_vk_q8_wo_bytes(dot_.get()));need(bytes_<=budget&&bytes_<=estimate(b),"GDN output rows: actual budget");
 }
 PortableNativeGdnOutputRows(const PortableNativeGdnOutputRows&)=delete;PortableNativeGdnOutputRows&operator=(const PortableNativeGdnOutputRows&)=delete;
 size_t bytes()const{return bytes_;}bool valid()const{return valid_;}
 ggml_tensor* output(size_t rows)const{need(valid_&&rows==rows_,"GDN output rows: invalid output");return output_[rows-1];}
 void run(const std::vector<float>& input,size_t rows){valid_=false;need((rows==1||rows==2)&&input.size()==rows*6144,"GDN output rows: input span");qsa_q8_input_domain(input,rows*192,"GDN output rows");
  ggml_backend_tensor_set(input_[rows-1],input.data(),0,input.size()*4);strata_vk_gdn_output_quantize_bounded(quant_.get(),input_[rows-1],packed_[rows-1],rows);
  std::vector<uint8_t>packed(rows*6912);ggml_backend_tensor_get(packed_[rows-1],packed.data(),0,packed.size());for(size_t r=0;r<rows;++r)qsa_gdn_output_packed_domain(packed.data()+r*6912,6912);
  strata_vk_gdn_output_run_bounded(dot_.get(),weight_,packed_[rows-1],output_[rows-1],rows);std::vector<float>values(rows*2560);ggml_backend_tensor_get(output_[rows-1],values.data(),0,values.size()*4);for(float v:values)need(qsa_wo_normal_or_zero(v),"GDN output rows: output domain");rows_=rows;valid_=true;
 }
};
}
