#pragma once
// Standalone QSA QKV row projections; no model chunk admission. Backend, weights
// and weight storage are borrowed and must outlive this owner and its runs.
#include "native_qsa_qkv_layout.hpp"
#include "native_qsa_wo_domain.hpp"
#include "portable_native_ple_host_bound.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
#include "ggml-alloc.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>

namespace strata::core::portable_detail {
class PortableNativeQsaQkvRows {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 static constexpr size_t width=2560,packed_bytes=2880;
 static constexpr std::array<size_t,3> outputs{{12288,512,512}};
 Context ctx_{nullptr,ggml_free}; Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> quant_{nullptr,strata_vk_native_stages_destroy};
 std::unique_ptr<StrataVkQ8Wo,decltype(&strata_vk_q8_wo_destroy)> dot_{nullptr,strata_vk_q8_wo_destroy};
 ggml_tensor *input_[2]{},*packed_[2]{},*weight_[3]{},*out_[3][2]{};
 size_t bytes_=0; bool valid_=false;
 static Context context(){Context c(ggml_init({32768,nullptr,true}),ggml_free);if(!c)throw std::runtime_error("QSA QKV rows: metadata");return c;}
 static void need(bool x,const char*message){if(!x)throw std::runtime_error(message);}
public:
 // Requested owner scratch payload: metadata plus packed T2 and the largest
 // simultaneous projection readback. Excludes caller input, role bytes and
 // backend/allocator internals; not a process-memory limit.
 static size_t host_bound(){return ple_host_add(32768,ple_host_add(2*packed_bytes,2*12288*sizeof(float)));}
 static size_t estimate(ggml_backend_t backend){
  need(backend,"QSA QKV rows: backend");auto c=context();
  ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2*width);ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,2*packed_bytes/4);
  for(auto n:outputs)ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2*n);
  return ple_host_add(ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(backend)),3u<<20);
 }
 PortableNativeQsaQkvRows(ggml_backend_t backend,const std::array<ggml_tensor*,3>& weights,
  const std::array<const void*,3>& roles,const std::array<size_t,3>& sizes,const std::string& directory,size_t budget){
  need(backend&&estimate(backend)<=budget,"QSA QKV rows: admission");
  size_t borrowed=0;ggml_backend_buffer_t seen[3]{};
  for(size_t i=0;i<3;++i){auto*w=weights[i];const size_t n=outputs[i]*qsa_qkv_row_bytes;
   need(w&&w->type==GGML_TYPE_Q8_0&&w->ne[0]==2560&&w->ne[1]==int64_t(outputs[i])&&w->buffer&&ggml_is_contiguous(w)&&ggml_nbytes(w)==n&&roles[i]&&sizes[i]==n,"QSA QKV rows: role");
   qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(roles[i]),outputs[i]*80,34),"QSA QKV weight",true);
   std::vector<uint8_t> check(65536);for(size_t at=0;at<n;at+=check.size()){auto count=std::min(check.size(),n-at);ggml_backend_tensor_get(w,check.data(),at,count);need(std::memcmp(check.data(),static_cast<const uint8_t*>(roles[i])+at,count)==0,"QSA QKV rows: role bytes differ");}
   bool duplicate=false;for(size_t j=0;j<i;++j)duplicate|=seen[j]==w->buffer;seen[i]=w->buffer;if(!duplicate)borrowed=ple_host_add(borrowed,ggml_backend_buffer_get_size(w->buffer));
  }
  need(std::filesystem::create_directory(directory),"QSA QKV rows: fresh receipt required");ctx_=context();
  input_[1]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,2*width);packed_[1]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_I32,2*packed_bytes/4);
  input_[0]=ggml_view_1d(ctx_.get(),input_[1],width,0);packed_[0]=ggml_view_1d(ctx_.get(),packed_[1],packed_bytes/4,0);
  for(size_t i=0;i<3;++i){out_[i][1]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,2*outputs[i]);out_[i][0]=ggml_view_1d(ctx_.get(),out_[i][1],outputs[i],0);ggml_set_output(out_[i][0]);ggml_set_output(out_[i][1]);
   weight_[i]=ggml_view_1d(ctx_.get(),weights[i],width*outputs[i],0);weight_[i]->type=GGML_TYPE_I32;weight_[i]->ne[0]=sizes[i]/4;weight_[i]->nb[0]=4;weight_[i]->nb[1]=weight_[i]->nb[2]=weight_[i]->nb[3]=sizes[i];
   need(ggml_backend_view_init(weight_[i])==GGML_STATUS_SUCCESS,"QSA QKV rows: weight view");}
  size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));need(bank<=budget,"QSA QKV rows: bank budget");
  quant_.reset(strata_vk_native_stages_create_quant_rcp_observation(backend,(std::filesystem::path(directory)/"quant-effective.spv").string().c_str(),budget-bank));
  const size_t qb=strata_vk_native_stages_bytes(quant_.get());need(qb<=budget-bank,"QSA QKV rows: quant budget");
  dot_.reset(strata_vk_qsa_qkv_rows_create(backend,(std::filesystem::path(directory)/"dot-effective.spv").string().c_str(),ple_host_add(budget-bank-qb,borrowed)));
  buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));need(bool(buffer_),"QSA QKV rows: allocation");
  bytes_=ple_host_add(ple_host_add(ggml_backend_buffer_get_size(buffer_.get()),qb),strata_vk_q8_wo_bytes(dot_.get()));need(bytes_<=budget&&bytes_<=estimate(backend),"QSA QKV rows: actual budget");
 }
 PortableNativeQsaQkvRows(const PortableNativeQsaQkvRows&)=delete;
 PortableNativeQsaQkvRows& operator=(const PortableNativeQsaQkvRows&)=delete;
 size_t bytes()const{return bytes_;} bool valid()const{return valid_;}
 // Output bank is only usable after a successful run; the next run invalidates it.
 ggml_tensor* output(size_t role,size_t rows)const{need(valid_&&role<3&&(rows==1||rows==2)&&rows==rows_,"QSA QKV rows: invalid output");return out_[role][rows-1];}
 void run(const std::vector<float>& input,size_t rows){
  valid_=false;need((rows==1||rows==2)&&input.size()==rows*width,"QSA QKV rows: shape");qsa_q8_input_domain(input,rows*80,"QSA QKV rows");
  ggml_backend_tensor_set(input_[rows-1],input.data(),0,input.size()*4);
  strata_vk_qsa_input_quantize_bounded(quant_.get(),input_[rows-1],packed_[rows-1],rows);
  std::vector<uint8_t> packed(rows*packed_bytes);ggml_backend_tensor_get(packed_[rows-1],packed.data(),0,packed.size());
  // QKV admits finite packed half-subnormal scales, unlike QSA WO.
  for(size_t r=0;r<rows;++r)qsa_qkv_packed_domain(packed.data()+r*packed_bytes,packed_bytes);
  for(size_t i=0;i<3;++i){strata_vk_qsa_qkv_run_bounded(dot_.get(),weight_[i],packed_[rows-1],out_[i][rows-1],outputs[i],rows);
   std::vector<float> values(rows*outputs[i]);ggml_backend_tensor_get(out_[i][rows-1],values.data(),0,values.size()*4);for(float x:values)need(qsa_wo_normal_or_zero(x),"QSA QKV rows: output domain");}
  rows_=rows;valid_=true;
 }
private: size_t rows_=0;
};
}
