#pragma once
// Internal baseline AB products. Backend and packed BF16 role tensors are borrowed.
// The envelope excludes backend-private workspace, driver and physical allocation costs.
#include "portable_native_ple_host_bound.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
namespace strata::core::portable_detail {
enum class PortableGdnAbMethod { legacy_fused, baseline_bf16 };
class PortableNativeGdnBf16AbRows {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 static constexpr size_t metadata=512u<<10, scratch=512u<<10, role_bytes=2560u*48u*2u;
 ggml_backend_t backend_; Context ctx_{nullptr,ggml_free}; Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::array<ggml_tensor*,2> weight_{},input_{}; std::array<std::array<ggml_tensor*,2>,2> output_{};
 std::array<ggml_cgraph*,2> graph_{}; size_t ready_=0,attempts_=0,bytes_=0;
 static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
 static Context context(){Context c(ggml_init({metadata,nullptr,true}),ggml_free);need(bool(c),"GDN BF16 AB: metadata");return c;}
 explicit PortableNativeGdnBf16AbRows(ggml_backend_t backend):backend_(backend),ctx_(context()){
  need(backend,"GDN BF16 AB: backend");
  for(auto &w:weight_){w=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_BF16,2560,48);need(w,"GDN BF16 AB: weight metadata");}
  for(size_t r=0;r<2;++r){input_[r]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,2560,r+1);graph_[r]=ggml_new_graph(ctx_.get());need(input_[r]&&graph_[r],"GDN BF16 AB: graph metadata");
   for(size_t i=0;i<2;++i){output_[i][r]=ggml_mul_mat(ctx_.get(),weight_[i],input_[r]);need(output_[i][r]&&ggml_backend_supports_op(backend_,output_[i][r]),"GDN BF16 AB: unsupported product");ggml_build_forward_expand(graph_[r],output_[i][r]);}}
 }
public:
 static size_t host_bound(){return metadata+scratch;}
 // Includes two exact-size borrowed role payloads; constructor charges actual
 // deduplicated buffer sizes instead. Caller may need a larger budget.
 static size_t estimate(ggml_backend_t backend){PortableNativeGdnBf16AbRows p(backend);return ple_host_add(host_bound()+2*role_bytes,ggml_backend_alloc_ctx_tensors_from_buft_size(p.ctx_.get(),ggml_backend_get_default_buffer_type(backend)));}
 static size_t estimate_with_borrowed(ggml_backend_t backend,size_t borrowed){return ple_host_add(estimate(backend)-2*role_bytes,borrowed);}
 PortableNativeGdnBf16AbRows(ggml_backend_t backend,const std::array<ggml_tensor*,2>& roles,size_t budget):PortableNativeGdnBf16AbRows(backend){
  size_t borrowed=0;std::array<ggml_backend_buffer_t,2> seen{};
  for(size_t i=0;i<2;++i){auto*t=roles[i];need(t&&t->buffer&&t->data&&ggml_is_contiguous(t)&&(t->type==GGML_TYPE_I32||t->type==GGML_TYPE_BF16)&&ggml_nbytes(t)==role_bytes&&ggml_backend_buffer_get_type(t->buffer)==ggml_backend_get_default_buffer_type(backend),"GDN BF16 AB: role descriptor");
   auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(t->buffer)),data=reinterpret_cast<uintptr_t>(t->data);size_t cap=ggml_backend_buffer_get_size(t->buffer);need(data>=base&&data-base<=cap&&role_bytes<=cap-(data-base),"GDN BF16 AB: role span");
   if(i==0||t->buffer!=seen[0])borrowed=ple_host_add(borrowed,cap);seen[i]=t->buffer;
  }
  if(roles[0]->buffer==roles[1]->buffer){auto a=reinterpret_cast<uintptr_t>(roles[0]->data),b=reinterpret_cast<uintptr_t>(roles[1]->data);need(a<=b?b-a>=role_bytes:a-b>=role_bytes,"GDN BF16 AB: overlapping roles");}
  const size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));need(ple_host_add(ple_host_add(host_bound(),borrowed),bank)<=budget,"GDN BF16 AB: admission");
  std::vector<ggml_bf16_t> values(role_bytes/2);for(auto*t:roles){ggml_backend_tensor_get(t,values.data(),0,role_bytes);for(auto v:values){float x=ggml_bf16_to_fp32(v);need(std::isfinite(x)&&(x==0.f||std::isnormal(x)),"GDN BF16 AB: weight domain");}}
  buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));need(bool(buffer_),"GDN BF16 AB: allocation");bytes_=ple_host_add(ple_host_add(host_bound(),borrowed),ggml_backend_buffer_get_size(buffer_.get()));need(bytes_<=budget,"GDN BF16 AB: actual admission");
  for(size_t i=0;i<2;++i){ggml_backend_tensor_get(roles[i],values.data(),0,role_bytes);ggml_backend_tensor_set(weight_[i],values.data(),0,role_bytes);}
 }
 size_t bytes()const{return bytes_;}size_t graph_attempts()const{return attempts_;}bool valid()const{return ready_!=0;}
 ggml_tensor* output(size_t i,size_t rows)const{need(i<2&&ready_==rows&&ready_,"GDN BF16 AB: stale output");return output_[i][rows-1];}
 void run(const std::vector<float>&mixed,size_t rows){ready_=0;need((rows==1||rows==2)&&mixed.size()==rows*2560,"GDN BF16 AB: input shape");std::vector<float> rounded(mixed.size());
  for(size_t i=0;i<mixed.size();++i){need(std::isfinite(mixed[i])&&(mixed[i]==0.f||std::isnormal(mixed[i])),"GDN BF16 AB: input domain");rounded[i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(mixed[i]));need(std::isfinite(rounded[i])&&(rounded[i]==0.f||std::isnormal(rounded[i])),"GDN BF16 AB: rounded input domain");}
  ggml_backend_tensor_set(input_[rows-1],rounded.data(),0,rounded.size()*sizeof(float));++attempts_;auto status=ggml_backend_graph_compute(backend_,graph_[rows-1]);ggml_backend_synchronize(backend_);need(status==GGML_STATUS_SUCCESS,"GDN BF16 AB: graph failure");
  for(size_t i=0;i<2;++i){std::vector<float> v(rows*48);ggml_backend_tensor_get(output_[i][rows-1],v.data(),0,v.size()*sizeof(float));for(float x:v)need(std::isfinite(x),"GDN BF16 AB: product domain");}ready_=rows;
 }
};
}
