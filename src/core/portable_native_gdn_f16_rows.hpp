#pragma once
// Bounded internal native-baseline projection. Raw Q8_0 roles and backend are
// borrowed. The F16 copy is private; this does not qualify a GPU reduction.
#include "portable_native_ple_host_bound.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core::portable_detail {
enum class PortableGdnProjectionMethod { legacy_q8_1, native_f16 };
class PortableNativeGdnF16Rows {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 static constexpr size_t metadata=512u<<10, scratch=256u<<10;
 ggml_backend_t backend_; Context ctx_{nullptr,ggml_free}; Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::vector<size_t> columns_; size_t width_,bytes_=0,ready_=0,attempts_=0;
 std::vector<ggml_tensor*> weight_; ggml_tensor* input_[2]{};
 std::vector<std::array<ggml_tensor*,2>> output_; std::array<ggml_cgraph*,2> graph_{};
 static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
 static Context context(){Context c(ggml_init({metadata,nullptr,true}),ggml_free);need(bool(c),"GDN F16: metadata");return c;}
 static size_t add(size_t a,size_t b){return ple_host_add(a,b);}
 static void shapes(size_t k,const std::vector<size_t>& n){
  need((k==2560 && n==std::vector<size_t>{10240,6144}) || (k==6144 && n==std::vector<size_t>{2560}),"GDN F16: shape");
 }
 void graph(){
  for(size_t r=0;r<2;++r){input_[r]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,width_,r+1);graph_[r]=ggml_new_graph(ctx_.get());need(input_[r]&&graph_[r],"GDN F16: graph metadata");
   for(size_t i=0;i<columns_.size();++i){output_[i][r]=ggml_mul_mat(ctx_.get(),weight_[i],input_[r]);need(output_[i][r]&&ggml_backend_supports_op(backend_,output_[i][r]),"GDN F16: operation unsupported");ggml_build_forward_expand(graph_[r],output_[i][r]);}
  }
 }
 static void decode(const uint8_t* p,size_t blocks,uint16_t* out){
  for(size_t b=0;b<blocks;++b){uint16_t h=uint16_t(p[34*b])|uint16_t(p[34*b+1])<<8;
   need((h&0x7c00)!=0x7c00,"GDN F16: nonfinite scale");float scale=ggml_fp16_to_fp32(h);
   for(size_t j=0;j<32;++j){unsigned u=p[34*b+2+j];int code=u<128?int(u):int(u)-256;
    uint16_t v=ggml_fp32_to_fp16(float(code)*scale);need((v&0x7c00)!=0x7c00,"GDN F16: decoded weight overflow");out[32*b+j]=v;}
  }
 }
public:
 // Includes logical source-role bytes, resident raw backend buffers, decoded
 // tensors, graph metadata and bounded host scratch. Backend-private workspace,
 // driver/allocator overhead and caller-owned activation storage are excluded.
 // The estimate assumes exact-size distinct raw buffers; construction also
 // checks their actual deduplicated sizes before allocating the decoded bank.
 static size_t estimate(ggml_backend_t b,size_t k,const std::vector<size_t>& n){
  need(b,"GDN F16: backend");shapes(k,n);PortableNativeGdnF16Rows tmp(b,k,n);size_t raw=0;
  for(size_t x:n)raw=add(raw,(k/32)*34*x);
  return add(add(add(metadata,scratch),add(raw,raw)),ggml_backend_alloc_ctx_tensors_from_buft_size(tmp.ctx_.get(),ggml_backend_get_default_buffer_type(b)));
 }
private:
 PortableNativeGdnF16Rows(ggml_backend_t b,size_t k,const std::vector<size_t>& n):backend_(b),ctx_(context()),columns_(n),width_(k),weight_(n.size()),output_(n.size()){
  need(b,"GDN F16: backend");shapes(k,n);
  for(size_t i=0;i<n.size();++i)weight_[i]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F16,k,n[i]);graph();
 }
public:
 PortableNativeGdnF16Rows(ggml_backend_t b,size_t k,const std::vector<size_t>& n,const std::vector<ggml_tensor*>& weights,
  const std::vector<const void*>& roles,const std::vector<size_t>& sizes,const std::string& dir,size_t budget):PortableNativeGdnF16Rows(b,k,n){
  shapes(k,n);need(b&&weights.size()==n.size()&&roles.size()==n.size()&&sizes.size()==n.size(),"GDN F16: arguments");
  size_t raw=0,resident=0;std::vector<ggml_backend_buffer_t> seen;
  for(size_t i=0;i<n.size();++i){auto*w=weights[i];size_t expected=(k/32)*34*n[i];
   need(w&&w->buffer&&w->type==GGML_TYPE_Q8_0&&w->ne[0]==int64_t(k)&&w->ne[1]==int64_t(n[i])&&w->ne[2]==1&&w->ne[3]==1&&ggml_is_contiguous(w)&&ggml_nbytes(w)==expected&&roles[i]&&sizes[i]==expected,"GDN F16: role");
   need(ggml_backend_buffer_get_type(w->buffer)==ggml_backend_get_default_buffer_type(b),"GDN F16: role backend");
   auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(w->buffer));auto data=reinterpret_cast<uintptr_t>(w->data);size_t span=ggml_backend_buffer_get_size(w->buffer);
   need(base&&data>=base&&data-base<=span&&expected<=span-(data-base),"GDN F16: role span");
   if(std::find(seen.begin(),seen.end(),w->buffer)==seen.end()){seen.push_back(w->buffer);resident=add(resident,span);}raw=add(raw,expected);
  }
  size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(b));
  size_t admitted=add(add(add(metadata,scratch),add(raw,resident)),bank);need(admitted<=budget,"GDN F16: admission");
  // Validate source bytes and every converted operand before allocating the
  // large decoded bank. Fixed-size buffers bound conversion/readback peaks.
  std::array<uint8_t,34*256> check{};std::array<uint16_t,32*256> decoded{};
  for(size_t i=0;i<n.size();++i){auto*p=static_cast<const uint8_t*>(roles[i]);size_t blocks=sizes[i]/34;
   for(size_t at=0;at<blocks;at+=256){size_t count=std::min(size_t(256),blocks-at);ggml_backend_tensor_get(weights[i],check.data(),at*34,count*34);
    need(std::memcmp(check.data(),p+at*34,count*34)==0,"GDN F16: role mismatch");decode(p+at*34,count,decoded.data());}
  }
  need(std::filesystem::create_directory(dir),"GDN F16: fresh receipt");
  buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),b));need(bool(buffer_),"GDN F16: allocation");
  bytes_=add(add(add(metadata,scratch),add(raw,resident)),ggml_backend_buffer_get_size(buffer_.get()));need(bytes_<=budget,"GDN F16: actual admission");
  for(size_t i=0;i<n.size();++i){auto*p=static_cast<const uint8_t*>(roles[i]);size_t blocks=sizes[i]/34;
   for(size_t at=0;at<blocks;at+=256){size_t count=std::min(size_t(256),blocks-at);decode(p+at*34,count,decoded.data());ggml_backend_tensor_set(weight_[i],decoded.data(),at*64,count*64);}
  }
 }
 size_t bytes()const{return bytes_;} size_t graph_attempts()const{return attempts_;} bool valid()const{return ready_!=0;}
 ggml_tensor* output(size_t role,size_t rows)const{need(ready_&&ready_==rows&&role<output_.size(),"GDN F16: stale output");return output_[role][rows-1];}
 void run(const std::vector<float>& input,size_t rows){
  ready_=0;need((rows==1||rows==2)&&input.size()==rows*width_,"GDN F16: input shape");std::vector<float> rounded(input.size());
  for(size_t i=0;i<input.size();++i){need(std::isfinite(input[i]),"GDN F16: nonfinite input");uint16_t h=ggml_fp32_to_fp16(input[i]);need((h&0x7c00)!=0x7c00,"GDN F16: input overflow");rounded[i]=ggml_fp16_to_fp32(h);}
  ggml_backend_tensor_set(input_[rows-1],rounded.data(),0,rounded.size()*sizeof(float));++attempts_;
  auto status=ggml_backend_graph_compute(backend_,graph_[rows-1]);ggml_backend_synchronize(backend_);need(status==GGML_STATUS_SUCCESS,"GDN F16: graph failure");
  for(size_t i=0;i<columns_.size();++i){std::vector<float> values(rows*columns_[i]);ggml_backend_tensor_get(output_[i][rows-1],values.data(),0,values.size()*sizeof(float));for(float v:values)need(std::isfinite(v),"GDN F16: nonfinite output");}
  ready_=rows;
 }
};
}
