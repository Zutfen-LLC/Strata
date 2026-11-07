#pragma once
// Internal HC prefill products only. No normalization, SiLU or publication.
// The backend is borrowed; packed immutable roles need only survive construction.
// Bounds exclude backend-private workspace, driver and physical allocation costs.
#include "portable_native_ple_host_bound.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core::portable_detail {
class PortableNativeHcBf16Products {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 static constexpr size_t metadata=512u<<10, max_role=10240u*320u*2u;
 static constexpr std::array<size_t,3> sizes{10240u*320u*2u,320u*10240u*2u,10240u*4u*2u};
 ggml_backend_t backend_;
 Context ctx_{nullptr,ggml_free}; Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::array<ggml_tensor*,3> weights_{};
 std::array<std::array<ggml_tensor*,2>,2> inputs_{}; // normalized, low
 std::array<std::array<ggml_tensor*,2>,3> outputs_{};
 std::array<std::array<ggml_cgraph*,2>,3> graphs_{};
 size_t rows_=0, stage_=0, attempts_=0, device_bytes_=0;
 static void need(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
 static Context context(size_t host_budget){need(host_budget>=host_bound(),"HC BF16 products: host budget");Context c(ggml_init({metadata,nullptr,true}),ggml_free);need(bool(c),"HC BF16 products: metadata");return c;}
 explicit PortableNativeHcBf16Products(ggml_backend_t backend,size_t host_budget):backend_(backend),ctx_(context(host_budget)) {
  need(backend_,"HC BF16 products: backend");
  const int k[]={10240,320,10240},n[]={320,10240,4};
  for(size_t i=0;i<3;++i){weights_[i]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_BF16,k[i],n[i]);need(weights_[i],"HC BF16 products: weight metadata");}
  for(size_t r=0;r<2;++r){
   inputs_[0][r]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,10240,r+1);
   inputs_[1][r]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,320,r+1);
   need(inputs_[0][r]&&inputs_[1][r],"HC BF16 products: input metadata");
   for(size_t i=0;i<3;++i){outputs_[i][r]=ggml_mul_mat(ctx_.get(),weights_[i],inputs_[i==1?1:0][r]);
    need(outputs_[i][r]&&ggml_backend_supports_op(backend_,outputs_[i][r]),"HC BF16 products: unsupported product");
    graphs_[i][r]=ggml_new_graph(ctx_.get());need(graphs_[i][r],"HC BF16 products: graph metadata");ggml_build_forward_expand(graphs_[i][r],outputs_[i][r]);}
  }
 }
 void invalidate(){stage_=rows_=0;}
 void upload(const std::vector<float>& input,size_t n,ggml_tensor* dst){
  need(input.size()==n,"HC BF16 products: input extent");std::vector<float> rounded(n);
  for(size_t i=0;i<n;++i){float x=input[i];need(std::isfinite(x)&&(x==0.f||std::isnormal(x)),"HC BF16 products: input domain");
   float y=ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));need(std::isfinite(y)&&(y==0.f||std::isnormal(y)),"HC BF16 products: rounded domain");rounded[i]=y;}
  ggml_backend_tensor_set(dst,rounded.data(),0,n*sizeof(float));
 }
 // The scalar shaders may supply an already-rounded, widened BF16 image.
 // Scan the actual device span before copying it; no host rounding or scalar
 // activation is performed here. Images must not alias this owner's scratch.
 // Their full borrowed buffers are not in device_bytes(); the enclosing owner
 // must charge them, with deduplication, for the duration of each call.
 void upload_image(const ggml_tensor* src,size_t n,ggml_tensor* dst){
  need(src&&src->buffer&&src->data&&src->type==GGML_TYPE_F32&&ggml_is_contiguous(src)&&ggml_nbytes(src)==n*sizeof(float)&&ggml_are_same_shape(src,dst)&&ggml_are_same_stride(src,dst)&&
       ggml_backend_buffer_get_type(src->buffer)==ggml_backend_get_default_buffer_type(backend_)&&src->buffer!=buffer_.get(),"HC BF16 products: image descriptor");
  auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(src->buffer)),data=reinterpret_cast<uintptr_t>(src->data);size_t cap=ggml_backend_buffer_get_size(src->buffer);
  need(data>=base&&data-base<=cap&&n*sizeof(float)<=cap-(data-base),"HC BF16 products: image span");
  std::vector<float> scan(n);ggml_backend_tensor_get(src,scan.data(),0,n*sizeof(float));
  for(float x:scan){uint32_t bits;std::memcpy(&bits,&x,sizeof(bits));need(std::isfinite(x)&&(x==0.f||std::isnormal(x))&&(bits&0xffffu)==0,"HC BF16 products: image domain");}
  ggml_backend_tensor_copy(src,dst);
 }
 void compute(size_t role,size_t rows){
  ++attempts_;auto status=ggml_backend_graph_compute(backend_,graphs_[role][rows-1]);ggml_backend_synchronize(backend_);
  need(status==GGML_STATUS_SUCCESS,"HC BF16 products: graph failure");
  const size_t n=(role==0?320:role==1?10240:4)*rows;std::vector<float> result(n);
  ggml_backend_tensor_get(outputs_[role][rows-1],result.data(),0,n*sizeof(float));
  for(float x:result)need(std::isfinite(x)&&(x==0.f||std::isnormal(x)),"HC BF16 products: output domain");
 }
public:
 // Covers metadata plus the largest role scan and the largest rounded input or
 // output scan conservatively at once. Allocator overhead is excluded.
 static size_t host_bound(){return metadata+max_role+2u*10240u*sizeof(float);}
 // Estimate includes graph-readable BF16 copies, both exact row banks and the
 // full deduplicated borrowed buffers supplied by the caller.
 static size_t device_bound(ggml_backend_t backend,size_t borrowed){PortableNativeHcBf16Products p(backend,host_bound());return ple_host_add(borrowed,ggml_backend_alloc_ctx_tensors_from_buft_size(p.ctx_.get(),ggml_backend_get_default_buffer_type(backend)));}
 // roles are down, up, injection (role bank indices 2, 3, 1).
 PortableNativeHcBf16Products(ggml_backend_t backend,const std::array<const ggml_tensor*,3>& roles,size_t host_budget,size_t device_budget)
 :PortableNativeHcBf16Products(backend,host_budget){
  size_t borrowed=0;
  for(size_t i=0;i<3;++i){auto*t=roles[i];need(t&&t->buffer&&t->data&&ggml_is_contiguous(t)&&t->type==GGML_TYPE_I32&&t->ne[0]==int64_t(sizes[i]/4)&&t->ne[1]==1&&t->ne[2]==1&&t->ne[3]==1&&ggml_nbytes(t)==sizes[i]&&ggml_backend_buffer_get_type(t->buffer)==ggml_backend_get_default_buffer_type(backend),"HC BF16 products: role descriptor");
   auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(t->buffer)),data=reinterpret_cast<uintptr_t>(t->data);size_t cap=ggml_backend_buffer_get_size(t->buffer);
   need(data>=base&&data-base<=cap&&sizes[i]<=cap-(data-base),"HC BF16 products: role span");
   bool seen=false;for(size_t j=0;j<i;++j)if(t->buffer==roles[j]->buffer){seen=true;auto other=reinterpret_cast<uintptr_t>(roles[j]->data);need(data<=other?other-data>=sizes[i]:data-other>=sizes[j],"HC BF16 products: overlapping roles");}
   if(!seen)borrowed=ple_host_add(borrowed,cap);
  }
  size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
  need(ple_host_add(borrowed,bank)<=device_budget,"HC BF16 products: device budget");
  std::vector<ggml_bf16_t> scan(max_role/2);
  for(size_t i=0;i<3;++i){ggml_backend_tensor_get(roles[i],scan.data(),0,sizes[i]);for(size_t j=0;j<sizes[i]/2;++j){float x=ggml_bf16_to_fp32(scan[j]);need(std::isfinite(x)&&(x==0.f||std::isnormal(x)),"HC BF16 products: weight domain");}}
  buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));need(bool(buffer_),"HC BF16 products: allocation");
  device_bytes_=ple_host_add(borrowed,ggml_backend_buffer_get_size(buffer_.get()));need(device_bytes_<=device_budget,"HC BF16 products: actual device budget");
  for(size_t i=0;i<3;++i){ggml_backend_tensor_get(roles[i],scan.data(),0,sizes[i]);ggml_backend_tensor_set(weights_[i],scan.data(),0,sizes[i]);}
 }
 size_t device_bytes()const{return device_bytes_;} size_t graph_attempts()const{return attempts_;}
 // Output is only valid until the next call, including a failed call. It is
 // scratch, not a published HC view. Sequence must be down, up, injection.
 ggml_tensor* output(size_t role)const{need(role<3&&stage_==role+1,"HC BF16 products: stale output");return outputs_[role][rows_-1];}
 void down(const std::vector<float>& normalized,size_t rows){invalidate();try{need(rows==1||rows==2,"HC BF16 products: rows");upload(normalized,rows*10240,inputs_[0][rows-1]);compute(0,rows);rows_=rows;stage_=1;}catch(...){invalidate();throw;}}
 void up(const std::vector<float>& low){try{need(stage_==1,"HC BF16 products: up order");upload(low,rows_*320,inputs_[1][rows_-1]);compute(1,rows_);stage_=2;}catch(...){invalidate();throw;}}
 void down_image(const ggml_tensor* normalized_image,size_t rows){invalidate();try{need(rows==1||rows==2,"HC BF16 products: rows");upload_image(normalized_image,rows*10240,inputs_[0][rows-1]);compute(0,rows);rows_=rows;stage_=1;}catch(...){invalidate();throw;}}
 void up_image(const ggml_tensor* low_image){try{need(stage_==1,"HC BF16 products: up order");upload_image(low_image,rows_*320,inputs_[1][rows_-1]);compute(1,rows_);stage_=2;}catch(...){invalidate();throw;}}
 void inject(){try{need(stage_==2,"HC BF16 products: injection order");compute(2,rows_);stage_=3;}catch(...){invalidate();throw;}}
};
}
