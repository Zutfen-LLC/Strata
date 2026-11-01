#pragma once
#include "strata_vk_native_gdn.hpp"
#include "native_hc_layout.hpp"
#include "ggml-alloc.h"
#include <array>
#include <memory>
namespace strata::core::portable_detail {
// Fixed T1 experimental bridge. Prefix writes persistent leaves, component writes
// a candidate, existing suffix/FFN/transaction decides whether to publish state.
class PortableNativeGdn {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buf_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeGdn,decltype(&strata_vk_native_gdn_destroy)>kernel_{nullptr,strata_vk_native_gdn_destroy};
 size_t bytes_=0;
public:
 std::array<ggml_tensor*,13>t{};ggml_tensor*history=nullptr,*auxiliary=nullptr;uint64_t calls=0;
 PortableNativeGdn(ggml_backend_t backend,size_t budget,const std::string&receipt){
  native_hc_need(backend&&!receipt.empty(),"native GDN: backend/receipt required");ctx_.reset(ggml_init({16384,nullptr,true}));native_hc_need(bool(ctx_),"native GDN: bank metadata");
  const size_t count[]={786432,10240,48,48,6144,128,786432,6144,48,6144,49,6144,6144};
  auxiliary=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,48+49+6144+6144);size_t auxoff=0;
  for(size_t i:{8,10,11,12}){t[i]=ggml_view_1d(ctx_.get(),auxiliary,count[i],auxoff*4);ggml_set_input(t[i]);auxoff+=count[i];}
  for(size_t i=1;i<13;++i)if(i!=5&&i!=8&&i!=10&&i!=11&&i!=12){t[i]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,count[i]);ggml_set_input(t[i]);}
  history=ggml_new_tensor_3d(ctx_.get(),GGML_TYPE_F32,3,10240,1);ggml_set_input(history);
  size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));native_hc_need(bound<=budget,"native GDN: aligned bank admission");buf_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));native_hc_need(bool(buf_),"native GDN: bank allocation");bytes_=ggml_backend_buffer_get_size(buf_.get());native_hc_need(bytes_<=budget,"native GDN: actual bank admission");kernel_.reset(strata_vk_native_gdn_create(backend,receipt.c_str(),budget-bytes_));bytes_+=strata_vk_native_gdn_bytes(kernel_.get());
 }
 size_t bytes()const{return bytes_;}
 void run(ggml_tensor*state,ggml_tensor*gamma){t[0]=state;t[5]=gamma;strata_vk_native_gdn_run(kernel_.get(),t.data(),auxiliary);++calls;}
};
}
