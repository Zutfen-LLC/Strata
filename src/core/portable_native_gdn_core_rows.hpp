#pragma once
// Standalone causal preparation/recurrence bank. No model chunk admission.
// Backend and role tensors (including storage) are borrowed for the owner's life.
#include "portable_native_ple_host_bound.hpp"
#include "strata_vk_native_gdn_conv_rows.hpp"
#include "strata_vk_native_gdn_ab_rows.hpp"
#include "strata_vk_native_gdn_rows.hpp"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>
namespace strata::core::portable_detail {
class PortableNativeGdnCoreRows {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 struct Bank {ggml_tensor *mixed,*qkv,*z,*h,*gate,*beta,*pre,*ex,*silu,*rs,*arg,*abdiag,*y,*oc,*aux,*decay,*scale,*sigexp,*sigmoid;};
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeGdnConvRows,decltype(&strata_vk_native_gdn_conv_rows_destroy)> conv_{nullptr,strata_vk_native_gdn_conv_rows_destroy};
 std::unique_ptr<StrataVkNativeGdnAbRows,decltype(&strata_vk_native_gdn_ab_rows_destroy)> ab_{nullptr,strata_vk_native_gdn_ab_rows_destroy};
 std::unique_ptr<StrataVkNativeGdnRows,decltype(&strata_vk_native_gdn_rows_destroy)> recurrence_{nullptr,strata_vk_native_gdn_rows_destroy};
 std::array<ggml_tensor*,2>state_{},history_{};std::array<Bank,2> bank_{};
 // Roles: convolution, alpha, beta, dt, A and gamma; storage is borrowed.
 std::array<ggml_tensor*,6> roles_{};ggml_backend_t backend_=nullptr;size_t bytes_=0;unsigned committed_=0;uint32_t rows_=0;bool candidate_=false,poisoned_=false;
 static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
 static Context context(){Context c(ggml_init({65536,nullptr,true}),ggml_free);need(bool(c),"GDN core rows: metadata");return c;}
 static bool span(ggml_backend_t backend,const ggml_tensor*t){
  if(!backend||!t||!t->data||!t->buffer||!ggml_is_contiguous(t)||ggml_backend_buffer_get_type(t->buffer)!=ggml_backend_get_default_buffer_type(backend))return false;
  const auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(t->buffer)),data=reinterpret_cast<uintptr_t>(t->data);const size_t size=ggml_backend_buffer_get_size(t->buffer);
  return data>=base&&data-base<=size&&ggml_nbytes(t)<=size-(data-base);
 }
 static ggml_tensor* make(ggml_context*c,size_t n){auto*t=ggml_new_tensor_1d(c,GGML_TYPE_F32,n);need(t,"GDN core rows: tensor metadata");return t;}
 static void allocate(ggml_context*c,std::array<ggml_tensor*,2>&state,std::array<ggml_tensor*,2>&history,std::array<Bank,2>&banks){
  for(size_t i=0;i<2;++i){state[i]=make(c,786432);history[i]=make(c,30720);}
  for(size_t k=0;k<2;++k){size_t r=k+1;auto&b=banks[k];b.mixed=make(c,r*2560);b.qkv=make(c,r*10240);b.z=make(c,r*6144);b.h=make(c,r*10240);b.gate=make(c,r*48);b.beta=make(c,r*48);b.pre=make(c,r*10240);b.ex=make(c,r*10240);b.silu=make(c,r*10240);b.rs=make(c,r*32);b.arg=make(c,r*32);b.abdiag=make(c,r*336);b.y=make(c,r*6144);b.oc=make(c,r*6144);b.aux=make(c,r*(48+49+6144+6144));size_t offset=0;
   auto view=[&](size_t n){auto*t=ggml_view_1d(c,b.aux,n,offset*4);need(t,"GDN core rows: auxiliary view");offset+=n;return t;};
   b.decay=view(r*48);b.scale=view(r*49);b.sigexp=view(r*6144);b.sigmoid=view(r*6144);
  }
 }
public:
 struct Caps{size_t conv,ab,recurrence;};
 // Host scratch requested by reset/read/run, excluding caller copies, roles,
 // backend internals and allocator overhead. Not an RSS bound.
 static size_t host_bound(){return ple_host_add(65536,ple_host_add(786432*sizeof(float),30720*sizeof(float)));}
 static size_t estimate(ggml_backend_t backend,Caps caps){need(backend,"GDN core rows: backend");auto c=context();std::array<ggml_tensor*,2>s{},h{};std::array<Bank,2>b{};allocate(c.get(),s,h,b);auto n=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(backend));return ple_host_add(n,ple_host_add(caps.conv,ple_host_add(caps.ab,caps.recurrence)));}
 PortableNativeGdnCoreRows(ggml_backend_t backend,const std::array<ggml_tensor*,6>&roles,Caps caps,size_t budget,const std::string&dir):roles_(roles),backend_(backend){
  need(backend&&!dir.empty()&&estimate(backend,caps)<=budget,"GDN core rows: admission");
  const size_t sizes[]={40960*4,61440*4,61440*4,48*4,48*4,128*4};size_t borrowed=0;std::array<ggml_backend_buffer_t,6>seen{};
  for(size_t i=0;i<6;++i){auto*t=roles[i];need(span(backend,t)&&ggml_nbytes(t)==sizes[i]&&((i==1||i==2)?(t->type==GGML_TYPE_I32||t->type==GGML_TYPE_BF16):t->type==GGML_TYPE_F32),"GDN core rows: role descriptor");bool duplicate=false;for(size_t j=0;j<i;++j)duplicate|=seen[j]==t->buffer;seen[i]=t->buffer;if(!duplicate)borrowed=ple_host_add(borrowed,ggml_backend_buffer_get_size(t->buffer));}
  need(std::filesystem::create_directory(dir),"GDN core rows: fresh receipt");ctx_=context();allocate(ctx_.get(),state_,history_,bank_);auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));need(bound<=budget,"GDN core rows: bank budget");
  // Wrappers count full borrowed buffers and the bank in their local guards.
  auto component_budget=[&](size_t cap){return ple_host_add(ple_host_add(bound,borrowed),ple_host_add(cap,cap));};
  conv_.reset(strata_vk_native_gdn_conv_rows_create(backend,(std::filesystem::path(dir)/"conv-effective.spv").string().c_str(),component_budget(caps.conv)));need(conv_&&strata_vk_native_gdn_conv_rows_bytes(conv_.get())<=caps.conv,"GDN core rows: conv cap");
  ab_.reset(strata_vk_native_gdn_ab_rows_create(backend,(std::filesystem::path(dir)/"ab-effective.spv").string().c_str(),component_budget(caps.ab)));need(ab_&&strata_vk_native_gdn_ab_rows_bytes(ab_.get())<=caps.ab,"GDN core rows: AB cap");
  recurrence_.reset(strata_vk_native_gdn_rows_create(backend,(std::filesystem::path(dir)/"recurrence-effective.spv").string().c_str(),component_budget(caps.recurrence)));need(recurrence_&&strata_vk_native_gdn_rows_bytes(recurrence_.get())<=caps.recurrence,"GDN core rows: recurrence cap");
  buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));need(bool(buffer_)&&ggml_backend_buffer_get_size(buffer_.get())<=bound,"GDN core rows: allocation");bytes_=ple_host_add(ggml_backend_buffer_get_size(buffer_.get()),ple_host_add(strata_vk_native_gdn_conv_rows_bytes(conv_.get()),ple_host_add(strata_vk_native_gdn_ab_rows_bytes(ab_.get()),strata_vk_native_gdn_rows_bytes(recurrence_.get()))));need(bytes_<=budget&&bytes_<=estimate(backend,caps),"GDN core rows: actual budget");reset();
 }
 PortableNativeGdnCoreRows(const PortableNativeGdnCoreRows&)=delete;PortableNativeGdnCoreRows&operator=(const PortableNativeGdnCoreRows&)=delete;
 size_t bytes()const{return bytes_;}bool candidate()const{return candidate_;}bool poisoned()const{return poisoned_;}
 static void validate_seed(const std::vector<float>&state,const std::vector<float>&history){
  need(state.size()==786432&&history.size()==30720,"GDN core rows: reset extent");
  for(float x:state)need(std::isfinite(x)&&(x==0.f||std::isnormal(x)),"GDN core rows: reset state domain");
  for(float x:history)need(std::isfinite(x)&&(x==0.f||std::isnormal(x)),"GDN core rows: reset history domain");
 }
 void reset(){std::vector<float>state(786432,0.f),history(30720,0.f);reset(state,history);}
 void reset(const std::vector<float>&state,const std::vector<float>&history){
  validate_seed(state,history);
  candidate_=false;poisoned_=true;for(auto*t:state_)ggml_backend_tensor_set(t,state.data(),0,state.size()*4);for(auto*t:history_)ggml_backend_tensor_set(t,history.data(),0,history.size()*4);committed_=0;rows_=0;poisoned_=false;
 }
 std::vector<float> committed_state()const{std::vector<float>v(786432);ggml_backend_tensor_get(state_[committed_],v.data(),0,v.size()*4);return v;}
 std::vector<float> committed_history()const{std::vector<float>v(30720);ggml_backend_tensor_get(history_[committed_],v.data(),0,v.size()*4);return v;}
 // Copies keep the caller's tensors and committed state read-only. Any failure
 // after execution begins invalidates the candidate and requires reset.
 void stage(const std::vector<float>&mixed,ggml_tensor*qkv,ggml_tensor*z,uint32_t rows){
  need(!poisoned_&&!candidate_&&(rows==1||rows==2)&&mixed.size()==size_t(rows)*2560&&span(backend_,qkv)&&span(backend_,z)&&qkv->type==GGML_TYPE_F32&&z->type==GGML_TYPE_F32&&ggml_nbytes(qkv)==size_t(rows)*10240*4&&ggml_nbytes(z)==size_t(rows)*6144*4,"GDN core rows: stage spans/state");
  auto&b=bank_[rows-1];auto next=1-committed_;candidate_=false;poisoned_=true;
  ggml_backend_tensor_set(b.mixed,mixed.data(),0,mixed.size()*4);std::vector<float>copy(rows*10240);ggml_backend_tensor_get(qkv,copy.data(),0,copy.size()*4);ggml_backend_tensor_set(b.qkv,copy.data(),0,copy.size()*4);copy.resize(rows*6144);ggml_backend_tensor_get(z,copy.data(),0,copy.size()*4);ggml_backend_tensor_set(b.z,copy.data(),0,copy.size()*4);
  ggml_tensor*ct[]={history_[committed_],b.qkv,roles_[0],history_[next],b.h,b.pre,b.ex,b.silu,b.rs,b.arg};strata_vk_native_gdn_conv_rows_run(conv_.get(),ct,rows);
  ggml_tensor*at[]={b.mixed,roles_[1],roles_[2],roles_[3],roles_[4],b.gate,b.beta,b.abdiag};need(strata_vk_native_gdn_ab_rows_run(ab_.get(),at,rows)==0,"GDN core rows: AB domain");
  ggml_tensor*rt[]={state_[committed_],b.h,b.gate,b.beta,b.z,roles_[5],state_[next],b.y,b.decay,b.oc,b.scale,b.sigexp,b.sigmoid};strata_vk_native_gdn_rows_run(recurrence_.get(),rt,b.aux,rows);
  rows_=rows;candidate_=true;poisoned_=false;
 }
 std::vector<float> output()const{need(candidate_&&!poisoned_,"GDN core rows: no candidate");std::vector<float>v(rows_*6144);ggml_backend_tensor_get(bank_[rows_-1].y,v.data(),0,v.size()*4);return v;}
private:
 friend class PortableNativePleGdnComposition;
 friend class PortableNativeOrdinaryGdnComposition;
 // Only the composition may swap after PLE has committed. All throwing work
 // and candidate validation must already have completed.
 void publish_prepared()noexcept{committed_=1-committed_;candidate_=false;rows_=0;}
 void invalidate()noexcept{candidate_=false;poisoned_=true;rows_=0;}
};
}
