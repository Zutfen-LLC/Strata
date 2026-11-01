#pragma once
#include "strata_vk_native_gdn.hpp"
#include "native_hc_layout.hpp"
#include "ggml-alloc.h"
#include <array>
#include <memory>
#include <filesystem>
#include <fstream>
#include <cstdio>
namespace strata::core::portable_detail {
// Fixed T1 experimental bridge. Prefix writes persistent leaves, component writes
// a candidate, existing suffix/FFN/transaction decides whether to publish state.
class PortableNativeGdn {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buf_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeGdn,decltype(&strata_vk_native_gdn_destroy)>kernel_{nullptr,strata_vk_native_gdn_destroy};
 size_t bytes_=0;
 std::filesystem::path directory_;bool record_=false,recorded_=false,committed_=false;
 static void word(std::ostream&out,uint32_t b){for(unsigned shift=0;shift<32;shift+=8)out.put(char(b>>shift));}
 static void tensor(std::ostream&out,const ggml_tensor*t){std::vector<uint32_t>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);for(uint32_t w:v)word(out,w);native_hc_need(bool(out),"native GDN: receipt write failed");}
 void file(const char*name,const ggml_tensor*t){auto path=directory_/name;native_hc_need(!std::filesystem::exists(path),"native GDN: fresh receipt file required");std::ofstream out(path,std::ios::binary);native_hc_need(bool(out),"native GDN: receipt open");tensor(out,t);}
 void refusal(uint64_t position)noexcept{try{auto dir=directory_/"input-refusal";native_hc_need(std::filesystem::create_directory(dir),"native GDN: preserve previous refusal");for(size_t i=0;i<6;++i){std::ofstream out(dir/("input-"+std::to_string(i)+".f32le"),std::ios::binary);tensor(out,t[i]);}std::ofstream out(dir/"metadata.json");out<<"{\"position\":"<<position<<",\"outputs_captured\":false,\"accepted\":false}\n";}catch(const std::exception&e){std::fprintf(stderr,"native GDN input receipt: %s\n",e.what());}}
 void first(uint64_t position){if(!record_||recorded_)return;native_hc_need(position==0,"native GDN: first receipt must use actual initial position");auto path=directory_/"fixture.bin";native_hc_need(!std::filesystem::exists(path),"native GDN: preserve first partial");std::ofstream fixture(path,std::ios::binary);word(fixture,0x314e4447);word(fixture,1);for(size_t i=0;i<6;++i)tensor(fixture,t[i]);fixture.close();native_hc_need(bool(fixture),"native GDN fixture close");
  const char*names[]={"next-state.f32le","y.f32le","decay-exp.f32le","oc.f32le","rs.f32le","sigmoid-exp.f32le","sigmoid.f32le"};for(size_t i=6;i<13;++i)file(names[i-6],t[i]);std::ofstream out(directory_/"first.json");out<<"{\"position\":0,\"native_state_layout\":\"row-head-column\",\"committed\":false,\"qualified\":false}\n";native_hc_need(bool(out),"native GDN first metadata");recorded_=true;
 }

public:
 std::array<ggml_tensor*,13>t{};ggml_tensor*history=nullptr,*auxiliary=nullptr;uint64_t calls=0;
 PortableNativeGdn(ggml_backend_t backend,size_t budget,const std::string&receipt,bool record_first=false){
  directory_=std::filesystem::path(receipt).parent_path();record_=record_first;
  if(!directory_.empty()&&!std::filesystem::exists(directory_))native_hc_need(std::filesystem::create_directory(directory_),"native GDN: fresh receipt directory");
  native_hc_need(backend&&!receipt.empty(),"native GDN: backend/receipt required");ctx_.reset(ggml_init({16384,nullptr,true}));native_hc_need(bool(ctx_),"native GDN: bank metadata");
  const size_t count[]={786432,10240,48,48,6144,128,786432,6144,48,6144,49,6144,6144};
  auxiliary=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,48+49+6144+6144);size_t auxoff=0;
  for(size_t i:{8,10,11,12}){t[i]=ggml_view_1d(ctx_.get(),auxiliary,count[i],auxoff*4);ggml_set_input(t[i]);auxoff+=count[i];}
  for(size_t i=1;i<13;++i)if(i!=5&&i!=8&&i!=10&&i!=11&&i!=12){t[i]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,count[i]);ggml_set_input(t[i]);}
  history=ggml_new_tensor_3d(ctx_.get(),GGML_TYPE_F32,3,10240,1);ggml_set_input(history);
  size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));native_hc_need(bound<=budget,"native GDN: aligned bank admission");buf_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));native_hc_need(bool(buf_),"native GDN: bank allocation");bytes_=ggml_backend_buffer_get_size(buf_.get());native_hc_need(bytes_<=budget,"native GDN: actual bank admission");kernel_.reset(strata_vk_native_gdn_create(backend,receipt.c_str(),budget-bytes_));bytes_+=strata_vk_native_gdn_bytes(kernel_.get());
 }
 size_t bytes()const{return bytes_;}
 void run(ggml_tensor*state,ggml_tensor*gamma,uint64_t position=0){t[0]=state;t[5]=gamma;try{strata_vk_native_gdn_run(kernel_.get(),t.data(),auxiliary);}catch(...){refusal(position);throw;}++calls;first(position);}
 // Receipt failure after actual commit must not turn a completed transaction into
 // a rejected step with divergent host/device state. The offline checker requires
 // this proof; diagnostic I/O failures remain explicit in stderr/missing receipt.
 void record_commit(ggml_tensor*state,uint64_t position,ggml_tensor*history=nullptr)noexcept{if(!record_||!recorded_||committed_||position!=0)return;try{file("committed-state.f32le",state);if(history)file("committed-history.f32le",history);committed_=true;}catch(const std::exception&e){std::fprintf(stderr,"native GDN committed receipt: %s\n",e.what());}}

};
}
