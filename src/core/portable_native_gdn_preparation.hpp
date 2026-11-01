#pragma once
#include "portable_native_gdn.hpp"
#include "strata_vk_native_gdn_conv.hpp"
#include "strata_vk_native_gdn_ab.hpp"
namespace strata::core::portable_detail {
// Borrows actual completed projection outputs and committed history. Candidate
// history/H/gate/beta reuse the existing recurrence bank; no second weight bank.
class PortableNativeGdnPreparation {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeGdnConv,decltype(&strata_vk_native_gdn_conv_destroy)>conv_{nullptr,strata_vk_native_gdn_conv_destroy};
 std::unique_ptr<StrataVkNativeGdnAb,decltype(&strata_vk_native_gdn_ab_destroy)>ab_{nullptr,strata_vk_native_gdn_ab_destroy};
 std::array<ggml_tensor*,10>c_{};std::array<ggml_tensor*,8>a_{};size_t bytes_=0;std::filesystem::path dir_;bool record_=false,recorded_=false;
 static Context context(){Context c(ggml_init({16384,nullptr,true}),ggml_free);native_hc_need(bool(c),"native GDN preparation metadata");return c;}
 static void word(std::ostream&o,uint32_t v){for(unsigned s=0;s<32;s+=8)o.put(char(v>>s));}
 static void tensor(std::ostream&o,ggml_tensor*t){std::vector<uint32_t>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);for(auto x:v)word(o,x);native_hc_need(bool(o),"native GDN preparation receipt write");}
 static void file(const std::filesystem::path&p,ggml_tensor*t){native_hc_need(!std::filesystem::exists(p),"native GDN preparation preserve partial");std::ofstream o(p,std::ios::binary);tensor(o,t);}
 static void leaves(ggml_context*c,std::array<ggml_tensor*,10>&conv,std::array<ggml_tensor*,8>&ab){for(size_t i=5;i<10;++i)conv[i]=ggml_new_tensor_1d(c,GGML_TYPE_F32,i<8?10240:32);ab[7]=ggml_new_tensor_1d(c,GGML_TYPE_F32,336);for(size_t i=5;i<10;++i)ggml_set_input(conv[i]);ggml_set_input(ab[7]);}
 void capture(const std::filesystem::path&dir,bool conv,bool outputs){native_hc_need(std::filesystem::create_directory(dir),"native GDN preparation fresh capture");std::ofstream f(dir/"fixture.bin",std::ios::binary);
  if(conv){word(f,0x31564347);word(f,1);for(size_t i=0;i<3;++i)tensor(f,c_[i]);}
  else{for(uint32_t v:{0x31424147u,1u,2560u,48u,1u,30u,0u,0u})word(f,v);for(size_t i=0;i<5;++i)tensor(f,a_[i]);}f.close();native_hc_need(bool(f),"native GDN preparation fixture close");
  if(!outputs)return;
  if(conv){const char*n[]={"next-history.f32le","h.f32le","pre.f32le","exp.f32le","silu.f32le","rs.f32le","norm-arg.f32le"};for(size_t i=3;i<10;++i)file(dir/n[i-3],c_[i]);}
  else{const char*n[]={"alpha","beta-logit","v","softplus-exp","softplus","beta-exp","beta-denominator"};std::array<uint32_t,336>v{};ggml_backend_tensor_get(a_[7],v.data(),0,v.size()*4);for(size_t i=0;i<7;++i){std::ofstream o(dir/(std::string(n[i])+".f32le"),std::ios::binary);for(size_t j=0;j<48;++j)word(o,v[i*48+j]);native_hc_need(bool(o),"native GDN ab diagnostic write");}file(dir/"gate.f32le",a_[5]);file(dir/"beta.f32le",a_[6]);}
 }
 void failure(const char*stage,uint64_t position,bool outputs,uint32_t status)noexcept{try{auto p=dir_/("failure-"+std::string(stage)+"-"+std::to_string(position));capture(p,std::string(stage)=="conv",outputs);std::ofstream o(p/"status.json");o<<"{\"position\":"<<position<<",\"status\":"<<status<<",\"current_outputs\":"<<(outputs?"true":"false")<<",\"accepted\":false}\n";}catch(const std::exception&e){std::fprintf(stderr,"native GDN preparation receipt: %s\n",e.what());}}
public:
 static size_t estimate(ggml_backend_t backend){auto c=context();std::array<ggml_tensor*,10>conv{};std::array<ggml_tensor*,8>ab{};leaves(c.get(),conv,ab);auto type=ggml_backend_get_default_buffer_type(backend);size_t b=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),type);auto m=context();ggml_new_tensor_1d(m.get(),GGML_TYPE_I32,128);auto extra=ggml_backend_alloc_ctx_tensors_from_buft_size(m.get(),type);native_hc_need(extra<=(SIZE_MAX-b)/2,"native GDN preparation estimate overflow");return b+2*extra;}
 PortableNativeGdnPreparation(ggml_backend_t backend,PortableNativeGdn&recur,ggml_tensor*wa,ggml_tensor*wb,ggml_tensor*dt,ggml_tensor*ssa,const std::string&directory,size_t budget,bool record):record_(record){
  native_hc_need(backend&&!directory.empty(),"native GDN preparation backend/receipt");dir_=directory;native_hc_need(std::filesystem::create_directory(dir_),"native GDN preparation fresh receipt namespace");ctx_=context();leaves(ctx_.get(),c_,a_);c_[3]=recur.history;c_[4]=recur.t[1];a_[1]=wa;a_[2]=wb;a_[3]=dt;a_[4]=ssa;a_[5]=recur.t[2];a_[6]=recur.t[3];auto type=ggml_backend_get_default_buffer_type(backend);size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),type);size_t reserve=estimate(backend);native_hc_need(reserve<=budget,"native GDN preparation complete aligned admission");
  conv_.reset(strata_vk_native_gdn_conv_create(backend,(dir_/"conv-effective.spv").c_str(),budget-bound));size_t meta=strata_vk_native_gdn_conv_bytes(conv_.get());native_hc_need(meta<=budget-bound,"native GDN conv metadata admission");ab_.reset(strata_vk_native_gdn_ab_create(backend,(dir_/"ab-effective.spv").c_str(),budget-bound-meta));meta+=strata_vk_native_gdn_ab_bytes(ab_.get());native_hc_need(meta<=budget-bound,"native GDN preparation metadata admission");buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));native_hc_need(bool(buffer_),"native GDN preparation bank allocation");bytes_=ggml_backend_buffer_get_size(buffer_.get())+meta;native_hc_need(bytes_<=reserve&&bytes_<=budget,"native GDN preparation actual bank exceeds estimate");
 }
 size_t bytes()const{return bytes_;}ggml_tensor*silu()const{return c_[7];}
 uint64_t conv_calls()const{return strata_vk_native_gdn_conv_calls(conv_.get());}uint64_t ab_calls()const{return strata_vk_native_gdn_ab_calls(ab_.get());}
 void run(ggml_tensor*history,ggml_tensor*qkv,ggml_tensor*mixed,ggml_tensor*weights,uint64_t position){c_[0]=history;c_[1]=qkv;c_[2]=weights;a_[0]=mixed;uint64_t before=conv_calls();try{strata_vk_native_gdn_conv_run(conv_.get(),c_.data(),true);}catch(...){bool completed=conv_calls()!=before;failure("conv",position,completed,completed?strata_vk_native_gdn_conv_status(conv_.get()):0);throw;}before=ab_calls();uint32_t status=0;try{status=strata_vk_native_gdn_ab_run(ab_.get(),a_.data());native_hc_need(status==0,"native GDN preparation a/b actual domain refused");}catch(...){failure("ab",position,ab_calls()!=before,status);throw;}
  if(record_&&!recorded_){native_hc_need(position==0,"native GDN preparation initial receipt position");capture(dir_/"conv-first",true,true);capture(dir_/"ab-first",false,true);recorded_=true;}
 }
};
}
