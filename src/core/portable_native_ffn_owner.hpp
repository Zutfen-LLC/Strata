#pragma once
// Per-layer fixed T1/T2 FFN row ownership. Backend is borrowed and must
// outlive the owner. GGUF views need only survive construction. The caller
// supplies the existing fixed expert callback at run time; no scheduling here.
#include "portable_native_ffn_rows.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "native_shared_layout.hpp"
#include "native_qsa_wo_domain.hpp"
namespace strata::core::portable_detail {
class PortableNativeFfnOwner {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context context_{nullptr,ggml_free};Buffer storage_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<PortableNativeFfnProjection> projection_;
 std::array<std::unique_ptr<PortableNativeRouter>,2> router_;
 std::array<std::unique_ptr<PortableNativeShared>,2> shared_;
 std::array<std::unique_ptr<PortableNativeMoeCombine>,2> combine_;
 std::unique_ptr<PortableNativeFfnRows> rows_;
 static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
public:
 using Roles=std::array<PortableTensorView,5>;
 struct Budgets {size_t host,device,roles,projection;std::array<size_t,2> router,shared,combine;};
 // Requested payload only; excludes backend-private graph workspace, driver,
 // allocator, source mappings, provider and caller-retained results.
 static constexpr size_t host_bound(){return 4u<<20;}
 static Roles from_gguf(const GgufModel& m,int layer){
  need(layer>=0&&layer<48,"FFN owner: layer");Roles result;
  const char* names[]={"ffn_gate_inp.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight","ffn_gate_inp_shexp.weight"};
  for(size_t i=0;i<5;++i){auto name="blk."+std::to_string(layer)+"."+names[i];size_t s=0;auto*t=m.find(name,&s);need(t&&m.in_bounds(*t,s),"FFN owner: missing/out-of-bounds role");result[i]={name,{},t->type,m.shard(s).tensor_data(*t),size_t(tensor_payload_bytes(*t))};for(auto d:t->shape){need(d<=INT64_MAX,"FFN owner: dimension overflow");result[i].shape.push_back(int64_t(d));}}
  return result;
 }
 static void validate(int layer,const Roles& r){
  need(layer>=0&&layer<48,"FFN owner: layer");const char* names[]={"ffn_gate_inp.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight","ffn_gate_inp_shexp.weight"};
  const std::array<std::vector<int64_t>,5> shapes{{{2560,512},{2560,640},{2560,640},{640,2560},{2560}}};
  for(size_t i=0;i<5;++i){bool scalar=i==4;size_t n=i==0?2560u*512*4:i<4?size_t(native_shared_weight_bytes(uint32_t(shapes[i][0]),uint32_t(shapes[i][1]))):2560u*(r[i].type==0?4:2);
   need(r[i].name=="blk."+std::to_string(layer)+"."+names[i]&&r[i].data&&r[i].shape==shapes[i]&&r[i].bytes==n&&(i==0?r[i].type==0:scalar?(r[i].type==0||r[i].type==30):r[i].type==8),"FFN owner: role descriptor");
   if(i>=1&&i<=3)qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(r[i].data),n/34,34),r[i].name,true);
  }
  for(size_t j=0;j<2560;++j){uint16_t h;if(r[4].type==0){float x;std::memcpy(&x,static_cast<const uint8_t*>(r[4].data)+j*4,4);h=native_shared_promoted_bf16(x);}else{std::memcpy(&h,static_cast<const uint8_t*>(r[4].data)+j*2,2);need((h&0x7f80u)!=0x7f80u,"FFN owner: nonfinite scalar");}}
 }
 PortableNativeFfnOwner(int layer,ggml_backend_t b,const Roles&r,const std::string& prefix,const Budgets& caps){
  need(b&&!prefix.empty(),"FFN owner: backend/prefix");validate(layer,r);need(caps.host>=host_bound(),"FFN owner: host budget");
  size_t total=ple_host_add(caps.roles,caps.projection);for(size_t i=0;i<2;++i){total=ple_host_add(total,caps.router[i]);total=ple_host_add(total,caps.shared[i]);total=ple_host_add(total,caps.combine[i]);}need(total<=caps.device,"FFN owner: aggregate device budget");
  context_.reset(ggml_init({16384,nullptr,true}));need(bool(context_),"FFN owner: metadata");
  auto* router=ggml_new_tensor_2d(context_.get(),GGML_TYPE_F32,2560,512);std::array<ggml_tensor*,4> w{};
  for(size_t i=0;i<3;++i)w[i]=ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,int64_t(r[i+1].bytes/4));w[3]=ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,1280);
  need(router&&w[0]&&w[1]&&w[2]&&w[3],"FFN owner: tensor metadata");auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(b));need(bound<=caps.roles,"FFN owner: role budget");
  storage_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),b));need(bool(storage_)&&ggml_backend_buffer_get_size(storage_.get())<=caps.roles,"FFN owner: role allocation");
  ggml_backend_tensor_set(router,r[0].data,0,r[0].bytes);for(size_t i=0;i<3;++i)ggml_backend_tensor_set(w[i],r[i+1].data,0,r[i+1].bytes);
  std::vector<uint16_t> scalar(2560);for(size_t j=0;j<2560;++j){if(r[4].type==0){float x;std::memcpy(&x,static_cast<const uint8_t*>(r[4].data)+j*4,4);scalar[j]=native_shared_promoted_bf16(x);}else std::memcpy(&scalar[j],static_cast<const uint8_t*>(r[4].data)+j*2,2);}ggml_backend_tensor_set(w[3],scalar.data(),0,scalar.size()*2);
  projection_=std::make_unique<PortableNativeFfnProjection>(b,router,caps.projection);
  for(size_t i=0;i<2;++i){auto n=uint32_t(i+1);router_[i]=std::make_unique<PortableNativeRouter>(b,n,caps.router[i],prefix+".router");shared_[i]=std::make_unique<PortableNativeShared>(b,2560,640,n,w,caps.shared[i],prefix+".shared");combine_[i]=std::make_unique<PortableNativeMoeCombine>(b,n,caps.combine[i],prefix+".combine");}
  rows_=std::make_unique<PortableNativeFfnRows>(layer,*projection_,PortableNativeFfnRows::native_dispatch({router_[0].get(),router_[1].get()},{shared_[0].get(),shared_[1].get()},{combine_[0].get(),combine_[1].get()}));
 }
 PortableNativeFfnOwner(int layer,ggml_backend_t b,const GgufModel&m,const std::string&p,const Budgets& caps):PortableNativeFfnOwner(layer,b,from_gguf(m,layer),p,caps){}
 PortableNativeFfnOwner(const PortableNativeFfnOwner&)=delete;PortableNativeFfnOwner&operator=(const PortableNativeFfnOwner&)=delete;
 PortableNativeFfnRows& rows(){return *rows_;}
};
}
