#pragma once
// Internal QSA role storage and per-layer stateful native components. Backend
// is borrowed and must outlive the owner. Mapped source is borrowed only during
// construction. The composition is exclusively borrowed from this owner.
#include "portable_native_qsa_composition.hpp"
#include "portable_native_ffn_hc_factory.hpp"
namespace strata::core::portable_detail {
class PortableNativeQsaOwner {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context context_{nullptr,ggml_free};Buffer storage_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<PortableNativeQsaComposition> composition_;
 static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
public:
 using Roles=std::array<PortableTensorView,14>;
 // Requested device reservations, summed once before uploads; these do not
 // establish physical admission or include backend-private/driver allocations.
 struct Budgets {size_t host,device,roles,qkv,projection,norm,indexer,select,attention,wo,hc;PortableNativeHcComposed::ComponentCaps hc_caps;};
 // Requested host payload allowance for component scratch/readback, role
 // comparison and diagnostics plus the composition and HC bounds. Excludes
 // mapped roles, caller-retained results, allocator and backend internals; it
 // is not a process RSS or physical-device admission bound.
 static size_t host_bound(uint32_t cells,size_t prefix,PortableHcReadSelection selection={}){
  return ple_host_add(32u<<20,ple_host_add(PortableNativeQsaComposition::host_bound(cells),NativeFfnHcFactory::selected_host_bound(prefix,selection)));
 }
 static Roles from_gguf(const GgufModel&m,int layer){
  need(layer>=3&&layer<48&&layer%4==3,"QSA owner: layer");
  const char* names[]={"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight","indexer.k_proj.weight","indexer.q_proj.weight","attn_q_norm.weight","attn_k_norm.weight","indexer.q_norm.weight","indexer.k_norm.weight","hc_attn_norm.weight","hc_attn_inject.weight","hc_attn_down.weight","hc_attn_up.weight"};Roles r;
  for(size_t i=0;i<r.size();++i){auto name="blk."+std::to_string(layer)+"."+names[i];size_t shard=0;auto*t=m.find(name,&shard);need(t&&m.in_bounds(*t,shard),"QSA owner: missing/out-of-bounds role");r[i]={name,{},t->type,m.shard(shard).tensor_data(*t),size_t(tensor_payload_bytes(*t))};for(auto d:t->shape){need(d<=INT64_MAX,"QSA owner: dimension overflow");r[i].shape.push_back(int64_t(d));}}
  return r;
 }
 static void validate(int layer,const Roles&r){
  need(layer>=3&&layer<48&&layer%4==3,"QSA owner: layer");
  const char* names[]={"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight","indexer.k_proj.weight","indexer.q_proj.weight","attn_q_norm.weight","attn_k_norm.weight","indexer.q_norm.weight","indexer.k_norm.weight","hc_attn_norm.weight","hc_attn_inject.weight","hc_attn_down.weight","hc_attn_up.weight"};
  const std::array<std::vector<int64_t>,14> shapes{{{2560,12288},{2560,512},{2560,512},{6144,2560},{2560,128},{2560,512},{256},{256},{128},{128},{10240},{10240,4},{10240,320},{320,10240}}};
  const std::array<size_t,14> count{2560u*12288,2560u*512,2560u*512,6144u*2560,2560u*128,2560u*512,256,256,128,128,10240,40960,3276800,3276800};
  for(size_t i=0;i<r.size();++i){bool q=i<4||i>=11;uint32_t type=q?8:i==4||i==5?30:0;size_t bytes=q?count[i]/32*34:type==30?count[i]*2:count[i]*4;
   need(r[i].name=="blk."+std::to_string(layer)+"."+names[i]&&r[i].data&&r[i].shape==shapes[i]&&r[i].type==type&&r[i].bytes==bytes,"QSA owner: role descriptor");
   if(q)qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(r[i].data),count[i]/32,34),r[i].name,true);
   if(type==30)qsa_indexer_bf16_raw_weights(r[i].data,count[i]);
   if(type==0){for(size_t j=0;j<count[i];++j){float x;std::memcpy(&x,static_cast<const uint8_t*>(r[i].data)+j*4,4);need(qsa_wo_normal_or_zero(x),"QSA owner: norm domain");}}
  }
 }
 PortableNativeQsaOwner(int layer,uint32_t cells,ggml_backend_t b,const Roles&r,const std::string&prefix,const Budgets& caps,PortableHcReadSelection selection={}){
  need(b&&!prefix.empty(),"QSA owner: backend/prefix");validate(layer,r);need(caps.host>=host_bound(cells,prefix.size()+3,selection),"QSA owner: host budget");
  size_t total=0;for(size_t n:{caps.roles,caps.qkv,caps.projection,caps.norm,caps.indexer,caps.select,caps.attention,caps.wo,caps.hc})total=ple_host_add(total,n);need(total<=caps.device,"QSA owner: aggregate device budget");
  need(PortableNativeQsaQkvRows::estimate(b)<=caps.qkv&&PortableNativeQsaIndexerRows::estimate(b)<=caps.projection&&PortableNativeQsaNormRope::estimate(b)<=caps.norm&&PortableNativeQsaIndexer::estimate(b,cells)<=caps.indexer&&PortableNativeQsaSelect::estimate(b,cells)<=caps.select&&PortableNativeQsaAttention::estimate(b,cells)<=caps.attention&&PortableNativeQsaWoRows::estimate(b)<=caps.wo&&NativeFfnHcFactory::selected_device_bound(b,caps.hc_caps,selection)<=caps.hc,"QSA owner: component budgets");
  context_.reset(ggml_init({16384,nullptr,true}));need(bool(context_),"QSA owner: metadata");std::array<ggml_tensor*,10>t{};
  t[0]=ggml_new_tensor_2d(context_.get(),GGML_TYPE_Q8_0,2560,12288);t[1]=ggml_new_tensor_2d(context_.get(),GGML_TYPE_Q8_0,2560,512);t[2]=ggml_new_tensor_2d(context_.get(),GGML_TYPE_Q8_0,2560,512);t[3]=ggml_new_tensor_2d(context_.get(),GGML_TYPE_Q8_0,6144,2560);
  t[4]=ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,2560*128/2);t[5]=ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,2560*512/2);
  for(size_t i=6;i<10;++i)t[i]=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,i<8?256:128);
  for(auto*x:t)need(x,"QSA owner: tensor metadata");auto n=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(b));need(n<=caps.roles,"QSA owner: role budget");
  storage_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),b));need(bool(storage_)&&ggml_backend_buffer_get_size(storage_.get())<=caps.roles,"QSA owner: role allocation");for(size_t i=0;i<t.size();++i)ggml_backend_tensor_set(t[i],r[i].data,0,r[i].bytes);
  auto qkv=std::make_unique<PortableNativeQsaQkvRows>(b,std::array<ggml_tensor*,3>{t[0],t[1],t[2]},std::array<const void*,3>{r[0].data,r[1].data,r[2].data},std::array<size_t,3>{r[0].bytes,r[1].bytes,r[2].bytes},prefix+".qkv",caps.qkv);
  auto proj=std::make_unique<PortableNativeQsaIndexerRows>(b,std::array<ggml_tensor*,2>{t[4],t[5]},std::array<const void*,2>{r[4].data,r[5].data},std::array<size_t,2>{r[4].bytes,r[5].bytes},prefix+".projection",caps.projection);
  // Q projection passes through q_norm; pooled K passes through k_norm.
  auto norm=std::make_unique<PortableNativeQsaNormRope>(b,std::array<ggml_tensor*,3>{t[6],t[7],t[8]},prefix+".norm",caps.norm,false);
  auto idx=std::make_unique<PortableNativeQsaIndexer>(b,t[9],cells,prefix+".indexer",caps.indexer,false);
  auto sel=std::make_unique<PortableNativeQsaSelect>(b,cells,prefix+".select",caps.select,false);
  auto att=std::make_unique<PortableNativeQsaAttention>(b,cells,prefix+".attention",caps.attention,false);
  auto wo=std::make_unique<PortableNativeQsaWoRows>(b,t[3],r[3].data,r[3].bytes,prefix+".wo",caps.wo);
  auto hc=NativeFfnHcFactory::create_selected(layer,b,std::vector<PortableTensorView>{r[10],r[11],r[12],r[13]},caps.hc,NativeFfnHcFactory::selected_host_bound(prefix.size()+3,selection),(prefix+".hc").c_str(),caps.hc_caps,selection,true);
  composition_=std::make_unique<PortableNativeQsaComposition>(layer,cells,PortableNativeQsaComposition::host_bound(cells),std::move(hc),std::move(qkv),std::move(proj),std::move(norm),std::move(idx),std::move(sel),std::move(att),std::move(wo));
 }
 PortableNativeQsaOwner(int layer,uint32_t cells,ggml_backend_t b,const GgufModel&m,const std::string&p,const Budgets& caps,PortableHcReadSelection selection={}):PortableNativeQsaOwner(layer,cells,b,from_gguf(m,layer),p,caps,selection){}
 PortableNativeQsaOwner(const PortableNativeQsaOwner&)=delete;PortableNativeQsaOwner&operator=(const PortableNativeQsaOwner&)=delete;
 PortableNativeQsaComposition& composition(){return *composition_;}
};
}
