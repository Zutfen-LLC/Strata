#pragma once
// Selected prefill HC composition. The wrapped composed owner remains the sole
// publication authority. The final head does not use this selection.
#include "portable_native_hc_composed.hpp"
#include "portable_native_hc_bf16_products.hpp"
namespace strata::core::portable_detail {
class PortableNativeHcPrefillSelection {
 using HC=PortableNativeHcComposed;
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 struct Images {ggml_tensor *normalized,*normalized_bf,*low,*low_bf,*mixed_bf,*mixed_half;};
 struct State {
  ggml_backend_t backend;const ggml_tensor* norm_weight=nullptr;
  std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> pending{nullptr,strata_vk_native_hc_pending_destroy};
  std::unique_ptr<StrataVkNativeHcPrefill,decltype(&strata_vk_native_hc_prefill_destroy)> scalar{nullptr,strata_vk_native_hc_prefill_destroy};
  std::unique_ptr<PortableNativeHcBf16Products> products;
  Context ctx{nullptr,ggml_free};Buffer buffer{nullptr,ggml_backend_buffer_free};
  std::array<Images,4> images{};std::array<const HC::Bank*,4> keys{};
  explicit State(ggml_backend_t b):backend(b){}
  Images& image(const HC::Bank& b){for(size_t i=0;i<4;++i)if(keys[i]==&b)return images[i];for(size_t i=0;i<4;++i)if(!keys[i]&&ggml_nbytes(images[i].normalized)==ggml_nbytes(b.normalized)){keys[i]=&b;return images[i];}throw std::runtime_error("HC prefill: image banks exhausted");}
  void norm(const HC::Bank& b,uint32_t rows,const ggml_tensor* nw){auto& im=image(b);strata_vk_native_hc_prefill_run(scalar.get(),0,rows,{b.residual,const_cast<ggml_tensor*>(nw),b.normalized,im.normalized,im.normalized_bf,b.rs});}
  void mix(const HC::Bank& b){auto& im=image(b);uint32_t rows=uint32_t(ggml_nbytes(b.normalized)/40960);products->down_image(im.normalized,rows);ggml_backend_tensor_copy(products->output(0),b.low_pre);
   strata_vk_native_hc_prefill_run(scalar.get(),1,rows,{b.low_pre,nullptr,b.low,im.low,im.low_bf,nullptr});products->up_image(im.low);ggml_backend_tensor_copy(products->output(1),b.up_pre);
   products->inject();ggml_backend_tensor_copy(products->output(2),b.inject);strata_vk_native_hc_prefill_run(scalar.get(),2,rows,{b.normalized,b.up_pre,b.mixed,nullptr,im.mixed_bf,im.mixed_half});}
 };
 static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
 static size_t source_bytes(const std::array<const ggml_tensor*,4>& roles){size_t n=0;for(size_t i=0;i<4;++i){auto*t=roles[i];need(t&&t->buffer,"HC prefill: source");bool seen=false;for(size_t j=0;j<i;++j)seen|=roles[j]->buffer==t->buffer;if(!seen)n=ple_host_add(n,ggml_backend_buffer_get_size(t->buffer));}return n;}
 static Context metadata(){Context c(ggml_init({16384,nullptr,true}),ggml_free);need(bool(c),"HC prefill: image metadata");return c;}
 static void make_images(ggml_context* c,std::array<Images,4>& all){for(size_t i=0;i<4;++i){int rows=int(i%2)+1;auto f=[&](ggml_type type,int n){auto*t=ggml_new_tensor_2d(c,type,n,rows);need(t,"HC prefill: image tensor");return t;};all[i]={f(GGML_TYPE_F32,10240),f(GGML_TYPE_I32,5120),f(GGML_TYPE_F32,320),f(GGML_TYPE_I32,160),f(GGML_TYPE_I32,1280),f(GGML_TYPE_I32,1280)};}}
public:
 using Caps=HC::PrefillCaps;
 // Conservative requested payload envelope, including complete source buffers,
 // the copied role bank twice (once in HC and once as borrowed product storage),
 // graph copies, image banks and both scalar/pending component caps. Metadata,
 // scan scratch and role upload/readback are host-only. No physical or backend
 // private cost is bounded. Caller-owned consumers remain outside this envelope.
 static size_t host_bound(size_t prefix){return ple_host_add(ple_host_add(HC::host_bound(2,prefix),PortableNativeHcBf16Products::host_bound()),ple_host_add(2u<<20,ple_host_mul(prefix,4)));}
 static size_t device_bound_with_source(ggml_backend_t backend,const std::array<const ggml_tensor*,4>& roles,Caps caps,size_t source){
  need(backend,"HC prefill: backend");
  Context status_ctx(ggml_init({4096,nullptr,true}),ggml_free);need(bool(status_ctx),"HC prefill: status metadata");need(ggml_new_tensor_1d(status_ctx.get(),GGML_TYPE_I32,128),"HC prefill: status tensor");
  const auto status_bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(status_ctx.get(),ggml_backend_get_default_buffer_type(backend));need(caps.pending>=status_bytes&&caps.scalar>=status_bytes,"HC prefill: component caps");
  const size_t hc_tensors=HC::tensor_bound(backend,roles);
  Context rc(ggml_init({8192,nullptr,true}),ggml_free);need(bool(rc),"HC prefill: role metadata");for(auto*t:roles)need(ggml_dup_tensor(rc.get(),t),"HC prefill: role metadata");auto type=ggml_backend_get_default_buffer_type(backend);size_t retained=ggml_backend_alloc_ctx_tensors_from_buft_size(rc.get(),type);
  auto ic=metadata();std::array<Images,4> tmp{};make_images(ic.get(),tmp);size_t image_bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(ic.get(),type);
  size_t total=ple_host_add(source,hc_tensors);total=ple_host_add(total,PortableNativeHcBf16Products::device_bound(backend,retained));total=ple_host_add(total,image_bytes);return ple_host_add(total,ple_host_add(caps.pending,caps.scalar));
 }
 static size_t device_bound(ggml_backend_t backend,const std::array<const ggml_tensor*,4>& roles,Caps caps){return device_bound_with_source(backend,roles,caps,source_bytes(roles));}
 static std::unique_ptr<HC> create(ggml_backend_t backend,const std::array<const ggml_tensor*,4>& roles,size_t host_budget,size_t device_budget,const char* prefix,Caps caps){
  need(prefix&&*prefix,"HC prefill: prefix");size_t len=std::strlen(prefix);need(host_bound(len)<=host_budget,"HC prefill: host budget");need(device_bound(backend,roles,caps)<=device_budget,"HC prefill: device budget");
  auto state=std::make_shared<State>(backend);HC::Dispatch dispatch;
  dispatch.pending=[state](const HC::Bank&b){strata_vk_native_hc_pending_run(state->pending.get(),b.input,b.block,b.previous,b.residual,b.pending_gate,b.pending_exp);};
  dispatch.norm=[state](const HC::Bank&b,uint32_t rows){state->norm(b,rows,state->norm_weight);};
  dispatch.mix=[state](const HC::Bank&b){state->mix(b);};
  auto owner=std::make_unique<HC>(backend,roles[0],roles[1],roles[2],roles[3],device_budget,nullptr,std::move(dispatch),host_budget);
  auto bank=owner->role_bank();state->norm_weight=bank->role(0);
  state->ctx=metadata();make_images(state->ctx.get(),state->images);
  state->buffer.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(),backend));need(bool(state->buffer),"HC prefill: image allocation");
  std::string p(prefix);state->pending.reset(strata_vk_native_hc_pending_create(backend,(p+".pending.spv").c_str(),caps.pending));need(bool(state->pending)&&strata_vk_native_hc_pending_bytes(state->pending.get())<=caps.pending,"HC prefill: pending cap");
  state->scalar.reset(strata_vk_native_hc_prefill_create(backend,(p+".prefill.spv").c_str(),caps.scalar));need(bool(state->scalar)&&strata_vk_native_hc_prefill_bytes(state->scalar.get())<=caps.scalar,"HC prefill: scalar cap");
  auto product_bound=PortableNativeHcBf16Products::device_bound(backend,ggml_backend_buffer_get_size(bank->role(0)->buffer));
  state->products=std::make_unique<PortableNativeHcBf16Products>(backend,std::array<const ggml_tensor*,3>{bank->role(2),bank->role(3),bank->role(1)},PortableNativeHcBf16Products::host_bound(),product_bound);
  owner->selected_bytes_=ple_host_add(ple_host_add(state->products->device_bytes(),ggml_backend_buffer_get_size(state->buffer.get())),ple_host_add(strata_vk_native_hc_pending_bytes(state->pending.get()),strata_vk_native_hc_prefill_bytes(state->scalar.get())));
  need(ple_host_add(owner->bytes(),source_bytes(roles))<=device_budget,"HC prefill: actual device budget");
  owner->read_method_=PortableHcReadMethod::baseline_prefill_bf16;
  return owner;
 }
};
inline std::unique_ptr<PortableNativeHcComposed> PortableNativeHcComposed::create_prefill(ggml_backend_t backend,
 const std::array<const ggml_tensor*,4>& roles,size_t host_budget,size_t device_budget,const char* prefix,PrefillCaps caps){
 return PortableNativeHcPrefillSelection::create(backend,roles,host_budget,device_budget,prefix,caps);
}
}
