#pragma once
// Internal QSA -> deferred FFN segment. Backend and runtime expert provider
// are borrowed; the GGUF mapping is needed only during construction. No model
// admission or backend-private/physical memory qualification is implied.
#include "portable_native_qsa_owner.hpp"
#include "portable_native_ffn_owner.hpp"
#include "portable_native_ffn_composition.hpp"
namespace strata::core::portable_detail {
class PortableNativeQsaLayerOwner {
 std::unique_ptr<PortableNativeFfnOwner> rows_;
 std::unique_ptr<PortableNativeQsaOwner> qsa_;
 std::unique_ptr<PortableNativeFfnComposition> ffn_;
public:
 struct Budgets {size_t host,device;PortableNativeQsaOwner::Budgets qsa;size_t ffn_device;PortableNativeHcComposed::ComponentCaps ffn_caps;PortableNativeFfnOwner::Budgets rows;};
 PortableNativeQsaLayerOwner(int layer,uint32_t cells,ggml_backend_t b,const GgufModel&m,const std::string&prefix,const Budgets& caps,PortableRouterMethod method=PortableRouterMethod::legacy_f32,PortableHcReadSelection selection={}){
  if(!b||prefix.empty()||layer==47)throw std::runtime_error("QSA layer owner: layer/backend");
  auto q=PortableNativeQsaOwner::from_gguf(m,layer);PortableNativeQsaOwner::validate(layer,q);
  auto r=PortableNativeFfnOwner::from_gguf(m,layer);PortableNativeFfnOwner::validate(layer,r,method);
  auto hc=NativeFfnHcFactory::from_gguf(m,layer);NativeFfnHcFactory::select(layer,hc,false);
  auto path=prefix+".ffn";auto host=ple_host_add(PortableNativeFfnComposition::host_bound(),NativeFfnHcFactory::selected_host_bound(path.size(),selection));
  if(ple_host_add(ple_host_add(caps.qsa.host,caps.rows.host),host)>caps.host||ple_host_add(ple_host_add(caps.qsa.device,caps.rows.device),caps.ffn_device)>caps.device||NativeFfnHcFactory::selected_device_bound(b,caps.ffn_caps,selection)>caps.ffn_device||caps.qsa.host<PortableNativeQsaOwner::host_bound(cells,(prefix+".qsa").size()+3,selection)||NativeFfnHcFactory::selected_device_bound(b,caps.qsa.hc_caps,selection)>caps.qsa.hc)throw std::runtime_error("QSA layer owner: aggregate budget");
  rows_=std::make_unique<PortableNativeFfnOwner>(layer,b,r,prefix+".rows",caps.rows,method);
  qsa_=std::make_unique<PortableNativeQsaOwner>(layer,cells,b,q,prefix+".qsa",caps.qsa,selection);
  ffn_=std::make_unique<PortableNativeFfnComposition>(qsa_->composition(),b,hc,rows_->rows(),caps.ffn_device,host,path.c_str(),caps.ffn_caps,selection);
 }
 PortableNativeQsaLayerOwner(const PortableNativeQsaLayerOwner&)=delete;PortableNativeQsaLayerOwner&operator=(const PortableNativeQsaLayerOwner&)=delete;
 PortableRouterMethod router_method()const{return rows_->router_method();}
 PortableNativeQsaComposition& attention(){return qsa_->composition();}PortableNativeFfnComposition& ffn(){return *ffn_;}
};
}
