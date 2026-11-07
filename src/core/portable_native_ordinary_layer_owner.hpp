#pragma once
// An owning ordinary GDN -> deferred FFN segment for the model row chain.
// The backend and the already-configured per-layer FFN rows (and their provider
// dependencies) are borrowed exclusively and outlive this segment. The GGUF
// mapping and views are only borrowed during construction. This is not whole
// model admission and does not account for borrowed rows or backend internals.
#include "portable_native_ordinary_gdn_factory.hpp"
#include "portable_native_ffn_composition.hpp"
namespace strata::core::portable_detail {
class PortableNativeOrdinaryLayerOwner {
    // The FFN borrows the GDN composition and must be destroyed first.
    std::unique_ptr<PortableNativeOrdinaryGdnOwner> gdn_;
    std::unique_ptr<PortableNativeFfnComposition> ffn_;
public:
    struct Budgets {
        size_t host,device;
        PortableNativeOrdinaryGdnOwner::Budgets gdn;
        size_t ffn_device;
        PortableNativeHcComposed::ComponentCaps ffn_caps;
    };
    PortableNativeOrdinaryLayerOwner(int layer,ggml_backend_t backend,const GgufModel& model,
        PortableNativeFfnRows& rows,const std::string& prefix,const Budgets& b) {
        if(!backend||prefix.empty()||rows.layer()!=layer)throw std::runtime_error("ordinary layer owner: layer/backend/rows");
        auto roles=PortableNativeOrdinaryGdnOwner::from_gguf(model,layer);
        PortableNativeOrdinaryGdnOwner::validate(layer,roles);
        auto hc=NativeFfnHcFactory::from_gguf(model,layer);
        NativeFfnHcFactory::select(layer,hc,false);
        auto path=prefix+".ffn";
        const auto ffn_host=ple_host_add(PortableNativeFfnComposition::host_bound(),NativeFfnHcFactory::host_bound(path.size()));
        if(ple_host_add(b.gdn.host,ffn_host)>b.host || ple_host_add(b.gdn.device,b.ffn_device)>b.device ||
           NativeFfnHcFactory::device_bound(backend,b.ffn_caps)>b.ffn_device)
            throw std::runtime_error("ordinary layer owner: aggregate budget");
        gdn_=std::make_unique<PortableNativeOrdinaryGdnOwner>(layer,backend,roles,prefix+".gdn",b.gdn);
        ffn_=std::make_unique<PortableNativeFfnComposition>(gdn_->composition(),backend,hc,rows,b.ffn_device,ffn_host,path.c_str(),b.ffn_caps);
    }
    PortableNativeOrdinaryLayerOwner(const PortableNativeOrdinaryLayerOwner&)=delete;
    PortableNativeOrdinaryLayerOwner& operator=(const PortableNativeOrdinaryLayerOwner&)=delete;
    PortableNativeOrdinaryGdnComposition& attention(){return gdn_->composition();}
    PortableNativeFfnComposition& ffn(){return *ffn_;}
};
}
