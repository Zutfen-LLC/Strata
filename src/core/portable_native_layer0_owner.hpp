#pragma once
// Internal entry owner. Backend is borrowed; the shared GGUF model keeps the
// embedding mapping alive through every run and until composition destruction.
// Budgets are requested envelopes, not physical or whole-model admission.
#include "portable_native_ordinary_gdn_factory.hpp"
#include "portable_native_ffn_owner.hpp"
#include "portable_native_layer0_composition.hpp"

namespace strata::core::portable_detail {
class PortableNativeLayer0Owner {
    std::shared_ptr<const GgufModel> model_;
    std::unique_ptr<PortableNativeFfnOwner> rows_;
    std::unique_ptr<PortableNativeGdnComponents> gdn_;
    std::unique_ptr<PortableNativeLayer0Composition> composition_;
    static void need(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
public:
    struct Budgets {
        size_t host,device;
        PortableNativeGdnComponents::Budgets gdn;
        PortableNativeFfnOwner::Budgets rows;
        size_t ffn_hc,write;
        PortableNativeHcComposed::ComponentCaps ffn_caps;
    };
    PortableNativeLayer0Owner(std::shared_ptr<const GgufModel> model,ggml_backend_t backend,
                              const std::string& prefix,const Budgets& b,PortableRouterMethod method=PortableRouterMethod::legacy_f32,PortableGdnProjectionMethod gdn_method=PortableGdnProjectionMethod::legacy_q8_1,PortableGdnAbMethod ab_method=PortableGdnAbMethod::legacy_fused):model_(std::move(model)) {
        need(model_&&backend&&!prefix.empty(),"layer0 owner: model/backend/prefix");
        auto g=PortableNativeGdnComponents::from_gguf(*model_,0);PortableNativeGdnComponents::validate(0,g,gdn_method);
        auto r=PortableNativeFfnOwner::from_gguf(*model_,0);PortableNativeFfnOwner::validate(0,r,method);
        auto f=NativeFfnHcFactory::from_gguf(*model_,0);NativeFfnHcFactory::select(0,f,false);
        size_t shard=0;auto* t=model_->find("token_embd.weight",&shard);
        need(t&&model_->in_bounds(*t,shard),"layer0 owner: embedding missing/out of bounds");
        PortableTensorView embedding{"token_embd.weight",{},t->type,model_->shard(shard).tensor_data(*t),size_t(tensor_payload_bytes(*t))};
        for(auto d:t->shape){need(d<=INT64_MAX,"layer0 owner: embedding dimension");embedding.shape.push_back(int64_t(d));}
        (void)native_layer0_embedding(embedding,{0});
        auto path=prefix+".ffn-hc";
        auto fh=NativeFfnHcFactory::host_bound(path.size());
        size_t h=ple_host_add(ple_host_add(b.gdn.host,b.rows.host),ple_host_add(fh,PortableNativeLayer0Composition::host_bound()));
        size_t d=ple_host_add(ple_host_add(b.gdn.device,b.rows.device),ple_host_add(b.ffn_hc,b.write));
        need(h<=b.host&&d<=b.device&&NativeFfnHcFactory::device_bound(backend,b.ffn_caps)<=b.ffn_hc,"layer0 owner: aggregate budget");
        PortableNativeGdnComponents::preflight(backend,(prefix+".gdn").size()+3,b.gdn,gdn_method,ab_method);
        rows_=std::make_unique<PortableNativeFfnOwner>(0,backend,r,prefix+".rows",b.rows,method);
        gdn_=std::make_unique<PortableNativeGdnComponents>(0,backend,g,prefix+".gdn",b.gdn,gdn_method,ab_method);
        std::vector<PortableTensorView> ah{g[9],g[10],g[11],g[12]};
        auto attention=NativeFfnHcFactory::create(0,backend,ah,b.gdn.hc,NativeFfnHcFactory::host_bound(prefix.size()+7),(prefix+".attn").c_str(),{},b.gdn.hc_caps,true);
        auto ffn=NativeFfnHcFactory::create(0,backend,f,b.ffn_hc,fh,path.c_str(),{},b.ffn_caps);
        composition_=std::make_unique<PortableNativeLayer0Composition>(embedding,std::move(attention),gdn_->input(),gdn_->take_core(),gdn_->output(),std::move(ffn),rows_->rows(),backend,b.write,PortableNativeLayer0Composition::host_bound(),(prefix+".write").c_str());
    }
    PortableNativeLayer0Owner(const PortableNativeLayer0Owner&)=delete;
    PortableNativeLayer0Owner& operator=(const PortableNativeLayer0Owner&)=delete;
    PortableRouterMethod router_method()const{return rows_->router_method();}
    PortableGdnProjectionMethod gdn_input_method()const{return gdn_->input_method();}
    PortableGdnProjectionMethod gdn_output_method()const{return gdn_->output_method();}
    PortableGdnAbMethod gdn_ab_method()const{return composition_->ab_method();}
    PortableNativeLayer0Composition& composition(){return *composition_;}
};
}
