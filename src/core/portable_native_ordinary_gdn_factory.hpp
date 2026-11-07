#pragma once
// Internal GDN components and ordinary layer owner. Not model admission. The
// selected backend is borrowed and must outlive both. Source views/mapping need
// only live through construction. A transferred core borrows component role
// storage and must be destroyed before the components. Compositions are exclusive.
#include "portable_native_gdn_input_factory.hpp"
#include "portable_native_ordinary_gdn_composition.hpp"
#include "portable_native_ffn_hc_factory.hpp"

namespace strata::core::portable_detail {
class PortableNativeGdnComponents {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    // Input/output and any transferred core borrow the uploaded role tensors.
    // The enclosing owner must destroy the composition before these components.
    Context context_{nullptr,ggml_free};
    Buffer storage_{nullptr,ggml_backend_buffer_free};
    std::unique_ptr<PortableNativeGdnInputOwner> input_;
    std::unique_ptr<PortableNativeGdnOutputRows> output_;
    std::unique_ptr<PortableNativeGdnCoreRows> core_;
    static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
public:
    // Reservations for disjoint requested device payloads. HC includes its
    // temporary upload and retained copy at construction. They are summed
    // before any role upload. These are not driver/private-workspace bounds.
    struct Budgets {
        size_t host,device,input,roles,core,output,hc;
        PortableNativeGdnCoreRows::Caps core_caps;
        PortableNativeHcComposed::ComponentCaps hc_caps;
    };
    using Roles=std::array<PortableTensorView,13>;
    static size_t host_bound(size_t prefix_bytes,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1) {
        size_t n=PortableNativeGdnInputOwner::host_bound(method);
        for(size_t v:{size_t(16384+245760),PortableNativeGdnCoreRows::host_bound(),
                      method==PortableGdnProjectionMethod::native_f16?PortableNativeGdnF16Rows::host_bound():PortableNativeGdnOutputRows::host_bound(),PortableNativeOrdinaryGdnComposition::host_bound(),
                      NativeFfnHcFactory::host_bound(prefix_bytes)}) n=ple_host_add(n,v);
        return n;
    }
    static void validate(int layer,const Roles& roles,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1) {
        need(method==PortableGdnProjectionMethod::legacy_q8_1||method==PortableGdnProjectionMethod::native_f16,"ordinary GDN factory: method");
        need(layer>=0&&layer<47&&layer%4!=3,"ordinary GDN factory: layer");
        const std::array<const char*,13> suffix{"attn_qkv.weight","attn_gate.weight","ssm_out.weight",
            "ssm_conv1d.weight","ssm_alpha.weight","ssm_beta.weight","ssm_dt.bias","ssm_a","ssm_norm.weight",
            "hc_attn_norm.weight","hc_attn_inject.weight","hc_attn_down.weight","hc_attn_up.weight"};
        const std::array<std::vector<int64_t>,13> shape{{{2560,10240},{2560,6144},{6144,2560},
            {4,10240},{2560,48},{2560,48},{48},{48},{128},
            {10240},{10240,4},{10240,320},{320,10240}}};
        const std::array<size_t,13> elements{2560u*10240,2560u*6144,6144u*2560,40960,122880,122880,48,48,128,10240,40960,3276800,3276800};
        for(size_t i=0;i<roles.size();++i){const auto& r=roles[i];const bool q=i<3||i==4||i==5||i>=10;
            need(r.name=="blk."+std::to_string(layer)+"."+suffix[i] && r.data && r.shape==shape[i] && r.type==(q?8u:0u) &&
                 r.bytes==(q?elements[i]/32*34:elements[i]*4),"ordinary GDN factory: role descriptor");}
        for(size_t i:{size_t(0),size_t(1),size_t(4),size_t(5),size_t(10),size_t(11),size_t(12)})
            qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(roles[i].data),elements[i]/32,34),roles[i].name,true);
        qsa_wo_weight_domain(roles[2].data,roles[2].bytes,roles[2].name);
        if(method==PortableGdnProjectionMethod::native_f16)for(size_t i=0;i<3;++i)PortableNativeGdnF16Rows::validate_role(roles[i].data,roles[i].bytes);
    }
    static Roles from_gguf(const GgufModel& model,int layer){
        need(layer>=0&&layer<47&&layer%4!=3,"ordinary GDN factory: layer");
        const std::array<const char*,13> suffix{"attn_qkv.weight","attn_gate.weight","ssm_out.weight",
            "ssm_conv1d.weight","ssm_alpha.weight","ssm_beta.weight","ssm_dt.bias","ssm_a","ssm_norm.weight",
            "hc_attn_norm.weight","hc_attn_inject.weight","hc_attn_down.weight","hc_attn_up.weight"};
        Roles result;
        for(size_t i=0;i<result.size();++i){auto name="blk."+std::to_string(layer)+"."+suffix[i];size_t shard=0;auto* t=model.find(name,&shard);
            need(t&&model.in_bounds(*t,shard),"ordinary GDN factory: missing/out-of-bounds GGUF role");
            result[i]={name,{},t->type,model.shard(shard).tensor_data(*t),size_t(tensor_payload_bytes(*t))};
            for(auto d:t->shape){need(d<=INT64_MAX,"ordinary GDN factory: dimension overflow");result[i].shape.push_back(int64_t(d));}}
        return result;
    }
    PortableNativeGdnComponents(int layer,ggml_backend_t backend,const Roles& roles,const std::string& prefix,const Budgets& b,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1){
        need(backend&&!prefix.empty(),"ordinary GDN factory: backend/prefix");
        validate(layer,roles,method);
        need(b.host>=host_bound(prefix.size()+3,method),"ordinary GDN factory: host budget");
        size_t total=0;for(size_t v:{b.input,b.roles,b.core,b.output,b.hc})total=ple_host_add(total,v);
        need(total<=b.device,"ordinary GDN factory: aggregate device budget");
        // The output tensor shares the whole core-role upload buffer. Charge
        // its full role cap again in the F16 projection envelope; the row
        // estimate already counts an exact-size output role, so this is a
        // conservative requested bound, not a physical-residency claim.
        size_t output_bound=PortableNativeGdnOutputRows::estimate(backend,method);
        if(method==PortableGdnProjectionMethod::native_f16)output_bound=ple_host_add(output_bound,b.roles);
        need(PortableNativeGdnInputOwner::device_bound(backend,method)<=b.input &&
             PortableNativeGdnCoreRows::estimate(backend,b.core_caps)<=b.core && output_bound<=b.output &&
             NativeFfnHcFactory::device_bound(backend,b.hc_caps)<=b.hc, "ordinary GDN factory: executor budgets");
        // Output Q8 and six core roles in native order. The AB matrices are
        // compat Q8->BF16, stored as packed I32 exactly as the native kernels expect.
        context_.reset(ggml_init({16384,nullptr,true}));need(bool(context_),"ordinary GDN factory: metadata");
        std::array<ggml_tensor*,7> t{};
        t[0]=ggml_new_tensor_2d(context_.get(),GGML_TYPE_Q8_0,6144,2560);
        const size_t counts[]={40960,61440,61440,48,48,128};
        for(size_t i=0;i<6;++i)t[i+1]=ggml_new_tensor_1d(context_.get(),i==1||i==2?GGML_TYPE_I32:GGML_TYPE_F32,counts[i]);
        for(auto* x:t)need(x,"ordinary GDN factory: tensor metadata");
        auto upload=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(backend));
        need(upload<=b.roles,"ordinary GDN factory: role budget");
        storage_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));need(bool(storage_),"ordinary GDN factory: role allocation");
        need(ggml_backend_buffer_get_size(storage_.get())<=b.roles,"ordinary GDN factory: actual role budget");
        ggml_backend_tensor_set(t[0],roles[2].data,0,roles[2].bytes);
        for(size_t i=0;i<6;++i){const auto& r=roles[i+3];if(i==1||i==2){auto x=native_hc_q8_bf16(static_cast<const uint8_t*>(r.data),r.bytes,122880);ggml_backend_tensor_set(t[i+1],x.data(),0,x.size()*2);}else ggml_backend_tensor_set(t[i+1],r.data,0,r.bytes);}
        input_=std::make_unique<PortableNativeGdnInputOwner>(layer,backend,std::array<PortableTensorView,2>{roles[0],roles[1]},prefix+".input",PortableNativeGdnInputOwner::host_bound(method),b.input,method);
        output_=std::make_unique<PortableNativeGdnOutputRows>(backend,t[0],roles[2].data,roles[2].bytes,prefix+".output",b.output,method);
        std::array<ggml_tensor*,6> core_roles{t[1],t[2],t[3],t[4],t[5],t[6]};
        core_=std::make_unique<PortableNativeGdnCoreRows>(backend,core_roles,b.core_caps,b.core,prefix+".core");
    }
    PortableNativeGdnInputRows& input(){return input_->rows();}
    PortableNativeGdnOutputRows& output(){return *output_;}
    PortableGdnProjectionMethod input_method()const{return input_->rows().method();}
    PortableGdnProjectionMethod output_method()const{return output_->method();}
    std::unique_ptr<PortableNativeGdnCoreRows> take_core(){need(bool(core_),"GDN components: core already transferred");return std::move(core_);}
    PortableNativeGdnComponents(const PortableNativeGdnComponents&)=delete;
    PortableNativeGdnComponents& operator=(const PortableNativeGdnComponents&)=delete;
};

class PortableNativeOrdinaryGdnOwner {
    std::unique_ptr<PortableNativeGdnComponents> components_;
    std::unique_ptr<PortableNativeOrdinaryGdnComposition> composition_;
    static void layer_check(int layer){if(layer<2||layer>=47||layer%4==3)throw std::runtime_error("ordinary GDN factory: layer");}
public:
    using Roles=PortableNativeGdnComponents::Roles;
    using Budgets=PortableNativeGdnComponents::Budgets;
    static size_t host_bound(size_t n,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1){return PortableNativeGdnComponents::host_bound(n,method);}
    static void validate(int layer,const Roles& roles,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1){layer_check(layer);PortableNativeGdnComponents::validate(layer,roles,method);}
    static Roles from_gguf(const GgufModel& m,int layer){layer_check(layer);return PortableNativeGdnComponents::from_gguf(m,layer);}
    PortableNativeOrdinaryGdnOwner(int layer,ggml_backend_t backend,const Roles& roles,const std::string& prefix,const Budgets& b,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1){
        validate(layer,roles,method);
        components_=std::make_unique<PortableNativeGdnComponents>(layer,backend,roles,prefix,b,method);
        std::vector<PortableTensorView> hc{roles[9],roles[10],roles[11],roles[12]};
        auto attention=NativeFfnHcFactory::create(layer,backend,hc,b.hc,NativeFfnHcFactory::host_bound(prefix.size()+3),(prefix+".hc").c_str(),{},b.hc_caps,false,false,true);
        composition_=std::make_unique<PortableNativeOrdinaryGdnComposition>(layer,std::move(attention),components_->input(),components_->take_core(),components_->output(),PortableNativeOrdinaryGdnComposition::host_bound());
    }
    PortableNativeOrdinaryGdnOwner(int layer,ggml_backend_t backend,const GgufModel& model,const std::string& prefix,const Budgets& b,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1)
      :PortableNativeOrdinaryGdnOwner(layer,backend,from_gguf(model,layer),prefix,b,method){}
    PortableNativeOrdinaryGdnOwner(const PortableNativeOrdinaryGdnOwner&)=delete;
    PortableNativeOrdinaryGdnOwner& operator=(const PortableNativeOrdinaryGdnOwner&)=delete;
    PortableGdnProjectionMethod input_method()const{return components_->input_method();}
    PortableGdnProjectionMethod output_method()const{return components_->output_method();}
    PortableNativeOrdinaryGdnComposition& composition(){return *composition_;}
};
}
