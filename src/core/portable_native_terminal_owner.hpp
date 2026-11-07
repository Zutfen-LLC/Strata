#pragma once
// Internal layer-47 terminal. The backend and runtime fixed expert provider
// are borrowed and must outlive every operation. Mapped source roles need only
// survive construction: QSA, FFN and head copy their required storage. Terminal
// is destroyed first, before the QSA, FFN and head it borrows. This is not
// full-model admission or a physical/backend-private workspace bound.
#include "portable_native_qsa_owner.hpp"
#include "portable_native_ffn_owner.hpp"
#include "portable_native_terminal_composition.hpp"

namespace strata::core::portable_detail {
class PortableNativeTerminalOwner {
    std::unique_ptr<PortableNativeFfnOwner> rows_;
    std::unique_ptr<PortableNativeQsaOwner> qsa_;
    std::unique_ptr<PortableHead> head_;
    std::unique_ptr<PortableNativeTerminalComposition> terminal_;
    static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
public:
    using HeadRoles=std::array<PortableTensorView,4>;
    struct Budgets {
        size_t host,device,head_host,head_weight,head_graph,terminal_hc,terminal_write;
        PortableNativeQsaOwner::Budgets qsa;
        PortableNativeFfnOwner::Budgets rows;
        PortableNativeHcComposed::ComponentCaps terminal_caps;
    };
    // Conservative requested host payload for two head metadata arenas and
    // native head host allowance, simultaneous Q8->BF16 conversion/readback,
    // logits and small construction objects. Excludes mapping, caller-held
    // logits and backend/allocator internals; not physical admission.
    static size_t head_host_bound(size_t vocab){
        need(vocab&&vocab<=1048576,"terminal owner: vocabulary");
        return ple_host_add(40u<<20,ple_host_mul(vocab,2*sizeof(float)));
    }
    static HeadRoles from_gguf(const GgufModel& m){
        const char* names[]={"output_hc_norm.weight","output_hc_down.weight","output_hc_up.weight","output.weight"};HeadRoles r;
        for(size_t i=0;i<r.size();++i){size_t shard=0;auto*t=m.find(names[i],&shard);need(t&&m.in_bounds(*t,shard),"terminal owner: missing/out-of-bounds head role");
            r[i]={names[i],{},t->type,m.shard(shard).tensor_data(*t),size_t(tensor_payload_bytes(*t))};
            for(auto d:t->shape){need(d<=INT64_MAX,"terminal owner: head dimension overflow");r[i].shape.push_back(int64_t(d));}}
        return r;
    }
    static void validate_head(const HeadRoles& r,size_t vocab){
        need(vocab&&vocab<=1048576,"terminal owner: vocabulary");
        const char* names[]={"output_hc_norm.weight","output_hc_down.weight","output_hc_up.weight","output.weight"};
        const std::array<std::vector<int64_t>,4> shapes{{{10240},{10240,320},{320,10240},{2560,int64_t(vocab)}}};
        const std::array<size_t,4> bytes{10240u*4,3276800u/32*34,3276800u/32*34,vocab*80*34};
        for(size_t i=0;i<r.size();++i)need(r[i].name==names[i]&&r[i].data&&r[i].shape==shapes[i]&&r[i].type==(i?8u:0u)&&r[i].bytes==bytes[i],"terminal owner: head role descriptor");
        qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(r[3].data),vocab*80,34),r[3].name,true);
    }
    PortableNativeTerminalOwner(uint32_t cells,size_t vocab,ggml_backend_t backend,const GgufModel& model,const std::string& prefix,const Budgets& b,PortableRouterMethod method=PortableRouterMethod::legacy_f32,PortableHcReadSelection selection={}){
        need(backend&&!prefix.empty(),"terminal owner: backend/prefix");
        auto q=PortableNativeQsaOwner::from_gguf(model,47);PortableNativeQsaOwner::validate(47,q);
        auto r=PortableNativeFfnOwner::from_gguf(model,47);PortableNativeFfnOwner::validate(47,r,method);
        auto hc=NativeFfnHcFactory::from_gguf(model,47);NativeFfnHcFactory::select(47,hc,false);
        auto hr=from_gguf(model);validate_head(hr,vocab);std::vector<PortableTensorView> hv(hr.begin(),hr.end());
        auto hp=prefix+".head",tp=prefix+".terminal-hc";
        auto th=ple_host_add(ple_host_add(2u<<20,ple_host_mul(vocab,2*sizeof(float))),NativeFfnHcFactory::selected_host_bound(tp.size(),selection));
        need(b.head_host>=head_host_bound(vocab),"terminal owner: head host budget");
        // Head graph admission is mixed host/device, so reserve it in both
        // envelopes rather than claiming unproven workspace placement.
        size_t host=0;for(size_t n:{b.qsa.host,b.rows.host,b.head_host,b.head_graph,th})host=ple_host_add(host,n);
        size_t device=0;for(size_t n:{b.qsa.device,b.rows.device,b.head_weight,b.head_graph,b.terminal_hc,b.terminal_write})device=ple_host_add(device,n);
        need(host<=b.host&&device<=b.device&&NativeFfnHcFactory::selected_device_bound(backend,b.terminal_caps,selection)<=b.terminal_hc&&b.qsa.host>=PortableNativeQsaOwner::host_bound(cells,(prefix+".qsa").size()+3,selection)&&NativeFfnHcFactory::selected_device_bound(backend,b.qsa.hc_caps,selection)<=b.qsa.hc,"terminal owner: aggregate budget");
        PortableHeadGeometry geo{2560,4,320,int64_t(vocab),1e-6f};PortableFfnOptions opts;
        opts.accelerator=backend;opts.request_accelerator=true;opts.resident_dense_weights=true;opts.allow_cpu_fallback=false;
        opts.native_head_hc_t1=true;opts.native_head_vocab_t1=true;opts.native_head_hc_receipt_prefix=hp;
        opts.weight_budget=b.head_weight;opts.graph_budget=b.head_graph;
        need(PortableHead::weight_allocation_size(backend,geo,hv,true)<=b.head_weight&&PortableHead::admission(geo,hv,opts,false).graph_bound<=b.head_graph,"terminal owner: head budgets");
        rows_=std::make_unique<PortableNativeFfnOwner>(47,backend,r,prefix+".rows",b.rows,method);
        qsa_=std::make_unique<PortableNativeQsaOwner>(47,cells,backend,q,prefix+".qsa",b.qsa,selection);
        std::string error;head_=PortableHead::create(geo,hv,opts,error);if(!head_)throw std::runtime_error("terminal owner: head: "+error);
        terminal_=std::make_unique<PortableNativeTerminalComposition>(qsa_->composition(),backend,hc,rows_->rows(),*head_,b.terminal_hc,th,b.terminal_write,tp.c_str(),(prefix+".write").c_str(),b.terminal_caps,selection);
    }
    PortableNativeTerminalOwner(const PortableNativeTerminalOwner&)=delete;
    PortableNativeTerminalOwner& operator=(const PortableNativeTerminalOwner&)=delete;
    PortableRouterMethod router_method()const{return rows_->router_method();}
    PortableNativeQsaComposition& attention(){return qsa_->composition();}
    PortableNativeTerminalComposition& terminal(){return *terminal_;}
    PortableHead& head(){return *head_;}
};
}
