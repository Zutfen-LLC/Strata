#pragma once
// Internal single-backend construction only; not public session admission.
// Backend and each fixed expert provider passed to run are borrowed. The model,
// table and every component outlive the executor, which is destroyed first.
#include "portable_model_inventory.hpp"
#include "portable_native_layer0_owner.hpp"
#include "portable_native_ple1_owner.hpp"
#include "portable_native_ordinary_layer_owner.hpp"
#include "portable_native_qsa_layer_owner.hpp"
#include "portable_native_terminal_owner.hpp"
#include "portable_native_model_rows.hpp"
namespace strata::core::portable_detail {
class PortableNativeModelOwner {
    std::shared_ptr<const GgufModel> model_;
    std::shared_ptr<PortablePleTable> table_;
    std::unique_ptr<PortableNativeLayer0Owner> entry_;
    std::unique_ptr<PortableNativePle1Owner> ple_;
    std::array<std::unique_ptr<PortableNativeOrdinaryLayerOwner>,48> gdn_;
    std::array<std::unique_ptr<PortableNativeQsaLayerOwner>,48> qsa_;
    std::unique_ptr<PortableNativeTerminalOwner> terminal_;
    std::unique_ptr<PortableNativeModelRows> rows_;
    static void need(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
public:
    struct Budgets {
        size_t mapped,host,device,executor_host;
        PortableNativeLayer0Owner::Budgets entry;
        PortableNativePle1Owner::Budgets ple;
        PortableNativeOrdinaryLayerOwner::Budgets gdn;
        PortableNativeQsaLayerOwner::Budgets qsa;
        PortableNativeTerminalOwner::Budgets terminal;
    };
    struct Accounting {size_t source_files=0,table_mapping=0,table_owned=0,host_requested=0,device_requested=0,explicit_role_device=0,other_device_including_scratch=0,executor_host=0,result_allowance=0;};
private:
    Accounting accounting_;
    static PortableTensorView embedding(const GgufModel& m){
        size_t s=0;auto*t=m.find("token_embd.weight",&s);need(t&&m.in_bounds(*t,s),"model owner: embedding missing/out of bounds");
        PortableTensorView v{t->name,{},t->type,m.shard(s).tensor_data(*t),size_t(tensor_payload_bytes(*t))};
        for(auto d:t->shape){need(d<=INT64_MAX,"model owner: dimension overflow");v.shape.push_back(int64_t(d));}return v;
    }
    static void hc_check(int layer,const GgufModel& m){
        auto views=NativeFfnHcFactory::from_gguf(m,layer);auto selected=NativeFfnHcFactory::select(layer,views,false);
        for(size_t i=1;i<4;++i)qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(selected[i]->data),selected[i]->bytes/34,34),selected[i]->name,true);
    }
public:
    // Accounting is a conservative requested-cap sum, not an RSS or physical
    // admission proof. Mixed PLE and head graph envelopes are charged to both
    // host and device. Source files and the independently mapped PLE shard (or
    // small-table copy) are separately counted; backend-private allocations,
    // allocator/driver costs, provider and caller-retained results are excluded.
    PortableNativeModelOwner(std::shared_ptr<const GgufModel> model,std::shared_ptr<PortablePleTable> table,
        ggml_backend_t backend,uint32_t cells,const std::string& prefix,const Budgets& b,PortableRouterMethod method=PortableRouterMethod::legacy_f32,PortableGdnProjectionMethod gdn_method=PortableGdnProjectionMethod::legacy_q8_1,PortableGdnAbMethod ab_method=PortableGdnAbMethod::legacy_fused,PortableHcReadSelection selection={})
      :model_(std::move(model)),table_(std::move(table)){
        selection.validate();need(model_&&table_&&backend&&!prefix.empty(),"model owner: source/table/backend/prefix");
        auto arch=check_architecture(model_->meta());need(arch.empty(),"model owner: architecture: "+arch);
        ModelGeometry geometry;auto ge=read_geometry(model_->meta(),geometry);need(ge.empty()&&geometry.family==ModelFamily::Qwen4Exp&&geometry.n_layers==48&&geometry.n_expert==512&&geometry.n_expert_used==10,"model owner: geometry/expert contract");
        need(cells>=1&&cells<=4096,"model owner: capacity");portable_model_inventory(*model_);
        auto e=embedding(*model_);(void)native_layer0_embedding(e,{0});size_t vocab=size_t(e.shape[1]);need(vocab<=1048576,"model owner: vocabulary capacity");
        if(model_->meta().get("tokenizer.ggml.tokens"))need(size_t(geometry.n_vocab)==vocab,"model owner: tokenizer vocabulary");
        need(b.executor_host>=ple_host_add(64u<<10,ple_host_mul(vocab,2*sizeof(float))),"model owner: executor host budget");
        auto head=PortableNativeTerminalOwner::from_gguf(*model_);PortableNativeTerminalOwner::validate_head(head,vocab);
        for(size_t i=1;i<3;++i)qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(head[i].data),head[i].bytes/34,34),head[i].name,true);
        for(size_t i=0;i<10240;++i){float x;std::memcpy(&x,static_cast<const uint8_t*>(head[0].data)+i*4,4);native_hc_normal(x);}
        auto p=PortableNativePle1Owner::from_gguf(*model_);PortableNativePle1Owner::validate(p);
        qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(p[1].data),p[1].bytes/34,34),p[1].name,true);
        for(size_t i=2;i<5;++i)for(size_t j=0;j<10240;++j){float x;std::memcpy(&x,static_cast<const uint8_t*>(p[i].data)+j*4,4);native_hc_normal(x);}
        (void)PortableNativePleConv::native_weights(p[5].data,p[5].bytes,p[5].type);
        for(int i=0;i<48;++i){
            auto r=PortableNativeFfnOwner::from_gguf(*model_,i);PortableNativeFfnOwner::validate(i,r,method);hc_check(i,*model_);
            if(i%4==3){auto q=PortableNativeQsaOwner::from_gguf(*model_,i);PortableNativeQsaOwner::validate(i,q);}
            else {auto g=PortableNativeGdnComponents::from_gguf(*model_,i);PortableNativeGdnComponents::validate(i,g,gdn_method);}
        }
        for(size_t i=0;i<model_->size();++i){auto n=model_->shard(i).file_size();need(n<=SIZE_MAX,"model owner: file size overflow");accounting_.source_files=ple_host_add(accounting_.source_files,size_t(n));}
        auto mapped_table=table_->mapped_file_bytes();need(mapped_table<=SIZE_MAX,"model owner: table mapping overflow");accounting_.table_mapping=size_t(mapped_table);accounting_.table_owned=table_->owned_bytes();
        need(ple_host_add(accounting_.source_files,accounting_.table_mapping)<=b.mapped,"model owner: mapped budget");
        accounting_.host_requested=ple_host_add(accounting_.table_owned,ple_host_add(ple_host_add(b.entry.host,b.ple.total),ple_host_add(b.terminal.host,b.executor_host)));
        accounting_.device_requested=ple_host_add(ple_host_add(b.entry.device,b.ple.total),b.terminal.device);
        for(int i=2;i<=46;++i){accounting_.host_requested=ple_host_add(accounting_.host_requested,i%4==3?b.qsa.host:b.gdn.host);accounting_.device_requested=ple_host_add(accounting_.device_requested,i%4==3?b.qsa.device:b.gdn.device);}
        // Explicit role caps are only a subset: input/QSA/HC component caps
        // also include copied roles, so the remainder is mixed, not pure scratch.
        size_t roles=0;for(size_t n:{b.entry.gdn.roles,b.entry.rows.roles,b.ple.role_device,b.ple.gdn.roles,b.ple.rows.roles,b.terminal.qsa.roles,b.terminal.rows.roles,b.terminal.head_weight})roles=ple_host_add(roles,n);
        for(int i=2;i<=46;++i){roles=ple_host_add(roles,i%4==3?b.qsa.qsa.roles:b.gdn.gdn.roles);roles=ple_host_add(roles,i%4==3?b.qsa.rows.roles:b.gdn.rows.roles);}
        need(roles<=accounting_.device_requested,"model owner: role accounting");accounting_.explicit_role_device=roles;accounting_.other_device_including_scratch=accounting_.device_requested-roles;
        accounting_.executor_host=b.executor_host;accounting_.result_allowance=ple_host_mul(vocab,2*sizeof(float));
        need(accounting_.host_requested<=b.host&&accounting_.device_requested<=b.device,"model owner: aggregate budget");
        PortableNativeGdnComponents::preflight(backend,(prefix+".entry.gdn").size()+3,b.entry.gdn,gdn_method,ab_method,selection);
        PortableNativeGdnComponents::preflight(backend,(prefix+".ple1.gdn").size()+3,b.ple.gdn,gdn_method,ab_method,selection);
        for(int i=2;i<=46;++i)if(i%4!=3)PortableNativeGdnComponents::preflight(backend,(prefix+".layer"+std::to_string(i)+".gdn").size()+3,b.gdn.gdn,gdn_method,ab_method,selection);
        // Preflight every selected HC envelope before any owning layer allocates.
        // These are conservative requested sums; role uploads are charged again
        // where the selected product owner borrows their full buffers.
        if(selection.method==PortableHcReadMethod::baseline_prefill_bf16){
            auto fh=[&](size_t n){return NativeFfnHcFactory::selected_host_bound(n,selection);};
            auto fd=[&](PortableNativeHcComposed::ComponentCaps c){return NativeFfnHcFactory::selected_device_bound(backend,c,selection);};
            auto ep=prefix+".entry";auto eh=fh((ep+".ffn-hc").size());
            need(fd(b.entry.ffn_caps)<=b.entry.ffn_hc&&ple_host_add(ple_host_add(b.entry.gdn.host,b.entry.rows.host),ple_host_add(eh,PortableNativeLayer0Composition::host_bound()))<=b.entry.host,"model owner: entry selected HC budget");
            auto pp=prefix+".ple1";auto fhost=ple_host_add(PortableNativeFfnComposition::host_bound(),fh((pp+".ffn").size()));
            need(fd(b.ple.ffn_caps)<=b.ple.ffn_device,"model owner: PLE FFN selected HC budget");
            size_t pt=PortableNativePle1Owner::role_host_bound();for(size_t n:{b.ple.gdn.host,b.ple.gdn.device,b.ple.rows.host,b.ple.rows.device,b.ple.ple,b.ple.role_device,b.ple.ffn_device,fhost,PortableNativePleGdnComposition::host_bound()})pt=ple_host_add(pt,n);need(pt<=b.ple.total,"model owner: PLE selected host budget");
            using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;Context hcctx(ggml_init({8192,nullptr,true}),ggml_free);need(bool(hcctx),"model owner: PLE estimate metadata");
            std::array<const ggml_tensor*,4> ht{ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_F32,10240),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,20480),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,1638400)};
            PortableNativePleAdmission::require_budget(PortableNativePleAdmission::calculate(backend,ht,pp+".ple",(pp+".ple-hc").size(),b.ple.ple_caps,selection,b.ple.role_device),b.ple.ple);
            for(int i=2;i<=46;++i){auto path=prefix+".layer"+std::to_string(i);auto h=ple_host_add(PortableNativeFfnComposition::host_bound(),fh((path+".ffn").size()));
                if(i%4==3){const auto& q=b.qsa;need(fd(q.ffn_caps)<=q.ffn_device&&fd(q.qsa.hc_caps)<=q.qsa.hc&&q.qsa.host>=PortableNativeQsaOwner::host_bound(cells,(path+".qsa").size()+3,selection)&&ple_host_add(ple_host_add(q.qsa.host,q.rows.host),h)<=q.host,"model owner: QSA selected HC budget");}
                else {const auto& g=b.gdn;need(fd(g.ffn_caps)<=g.ffn_device&&ple_host_add(ple_host_add(g.gdn.host,g.rows.host),h)<=g.host,"model owner: GDN selected HC budget");}}
            auto tp=prefix+".terminal";const auto& t=b.terminal;size_t th=ple_host_add(ple_host_add(2u<<20,ple_host_mul(vocab,2*sizeof(float))),fh((tp+".terminal-hc").size()));
            size_t host=0;for(size_t n:{t.qsa.host,t.rows.host,t.head_host,t.head_graph,th})host=ple_host_add(host,n);
            need(host<=t.host&&fd(t.terminal_caps)<=t.terminal_hc&&fd(t.qsa.hc_caps)<=t.qsa.hc&&t.qsa.host>=PortableNativeQsaOwner::host_bound(cells,(tp+".qsa").size()+3,selection),"model owner: terminal selected HC budget");
        }
        entry_=std::make_unique<PortableNativeLayer0Owner>(model_,backend,prefix+".entry",b.entry,method,gdn_method,ab_method,selection);
        ple_=std::make_unique<PortableNativePle1Owner>(model_,table_,backend,prefix+".ple1",b.ple,method,gdn_method,ab_method,selection);
        std::array<PortableNativeModelRows::Layer,48> layers{};
        for(int i=2;i<=46;++i){auto path=prefix+".layer"+std::to_string(i);
            if(i%4==3){qsa_[i]=std::make_unique<PortableNativeQsaLayerOwner>(i,cells,backend,*model_,path,b.qsa,method,selection);layers[i].qsa=&qsa_[i]->attention();layers[i].ffn=&qsa_[i]->ffn();}
            else {gdn_[i]=std::make_unique<PortableNativeOrdinaryLayerOwner>(i,backend,*model_,path,b.gdn,method,gdn_method,ab_method,selection);layers[i].gdn=&gdn_[i]->attention();layers[i].ffn=&gdn_[i]->ffn();}}
        terminal_=std::make_unique<PortableNativeTerminalOwner>(cells,vocab,backend,*model_,prefix+".terminal",b.terminal,method,selection);
        rows_=std::make_unique<PortableNativeModelRows>(entry_->composition(),ple_->ple(),ple_->attention(),ple_->ffn(),layers,terminal_->attention(),terminal_->terminal(),b.executor_host);
    }
    static std::unique_ptr<PortableNativeModelOwner> from_model(std::shared_ptr<const GgufModel> model,ggml_backend_t backend,uint32_t cells,const std::string& prefix,const Budgets& b,PortableRouterMethod method=PortableRouterMethod::legacy_f32,PortableGdnProjectionMethod gdn_method=PortableGdnProjectionMethod::legacy_q8_1,PortableGdnAbMethod ab_method=PortableGdnAbMethod::legacy_fused,PortableHcReadSelection selection={}){
        selection.validate();need(bool(model),"model owner: missing model");std::string error;auto table=PortablePleTable::from_gguf(*model,error);need(bool(table),"model owner: "+error);
        return std::make_unique<PortableNativeModelOwner>(std::move(model),std::move(table),backend,cells,prefix,b,method,gdn_method,ab_method,selection);
    }
    PortableNativeModelOwner(const PortableNativeModelOwner&)=delete;PortableNativeModelOwner& operator=(const PortableNativeModelOwner&)=delete;
    std::array<PortableRouterMethod,48> router_methods()const{
        std::array<PortableRouterMethod,48> result{};result[0]=entry_->router_method();result[1]=ple_->router_method();
        for(int i=2;i<=46;++i)result[i]=i%4==3?qsa_[i]->router_method():gdn_[i]->router_method();
        result[47]=terminal_->router_method();return result;
    }
    // Slots at QSA layers are left at the legacy sentinel; only GDN layers
    // represent a projection object. Values come from the actual row owners.
    std::array<PortableGdnProjectionMethod,48> gdn_input_methods()const{
        std::array<PortableGdnProjectionMethod,48> result{};result[0]=entry_->gdn_input_method();result[1]=ple_->gdn_input_method();
        for(int i=2;i<=46;++i)if(i%4!=3)result[i]=gdn_[i]->gdn_input_method();return result;
    }
    std::array<PortableGdnProjectionMethod,48> gdn_output_methods()const{
        std::array<PortableGdnProjectionMethod,48> result{};result[0]=entry_->gdn_output_method();result[1]=ple_->gdn_output_method();
        for(int i=2;i<=46;++i)if(i%4!=3)result[i]=gdn_[i]->gdn_output_method();return result;
    }
    std::array<PortableGdnAbMethod,48> gdn_ab_methods()const{
        std::array<PortableGdnAbMethod,48> result{};result[0]=entry_->gdn_ab_method();result[1]=ple_->gdn_ab_method();
        for(int i=2;i<=46;++i)if(i%4!=3)result[i]=gdn_[i]->gdn_ab_method();return result;
    }
    std::array<PortableHcReadMethod,48> attention_hc_read_methods()const{
        std::array<PortableHcReadMethod,48> r{};r[0]=entry_->composition().attention_hc_read_method();r[1]=ple_->ple().hc_read_method();
        for(int i=2;i<=46;++i)r[i]=i%4==3?qsa_[i]->attention().hc_read_method():gdn_[i]->attention().hc_read_method();r[47]=terminal_->attention().hc_read_method();return r;
    }
    std::array<PortableHcReadMethod,48> ffn_hc_read_methods()const{
        std::array<PortableHcReadMethod,48> r{};r[0]=entry_->composition().ffn_hc_read_method();r[1]=ple_->ffn().hc_read_method();
        for(int i=2;i<=46;++i)r[i]=i%4==3?qsa_[i]->ffn().hc_read_method():gdn_[i]->ffn().hc_read_method();r[47]=terminal_->terminal().hc_read_method();return r;
    }
    const Accounting& accounting()const{return accounting_;}
    uint64_t position()const{return rows_->position();}bool poisoned()const{return rows_->poisoned();}
    auto run(uint64_t first,const std::vector<int32_t>& tokens,const PortableExpertRowsBatch& provider){return rows_->run(first,tokens,provider);}
    void reset(){rows_->reset();}
};
}
