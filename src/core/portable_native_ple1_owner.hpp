#pragma once
// Internal layer-one owner. The backend is borrowed and must outlive this
// owner. The model mapping, table, converted host value and resident tensors
// outlive the PLE executor, whose Roles explicitly borrow them. The source
// table made by from_gguf owns its own shard mapping. No model admission.
#include "portable_native_ordinary_gdn_factory.hpp"
#include "portable_native_ffn_owner.hpp"
#include "portable_native_ffn_composition.hpp"

namespace strata::core::portable_detail {
class PortableNativePle1Owner {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    std::shared_ptr<const GgufModel> model_;
    std::shared_ptr<PortablePleTable> table_;
    Context context_{nullptr,ggml_free};
    Buffer storage_{nullptr,ggml_backend_buffer_free};
    std::vector<uint16_t> value_;
    std::unique_ptr<PortableNativeFfnOwner> rows_;
    std::unique_ptr<PortableNativeGdnComponents> gdn_;
    std::unique_ptr<PortableNativePleComposition> ple_;
    std::unique_ptr<PortableNativePleGdnComposition> attention_;
    std::unique_ptr<PortableNativeFfnComposition> ffn_;
    static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
public:
    using Roles=std::array<PortableTensorView,6>;
    // total is a mixed conservative sum: gdn/rows host and device, PLE
    // admission's mixed bound, role device, retained value and conversion
    // peak, FFN host/device, and PLE-GDN host. Admission already counts the
    // separate T1 and T2 retained BF16 value copies. Mappings/table, backend
    // private resources, allocator overhead and caller results are excluded.
    struct Budgets {
        size_t total,ple,role_device,ffn_device;
        PortableNativeGdnComponents::Budgets gdn;
        PortableNativeFfnOwner::Budgets rows;
        PortableNativeHcComposed::ComponentCaps ple_caps,ffn_caps;
    };
    static constexpr size_t role_host_bound(){return 16384+2560u*2560*2+3276800u*2;}
    static Roles from_gguf(const GgufModel& model){
        const char* names[]={"ple_key.weight","ple_value.weight","ple_norm_key.weight","ple_norm_query.weight","ple_norm_conv.weight","ple_conv1d.weight"};
        Roles r;
        for(size_t i=0;i<r.size();++i){auto name=std::string("blk.1.")+names[i];size_t shard=0;auto* t=model.find(name,&shard);
            need(t&&model.in_bounds(*t,shard),"PLE1 owner: missing/out-of-bounds role");
            r[i]={name,{},t->type,model.shard(shard).tensor_data(*t),size_t(tensor_payload_bytes(*t))};
            for(auto d:t->shape){need(d<=INT64_MAX,"PLE1 owner: dimension overflow");r[i].shape.push_back(int64_t(d));}}
        return r;
    }
    static void validate(const Roles& r){
        const char* names[]={"ple_key.weight","ple_value.weight","ple_norm_key.weight","ple_norm_query.weight","ple_norm_conv.weight","ple_conv1d.weight"};
        for(size_t i=0;i<r.size();++i){
            const auto shape=i==0?std::vector<int64_t>{2560,10240}:i==1?std::vector<int64_t>{2560,2560}:i<5?std::vector<int64_t>{2560,4}:std::vector<int64_t>{4,10240};
            bool shape_ok=r[i].shape==shape||(i>=2&&i<5&&r[i].shape==std::vector<int64_t>{10240});
            bool type_ok=i<2?r[i].type==8:i<5?r[i].type==0:(r[i].type==0||r[i].type==1);
            size_t bytes=i==0?2560u*10240/32*34:i==1?2560u*2560/32*34:i<5?10240u*4:40960u*(r[i].type==1?2:4);
            need(r[i].name==std::string("blk.1.")+names[i]&&r[i].data&&shape_ok&&type_ok&&r[i].bytes==bytes,"PLE1 owner: role descriptor");
        }
        qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(r[0].data),2560u*10240/32,34),r[0].name,true);
    }
    // Production entry: the table factory validates fleet metadata, hashes and
    // geometry and reopens the source shard. Failure is propagated verbatim.
    static std::unique_ptr<PortableNativePle1Owner> from_model(std::shared_ptr<const GgufModel> model,ggml_backend_t backend,const std::string& prefix,const Budgets& b,PortableRouterMethod method=PortableRouterMethod::legacy_f32){
        need(bool(model),"PLE1 owner: missing model");std::string error;auto table=PortablePleTable::from_gguf(*model,error);
        if(!table)throw std::runtime_error("PLE1 owner: "+error);
        return std::make_unique<PortableNativePle1Owner>(std::move(model),std::move(table),backend,prefix,b,method);
    }
    // Supplying an already checked table also permits the bounded copy() test
    // seam; it is not evidence for fleet table metadata or loading.
    PortableNativePle1Owner(std::shared_ptr<const GgufModel> model,std::shared_ptr<PortablePleTable> table,
                            ggml_backend_t backend,const std::string& prefix,const Budgets& b,PortableRouterMethod method=PortableRouterMethod::legacy_f32)
      :model_(std::move(model)),table_(std::move(table)) {
        need(model_&&table_&&backend&&!prefix.empty(),"PLE1 owner: model/table/backend/prefix");
        auto p=from_gguf(*model_);validate(p);
        auto g=PortableNativeGdnComponents::from_gguf(*model_,1);PortableNativeGdnComponents::validate(1,g);
        auto r=PortableNativeFfnOwner::from_gguf(*model_,1);PortableNativeFfnOwner::validate(1,r,method);
        auto f=NativeFfnHcFactory::from_gguf(*model_,1);NativeFfnHcFactory::select(1,f,false);
        std::vector<PortableTensorView> ah{g[9],g[10],g[11],g[12]};auto hc=NativeFfnHcFactory::select(1,ah,true);
        auto pp=prefix+".ple-hc",directory=prefix+".ple",fp=prefix+".ffn";
        auto fhost=ple_host_add(PortableNativeFfnComposition::host_bound(),NativeFfnHcFactory::host_bound(fp.size()));
        size_t total=role_host_bound();for(size_t n:{b.gdn.host,b.gdn.device,b.rows.host,b.rows.device,b.ple,b.role_device,b.ffn_device,fhost,PortableNativePleGdnComposition::host_bound()})total=ple_host_add(total,n);
        need(total<=b.total&&NativeFfnHcFactory::device_bound(backend,b.ffn_caps)<=b.ffn_device,"PLE1 owner: aggregate budget");
        context_.reset(ggml_init({16384,nullptr,true}));need(bool(context_),"PLE1 owner: metadata");
        auto* key=ggml_new_tensor_2d(context_.get(),GGML_TYPE_Q8_0,2560,10240);
        auto* value=ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,2560*2560/2);
        std::array<const ggml_tensor*,4> ht{ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,10240),ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,20480),ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,1638400)};
        need(key&&value&&ht[0]&&ht[1]&&ht[2]&&ht[3],"PLE1 owner: tensor metadata");
        auto upload=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(backend));
        need(upload<=b.role_device,"PLE1 owner: role budget");
        auto bounds=PortableNativePleAdmission::calculate(backend,ht,directory,pp.size(),b.ple_caps);
        PortableNativePleAdmission::require_budget(bounds,b.ple);
        storage_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));need(storage_&&ggml_backend_buffer_get_size(storage_.get())<=b.role_device,"PLE1 owner: role allocation");
        ggml_backend_tensor_set(key,p[0].data,0,p[0].bytes);
        value_=native_hc_q8_bf16(static_cast<const uint8_t*>(p[1].data),p[1].bytes,2560u*2560);
        ggml_backend_tensor_set(value,value_.data(),0,value_.size()*2);
        ggml_backend_tensor_set(const_cast<ggml_tensor*>(ht[0]),hc[0]->data,0,hc[0]->bytes);
        const size_t count[]={0,40960,3276800,3276800};
        for(size_t i=1;i<4;++i){auto x=native_hc_q8_bf16(static_cast<const uint8_t*>(hc[i]->data),hc[i]->bytes,count[i]);ggml_backend_tensor_set(const_cast<ggml_tensor*>(ht[i]),x.data(),0,x.size()*2);}
        rows_=std::make_unique<PortableNativeFfnOwner>(1,backend,r,prefix+".rows",b.rows,method);
        gdn_=std::make_unique<PortableNativeGdnComponents>(1,backend,g,prefix+".gdn",b.gdn);
        PortableNativePleExecutor::Roles roles{backend,table_.get(),key,p[0].data,p[0].bytes,value,value_.data(),value_.size(),
            {static_cast<const float*>(p[2].data),static_cast<const float*>(p[3].data),static_cast<const float*>(p[4].data)},p[5].data,p[5].bytes,p[5].type,directory};
        ple_=std::make_unique<PortableNativePleComposition>(roles,ht,backend,b.ple_caps,b.ple,pp);
        attention_=std::make_unique<PortableNativePleGdnComposition>(*ple_,gdn_->input(),gdn_->take_core(),gdn_->output(),PortableNativePleGdnComposition::host_bound());
        ffn_=std::make_unique<PortableNativeFfnComposition>(*attention_,backend,f,rows_->rows(),b.ffn_device,fhost,fp.c_str(),b.ffn_caps);
    }
    PortableNativePle1Owner(const PortableNativePle1Owner&)=delete;
    PortableNativePle1Owner& operator=(const PortableNativePle1Owner&)=delete;
    PortableRouterMethod router_method()const{return rows_->router_method();}
    PortableNativePleComposition& ple(){return *ple_;}
    PortableNativePleGdnComposition& attention(){return *attention_;}
    PortableNativeFfnComposition& ffn(){return *ffn_;}
    const PortablePleTable& table()const{return *table_;}
};
}
