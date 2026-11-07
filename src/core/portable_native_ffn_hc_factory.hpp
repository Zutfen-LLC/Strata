#pragma once
// Exact named FFN HC roles -> copied composed-HC owner. No model admission.
#include "portable_native_hc_composed.hpp"
#include "native_hc_layout.hpp"
#include "strata/core/portable_ffn.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
namespace strata::core::portable_detail {
class NativeFfnHcFactory {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
public:
    // Conservative requested host payload. Includes one largest Q8->BF16
    // conversion while upload is resident, temporary metadata and the HC
    // owner's own preflight/candidate bound. Source GGUF mapping, allocator,
    // driver and caller captures are excluded.
    static size_t host_bound(size_t prefix_bytes){return ple_host_add(6553600+16384,PortableNativeHcComposed::host_bound(2,prefix_bytes));}
    static std::unique_ptr<PortableNativeHcComposed> create(int layer,ggml_backend_t backend,
        const std::vector<PortableTensorView>& views,size_t device_budget,size_t host_budget,
        const char* prefix,PortableNativeHcComposed::Dispatch test_dispatch={},
        std::optional<PortableNativeHcComposed::ComponentCaps> caps=std::nullopt, bool layer0_attention=false){
        need(layer>=0&&layer<47&&(!layer0_attention||layer==0)&&backend&&views.size()==4,"FFN HC factory: layer/backend/role count");
        const bool injected=bool(test_dispatch.pending)||bool(test_dispatch.norm)||bool(test_dispatch.mix);
        if(!injected)native_hc_environment();
        need(injected||(prefix&&*prefix),"FFN HC factory: receipt prefix");
        need(host_bound(!injected&&prefix?std::strlen(prefix):0)<=host_budget,"FFN HC factory: host budget");
        const std::string base="blk."+std::to_string(layer)+".";
        const std::string role=layer0_attention?"hc_attn_":"hc_ffn_";
        const std::array<std::string,4> names{base+role+"norm.weight",base+role+"inject.weight",base+role+"down.weight",base+role+"up.weight"};
        const std::array<std::vector<int64_t>,4> shapes{{{10240},{10240,4},{10240,320},{320,10240}}};
        const std::array<size_t,4> elements{{10240,40960,3276800,3276800}};
        std::array<const PortableTensorView*,4> selected{};
        for(const auto& view:views){auto it=std::find(names.begin(),names.end(),view.name);need(it!=names.end(),"FFN HC factory: unexpected name/layer");size_t i=size_t(it-names.begin());
            need(!selected[i]&&view.data&&view.shape==shapes[i]&&view.type==(i?8u:0u)&&view.bytes==(i?elements[i]/32*34:elements[i]*4),"FFN HC factory: role descriptor");selected[i]=&view;}
        for(auto* v:selected)need(v,"FFN HC factory: missing role");
        Context ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"FFN HC factory: metadata");
        std::array<ggml_tensor*,4> roles{};
        roles[0]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,10240);
        for(size_t i=1;i<4;++i)roles[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,int64_t(elements[i]/2));
        for(auto* r:roles)need(r,"FFN HC factory: tensor metadata");
        const auto upload=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend));
        std::array<const ggml_tensor*,4> source{roles[0],roles[1],roles[2],roles[3]};
        const auto retained=PortableNativeHcComposed::tensor_bound(backend,source);
        size_t component=0;if(caps)component=ple_host_add(ple_host_add(caps->pending,caps->norm),caps->mix);
        need(upload<=device_budget&&retained<=device_budget-upload&&component<=device_budget-upload-retained,"FFN HC factory: construction device budget");
        Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend),ggml_backend_buffer_free);need(bool(buffer),"FFN HC factory: upload allocation");
        const auto actual=ggml_backend_buffer_get_size(buffer.get());need(actual<=device_budget&&retained<=device_budget-actual&&component<=device_budget-actual-retained,"FFN HC factory: actual construction budget");
        ggml_backend_tensor_set(roles[0],selected[0]->data,0,selected[0]->bytes);
        for(size_t i=1;i<4;++i){auto packed=native_hc_q8_bf16(static_cast<const uint8_t*>(selected[i]->data),selected[i]->bytes,elements[i]);ggml_backend_tensor_set(roles[i],packed.data(),0,packed.size()*2);}
        auto result=std::make_unique<PortableNativeHcComposed>(backend,roles[0],roles[1],roles[2],roles[3],device_budget-actual,prefix,std::move(test_dispatch),host_budget,caps);
        result->checked_layer_=layer; result->checked_attention_=layer0_attention;
        return result;
    }
    static std::vector<PortableTensorView> from_gguf(const GgufModel& model,int layer){
        std::vector<PortableTensorView> views;views.reserve(4);
        for(const char* suffix:{"hc_ffn_norm.weight","hc_ffn_inject.weight","hc_ffn_down.weight","hc_ffn_up.weight"}){
            auto name="blk."+std::to_string(layer)+"."+suffix;size_t shard=0;const auto* t=model.find(name,&shard);
            need(t&&model.in_bounds(*t,shard),"FFN HC factory: missing/out-of-bounds GGUF role");
            PortableTensorView v{name,{},t->type,model.shard(shard).tensor_data(*t),size_t(tensor_payload_bytes(*t))};
            for(auto d:t->shape){need(d<=INT64_MAX,"FFN HC factory: dimension overflow");v.shape.push_back(int64_t(d));}views.push_back(std::move(v));
        }return views;
    }
};
}
