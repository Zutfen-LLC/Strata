#pragma once
// Internal GDN input role upload and executor ownership. This is not model
// admission: other roles, providers, mapping, driver and allocator overhead
// must be admitted separately. Backend and mapped source outlive construction;
// the backend must outlive this owner. No host pointer is used as device storage.
#include "portable_native_gdn_input_rows.hpp"
#include "strata/core/portable_ffn.hpp"
#include "strata/artifact/gguf_reader.hpp"

namespace strata::core::portable_detail {
class PortableNativeGdnInputOwner {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    Context context_{nullptr, ggml_free};
    Buffer storage_{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<PortableNativeGdnInputRows> rows_;
    int layer_;
    static void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
public:
    // Excludes source mapping and backend-private/allocator overhead. The row
    // estimate is the existing owner's requested device envelope, not a
    // physical memory measurement. Host covers metadata and its role readback.
    static size_t host_bound() { return ple_host_add(16384 + 65536, PortableNativeGdnInputRows::host_bound()); }
    static size_t host_bound(PortableGdnProjectionMethod method) {
        need(method==PortableGdnProjectionMethod::legacy_q8_1||method==PortableGdnProjectionMethod::native_f16,"GDN input factory: method");
        return method==PortableGdnProjectionMethod::native_f16?ple_host_add(16384+65536,PortableNativeGdnF16Rows::host_bound()):host_bound();
    }
    static size_t device_bound(ggml_backend_t backend,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1) {
        need(backend, "GDN input factory: backend");
        Context c(ggml_init({16384,nullptr,true}),ggml_free);need(bool(c), "GDN input factory: metadata");
        need(ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,10240) &&
             ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,6144), "GDN input factory: tensor metadata");
        const size_t upload=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(backend));
        // The selected row estimate already charges an exact-size raw buffer.
        // Reserve a second upload envelope for alignment/aggregate-buffer
        // differences; this deliberately overcounts rather than assuming RSS.
        size_t result=ple_host_add(upload,PortableNativeGdnInputRows::estimate(backend,method));
        return method==PortableGdnProjectionMethod::native_f16?ple_host_add(result,upload):result;
    }
    PortableNativeGdnInputOwner(int layer, ggml_backend_t backend,
        const std::array<PortableTensorView,2>& roles, const std::string& directory,
        size_t host_budget, size_t device_budget,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1) : layer_(layer) {
        need(layer>=0 && layer<48 && layer%4!=3 && backend, "GDN input factory: layer/backend");
        need(host_budget>=host_bound(method), "GDN input factory: host budget");
        const std::array<int64_t,2> outputs{10240,6144};
        const std::array<std::string,2> names{"blk."+std::to_string(layer)+".attn_qkv.weight", "blk."+std::to_string(layer)+".attn_gate.weight"};
        for(size_t i=0;i<2;++i) {
            need(roles[i].name==names[i] && roles[i].type==8 && roles[i].data &&
                 roles[i].shape==std::vector<int64_t>({2560,outputs[i]}) &&
                 roles[i].bytes==size_t(outputs[i])*80*34, "GDN input factory: role descriptor");
            qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(roles[i].data),size_t(outputs[i])*80,34), names[i], true);
            if(method==PortableGdnProjectionMethod::native_f16)PortableNativeGdnF16Rows::validate_role(roles[i].data,roles[i].bytes);
        }
        if(method==PortableGdnProjectionMethod::native_f16)
            need(device_bound(backend,method)<=device_budget,"GDN input factory: selected device budget");
        context_.reset(ggml_init({16384,nullptr,true}));
        need(bool(context_), "GDN input factory: metadata");
        std::array<ggml_tensor*,2> weights{};
        for(size_t i=0;i<2;++i) { weights[i]=ggml_new_tensor_2d(context_.get(),GGML_TYPE_Q8_0,2560,outputs[i]); need(weights[i], "GDN input factory: tensor metadata"); }
        const size_t upload=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(backend));
        const size_t scratch=PortableNativeGdnInputRows::estimate(backend,method);
        need(upload<=device_budget && scratch<=device_budget-upload, "GDN input factory: device budget");
        storage_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));
        need(bool(storage_), "GDN input factory: allocation");
        const size_t actual=ggml_backend_buffer_get_size(storage_.get());
        need(actual<=device_budget && scratch<=device_budget-actual, "GDN input factory: actual device budget");
        for(size_t i=0;i<2;++i) ggml_backend_tensor_set(weights[i],roles[i].data,0,roles[i].bytes);
        rows_=std::make_unique<PortableNativeGdnInputRows>(backend,weights,
            std::array<const void*,2>{roles[0].data,roles[1].data},
            std::array<size_t,2>{roles[0].bytes,roles[1].bytes},directory,device_budget-actual,method);
    }
    static std::array<PortableTensorView,2> from_gguf(const GgufModel& model, int layer) {
        need(layer>=0 && layer<48 && layer%4!=3, "GDN input factory: layer");
        std::array<PortableTensorView,2> result;
        const std::array<const char*,2> suffix{"attn_qkv.weight","attn_gate.weight"};
        for(size_t i=0;i<2;++i) {
            auto name="blk."+std::to_string(layer)+"."+suffix[i]; size_t shard=0;
            const auto* t=model.find(name,&shard);
            need(t && model.in_bounds(*t,shard), "GDN input factory: missing/out-of-bounds GGUF role");
            result[i]={name,{},t->type,model.shard(shard).tensor_data(*t),size_t(tensor_payload_bytes(*t))};
            for(auto d:t->shape) { need(d<=INT64_MAX, "GDN input factory: dimension overflow"); result[i].shape.push_back(int64_t(d)); }
        }
        return result;
    }
    PortableNativeGdnInputOwner(int layer, ggml_backend_t backend, const GgufModel& model,
        const std::string& directory, size_t host_budget, size_t device_budget,PortableGdnProjectionMethod method=PortableGdnProjectionMethod::legacy_q8_1)
      : PortableNativeGdnInputOwner(layer,backend,from_gguf(model,layer),directory,host_budget,device_budget,method) {}
    PortableNativeGdnInputOwner(const PortableNativeGdnInputOwner&)=delete;
    PortableNativeGdnInputOwner& operator=(const PortableNativeGdnInputOwner&)=delete;
    int layer() const { return layer_; }
    PortableNativeGdnInputRows& rows() { return *rows_; }
    const PortableNativeGdnInputRows& rows() const { return *rows_; }
};
}
