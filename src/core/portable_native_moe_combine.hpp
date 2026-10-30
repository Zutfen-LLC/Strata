#pragma once
#include "strata_vk_native_moe_combine.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include <memory>
#include <atomic>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <vector>
namespace strata::core::portable_detail {
// One serialized FFN execution shape, using the existing host row boundary.
// The provider has already restored CPU misses / GPU replies to router slots.
class PortableNativeMoeCombine {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    using Component = std::unique_ptr<StrataVkNativeMoeCombine, decltype(&strata_vk_native_moe_combine_destroy)>;
    Context context_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    Component component_{nullptr,strata_vk_native_moe_combine_destroy};
    ggml_tensor *parts_=nullptr,*weights_=nullptr,*shared_=nullptr,*output_=nullptr;
    uint32_t tokens_;
    size_t bytes_=0;
    static void need(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
    static std::string receipt_path(const std::string& prefix,uint32_t tokens) {
        need(!prefix.empty(),"portable native combine: private receipt prefix required");
        static std::atomic<uint64_t> next{0};
        // Atomic directory creation reserves an unused owner namespace, including
        // after restart/replanning. Never reuse an old receipt or truncate it.
        for(size_t attempt=0;attempt<1024;++attempt) {
            const auto directory=prefix+".t-"+std::to_string(tokens)+".owner-"+std::to_string(next.fetch_add(1));
            if(std::filesystem::create_directory(directory))return (std::filesystem::path(directory)/"effective.spv").string();
        }
        throw std::runtime_error("portable native combine: receipt namespace exhausted; use a fresh private prefix");
    }
public:
    static constexpr size_t metadata_bytes=16384;
    PortableNativeMoeCombine(ggml_backend_t backend,int64_t tokens,size_t budget,const std::string& prefix):tokens_(uint32_t(tokens)) {
        need(backend && tokens>=1 && tokens<=8,"portable native combine: invalid backend/shape");
        context_.reset(ggml_init({metadata_bytes,nullptr,true}));
        need(bool(context_),"portable native combine: metadata allocation failed");
        parts_=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,tokens*10*2560);
        weights_=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,tokens*10);
        shared_=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,tokens*2560);
        output_=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,tokens*2560);
        const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(backend));
        need(bound<=budget,"portable native combine: staging exceeds graph budget");
        const auto receipt=receipt_path(prefix,tokens_);
        component_.reset(strata_vk_native_moe_combine_create(backend,receipt.c_str()));
        need(bool(component_),"portable native combine: component creation failed");
        const auto owner=strata_vk_native_moe_combine_bytes(component_.get());
        need(owner<=budget-bound,"portable native combine: component exceeds graph budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));
        need(bool(buffer_),"portable native combine: staging allocation failed");
        const auto actual=ggml_backend_buffer_get_size(buffer_.get());
        need(actual<=budget-owner,"portable native combine: aligned staging exceeds graph budget");
        bytes_=actual+owner;
    }
    size_t bytes() const {return bytes_;}
    std::vector<float> run(const std::vector<float>& rows,const std::vector<float>& weights,
                           const std::vector<float>& gated_shared) {
        need(rows.size()==size_t(tokens_)*10*2560 && weights.size()==size_t(tokens_)*10 &&
             gated_shared.size()==size_t(tokens_)*2560,"portable native combine: boundary shape mismatch");
        ggml_backend_tensor_set(parts_,rows.data(),0,rows.size()*sizeof(float));
        ggml_backend_tensor_set(weights_,weights.data(),0,weights.size()*sizeof(float));
        ggml_backend_tensor_set(shared_,gated_shared.data(),0,gated_shared.size()*sizeof(float));
        strata_vk_native_moe_combine_run(component_.get(),parts_,weights_,shared_,output_,tokens_);
        std::vector<float> block(gated_shared.size());
        ggml_backend_tensor_get(output_,block.data(),0,block.size()*sizeof(float));
        return block;
    }
};
}
