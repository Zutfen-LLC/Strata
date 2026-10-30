#pragma once
#include "strata_vk_native_router.hpp"
#include "strata/core/portable_ffn.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
namespace strata::core::portable_detail {
// One serialized FFN execution shape. Input is the existing live device logits;
// only original-order IDs/weights return to the existing host expert boundary.
class PortableNativeRouter {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    using Component=std::unique_ptr<StrataVkNativeRouter,decltype(&strata_vk_native_router_destroy)>;
    Context context_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    Component component_{nullptr,strata_vk_native_router_destroy};
    ggml_tensor *ids_=nullptr,*weights_=nullptr;
    uint32_t tokens_;
    size_t bytes_=0;
    static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
    static std::string receipt_path(const std::string& prefix,uint32_t tokens) {
        need(!prefix.empty(),"portable native router: private receipt prefix required");
        static std::atomic<uint64_t> next{0};
        for(size_t attempt=0;attempt<1024;++attempt) {
            const auto dir=prefix+".t-"+std::to_string(tokens)+".owner-"+std::to_string(next.fetch_add(1));
            if(std::filesystem::create_directory(dir))return (std::filesystem::path(dir)/"effective.spv").string();
        }
        throw std::runtime_error("portable native router: receipt namespace exhausted");
    }
public:
    // 16 KiB tensor context + 4 KiB component context, rounded up for owners.
    static constexpr size_t metadata_bytes=24u<<10;
    PortableNativeRouter(ggml_backend_t backend,int64_t tokens,size_t budget,const std::string& prefix):tokens_(uint32_t(tokens)) {
        need(backend && tokens>=1 && tokens<=8,"portable native router: invalid backend/shape");
        context_.reset(ggml_init({16384,nullptr,true}));need(bool(context_),"portable native router: metadata allocation failed");
        ids_=ggml_new_tensor_1d(context_.get(),GGML_TYPE_I32,tokens*10);
        weights_=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,tokens*10);
        const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(backend));
        need(bound<=budget,"portable native router: outputs exceed graph budget");
        const auto receipt=receipt_path(prefix,tokens_);
        component_.reset(strata_vk_native_router_create(backend,receipt.c_str()));
        need(bool(component_),"portable native router: component creation failed");
        const auto owner=strata_vk_native_router_bytes(component_.get());
        need(owner<=budget-bound,"portable native router: component exceeds graph budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));
        need(bool(buffer_),"portable native router: output allocation failed");
        const auto actual=ggml_backend_buffer_get_size(buffer_.get());
        need(actual<=budget-owner,"portable native router: aligned outputs exceed graph budget");
        bytes_=actual+owner;
    }
    size_t bytes() const{return bytes_;}
    size_t host_bound() const{return metadata_bytes+size_t(tokens_)*(10*16+sizeof(PortableRoute));}
    void run(const ggml_tensor* logits,std::vector<PortableRoute>& routes) {
        need(routes.size()==tokens_,"portable native router: route count mismatch");
        strata_vk_native_router_run(component_.get(),logits,ids_,weights_,tokens_);
        std::vector<int32_t> ids(size_t(tokens_)*10);std::vector<float> weights(size_t(tokens_)*10);
        ggml_backend_tensor_get(ids_,ids.data(),0,ids.size()*4);
        ggml_backend_tensor_get(weights_,weights.data(),0,weights.size()*4);
        for(size_t t=0;t<tokens_;++t) {
            std::array<bool,512> seen{};bool positive=false;
            for(size_t k=0;k<10;++k) {
                const auto id=ids[t*10+k];const auto weight=weights[t*10+k];
                need(id>=0 && id<512 && !seen[size_t(id)],"portable native router: invalid/duplicate expert ID");
                seen[size_t(id)]=true;
                need(std::isfinite(weight) && weight>=0,"portable native router: invalid routing weight");
                positive|=weight>0;
            }
            need(positive,"portable native router: zero routing weights");
        }
        for(size_t t=0;t<tokens_;++t) {
            routes[t].ids.assign(ids.begin()+t*10,ids.begin()+(t+1)*10);
            routes[t].weights.assign(weights.begin()+t*10,weights.begin()+(t+1)*10);
            routes[t].native_router=true;
        }
    }
};
}
