#pragma once
#include "strata_vk_native_shared.hpp"
#include "strata_vk_native_stages.hpp"
#include "native_shared_layout.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>
namespace strata::core::portable_detail {
// One serialized execution shape. Immutable device weight tensors are borrowed
// from the shared T1/chunk weight owner. Input is the live pre-graph mixed row.
class PortableNativeShared {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    Context context_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeShared,decltype(&strata_vk_native_shared_destroy)> component_{nullptr,strata_vk_native_shared_destroy};
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> stages_{nullptr,strata_vk_native_stages_destroy};
    std::array<ggml_tensor*,4> weights_{};
    enum {QX,GATE,UP,H,QH,DOWN,RAW,SIGMOID,OUT,DUMMY,COUNT};
    std::array<ggml_tensor*,COUNT> tensors_{};
    uint32_t hidden_,ff_,tokens_;
    size_t bytes_=0;
    static std::string receipt_directory(const std::string& prefix,uint32_t tokens) {
        native_shared_need(!prefix.empty(),"portable native shared: private receipt prefix required");
        static std::atomic<uint64_t> next{0};
        for(size_t i=0;i<1024;++i) {
            auto path=prefix+".t-"+std::to_string(tokens)+".owner-"+std::to_string(next.fetch_add(1));
            if(std::filesystem::create_directory(path))return path;
        }
        throw std::runtime_error("portable native shared: receipt namespace exhausted");
    }
public:
    static constexpr size_t metadata_bytes=32u<<10; // 16 KiB tensors, two 4 KiB components and owners
    PortableNativeShared(ggml_backend_t backend,uint32_t h,uint32_t f,uint32_t t,
        const std::array<ggml_tensor*,4>& weights,size_t budget,const std::string& prefix)
        :weights_(weights),hidden_(h),ff_(f),tokens_(t) {
        native_shared_shape(h,f,t);native_shared_need(backend,"portable native shared: accelerator required");
        context_.reset(ggml_init({16384,nullptr,true}));native_shared_need(bool(context_),"portable native shared: metadata allocation");
        const std::array<int64_t,COUNT> n{int64_t(t)*(h/32)*9,int64_t(t)*f,int64_t(t)*f,int64_t(t)*f,
            int64_t(t)*(f/32)*9,int64_t(t)*h,t,t,int64_t(t)*h,1};
        for(size_t i=0;i<COUNT;++i)tensors_[i]=ggml_new_tensor_1d(context_.get(),i==QX||i==QH?GGML_TYPE_I32:GGML_TYPE_F32,n[i]);
        const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(backend));
        native_shared_need(bound<=budget,"portable native shared: scratch graph budget exceeded");
        const auto dir=receipt_directory(prefix,t);
        component_.reset(strata_vk_native_shared_create(backend,(dir+"/dense-scalar-effective.spv").c_str(),budget-bound));
        native_shared_need(bool(component_),"portable native shared: component creation failed");
        const auto first=strata_vk_native_shared_bytes(component_.get());
        native_shared_need(first<=budget-bound,"portable native shared: component graph budget exceeded");
        stages_.reset(strata_vk_native_stages_create_quant_rn_observation(backend,(dir+"/quant-effective.spv").c_str(),budget-bound-first));
        native_shared_need(bool(stages_),"portable native shared: quantizer creation failed");
        const auto second=strata_vk_native_stages_bytes(stages_.get());
        native_shared_need(second<=budget-bound-first,"portable native shared: quantizer graph budget exceeded");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));native_shared_need(bool(buffer_),"portable native shared: scratch allocation failed");
        const auto actual=ggml_backend_buffer_get_size(buffer_.get());
        native_shared_need(actual<=budget-first-second,"portable native shared: actual scratch graph budget exceeded");
        bytes_=actual+first+second;
    }
    size_t bytes() const{return bytes_;}
    size_t host_bound() const{return metadata_bytes+size_t(tokens_)*hidden_*4;}
    std::vector<float> run(const ggml_tensor* mixed) {
        auto& v=tensors_;const int32_t starts[]={0,int32_t(tokens_)};
        strata_vk_native_stages_run(stages_.get(),0,mixed,mixed,v[DOWN],v[QX],hidden_,starts,2);
        auto dense=[&](size_t weight,size_t input,size_t output,uint32_t ni,uint32_t no){
            strata_vk_native_shared_run(component_.get(),0,weights_[weight],v[input],v[output],v[DUMMY],ni,no,tokens_);};
        dense(0,QX,GATE,hidden_,ff_);dense(1,QX,UP,hidden_,ff_);
        strata_vk_native_stages_run(stages_.get(),1,v[GATE],v[UP],v[H],v[QH],ff_,starts,2);
        strata_vk_native_stages_run(stages_.get(),0,v[H],v[H],v[H],v[QH],ff_,starts,2);
        dense(2,QH,DOWN,ff_,hidden_);
        strata_vk_native_shared_run(component_.get(),1,weights_[3],mixed,v[RAW],v[DUMMY],hidden_,1,tokens_);
        strata_vk_native_shared_run(component_.get(),2,v[RAW],v[DOWN],v[OUT],v[SIGMOID],hidden_,1,tokens_);
        std::vector<float> result(size_t(tokens_)*hidden_);ggml_backend_tensor_get(v[OUT],result.data(),0,result.size()*4);
        for(float x:result)native_shared_need(std::isfinite(x),"portable native shared: nonfinite final output");
        return result;
    }
};
}
