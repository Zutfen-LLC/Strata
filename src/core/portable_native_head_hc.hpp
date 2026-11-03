#pragma once
#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include "ggml-alloc.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>

namespace strata::core::portable_detail {
// Bounded T1 head read. The caller owns the materialized residual and the
// already authenticated F32/BF16 role banks for the duration of each call.
// The caller must separately admit those borrowed weights and the residual;
// bytes() accounts only for this owner's scratch and component resources.
// This component does not create an injection matrix or injection result.
class PortableNativeHeadHc {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    Context ctx_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeHc,decltype(&strata_vk_native_hc_destroy)> norm_{nullptr,strata_vk_native_hc_destroy};
    std::unique_ptr<StrataVkNativeHcMix,decltype(&strata_vk_native_hc_mix_destroy)> mix_{nullptr,strata_vk_native_hc_mix_destroy};
    ggml_backend_t backend_;
    const ggml_tensor *weight_,*down_,*up_;
    ggml_tensor *normalized_=nullptr,*rs_=nullptr,*low_pre_=nullptr,*low_=nullptr,*low_exp_=nullptr;
    ggml_tensor *up_pre_=nullptr,*gates_=nullptr,*up_exp_=nullptr,*mixed_=nullptr;
    size_t bytes_=0;
    uint64_t calls_=0;
    bool valid_=false;
    bool recorded_=false;
    std::filesystem::path receipts_;
    void borrowed(const ggml_tensor* t,ggml_type type,size_t bytes) const {
        native_hc_need(t&&t->type==type&&ggml_nbytes(t)==bytes&&ggml_is_contiguous(t)&&t->buffer,
                       "native head HC: contiguous typed borrowed bank required");
        auto* type_owner=ggml_backend_buffer_get_type(t->buffer);
        auto* device=ggml_backend_buft_get_device(type_owner);
        auto* expected=ggml_backend_get_device(backend_);
        native_hc_need(type_owner==ggml_backend_get_default_buffer_type(backend_)&&expected&&
                       (device?device==expected:(ggml_backend_buft_is_host(type_owner)&&ggml_backend_dev_type(expected)==GGML_BACKEND_DEVICE_TYPE_CPU)),
                       "native head HC: borrowed bank backend/device differs");
    }
public:
    PortableNativeHeadHc(ggml_backend_t backend,const ggml_tensor* weight,const ggml_tensor* down,const ggml_tensor* up,
                         size_t budget,const std::filesystem::path& receipts)
        :backend_(backend),weight_(weight),down_(down),up_(up),receipts_(receipts) {
        native_hc_environment();native_hc_need(backend&&!receipts.empty(),"native head HC: backend and receipt path required");
        borrowed(weight_,GGML_TYPE_F32,40960);borrowed(down_,GGML_TYPE_I32,6553600);borrowed(up_,GGML_TYPE_I32,6553600);
        ctx_.reset(ggml_init({16384,nullptr,true}));native_hc_need(bool(ctx_),"native head HC: metadata allocation");
        auto leaf=[&](int64_t n){auto* t=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,n);ggml_set_input(t);ggml_set_output(t);return t;};
        normalized_=leaf(10240);rs_=leaf(4);low_pre_=leaf(320);low_=leaf(320);low_exp_=leaf(320);
        up_pre_=leaf(10240);gates_=leaf(10240);up_exp_=leaf(10240);mixed_=leaf(2560);
        const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        native_hc_need(bound<=budget,"native head HC: scratch budget");
        native_hc_need(std::filesystem::create_directory(receipts),"native head HC: fresh receipt directory required");
        norm_.reset(strata_vk_native_hc_create(backend,(receipts/"effective.spv").c_str(),budget-bound));
        native_hc_need(bool(norm_),"native head HC: norm component unavailable");
        const size_t nb=strata_vk_native_hc_bytes(norm_.get());native_hc_need(nb<=budget-bound,"native head HC: norm budget");
        mix_.reset(strata_vk_native_hc_mix_create(backend,(receipts/"mix-effective.spv").c_str(),budget-bound-nb));
        native_hc_need(bool(mix_),"native head HC: mixer component unavailable");
        const size_t mb=strata_vk_native_hc_mix_bytes(mix_.get());native_hc_need(mb<=budget-bound-nb,"native head HC: mixer budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));native_hc_need(bool(buffer_),"native head HC: scratch allocation");
        const size_t actual=ggml_backend_buffer_get_size(buffer_.get());native_hc_need(actual<=bound,"native head HC: allocation estimate exceeded");
        bytes_=actual+nb+mb;
    }
    size_t bytes()const{return bytes_;}
    uint64_t calls()const{return calls_;}
    // Graph construction only. The bank's contents are not valid before run.
    ggml_tensor* mixed_bank_for_graph()const{return mixed_;}
    // The returned bank is valid until the next call or owner destruction;
    // a failed call invalidates it before any downstream consumer can use it.
    ggml_tensor* run(const ggml_tensor* materialized_residual){
        valid_=false;native_hc_need(calls_!=UINT64_MAX,"native head HC: call counter exhausted");
        borrowed(materialized_residual,GGML_TYPE_F32,40960);
        strata_vk_native_hc_norm_only(norm_.get(),materialized_residual,weight_,normalized_,rs_);
        strata_vk_native_hc_mix_run(mix_.get(),normalized_,down_,up_,low_pre_,low_,low_exp_,up_pre_,gates_,up_exp_,mixed_);
        ++calls_;valid_=true;return mixed_;
    }
    ggml_tensor* mixed()const{native_hc_need(valid_,"native head HC: no successful current result");return mixed_;}
    void record_first(const ggml_tensor* input){
        native_hc_need(valid_,"native head HC: current result required for receipt");if(recorded_)return;
        for(auto item:{std::pair{"residual",input},std::pair{"normalized",static_cast<const ggml_tensor*>(normalized_)},std::pair{"rs",static_cast<const ggml_tensor*>(rs_)},
                       std::pair{"low-pre",static_cast<const ggml_tensor*>(low_pre_)},std::pair{"low",static_cast<const ggml_tensor*>(low_)},std::pair{"low-exp",static_cast<const ggml_tensor*>(low_exp_)},
                       std::pair{"up-pre",static_cast<const ggml_tensor*>(up_pre_)},std::pair{"gates",static_cast<const ggml_tensor*>(gates_)},std::pair{"up-exp",static_cast<const ggml_tensor*>(up_exp_)},std::pair{"mixed",static_cast<const ggml_tensor*>(mixed_)}}){
            std::vector<float> values(size_t(ggml_nelements(item.second)));ggml_backend_tensor_get(item.second,values.data(),0,values.size()*sizeof(float));
            const auto path=receipts_/(std::string("first-")+item.first+".f32le");auto* f=std::fopen(path.string().c_str(),"wbx");native_hc_need(f,"native head HC: fresh receipt required");
            const bool written=std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native head HC: receipt write failed");
        }
        auto* f=std::fopen((receipts_/"first.json").string().c_str(),"wx");native_hc_need(f,"native head HC: fresh metadata required");
        const bool written=std::fprintf(f,"{\"materialized\":true,\"injection\":false,\"qualified\":false}\n")>0;const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native head HC: metadata write failed");recorded_=true;
    }
};
}
