#pragma once
#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include "ggml-alloc.h"
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <memory>
namespace strata::core::portable_detail {
// Diagnostic serial owner. Graphs borrow these allocated leaves; no scheduler
// may reuse their storage between residual upload and component/consumer calls.
class PortableNativeHc {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    Context context_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeHc,decltype(&strata_vk_native_hc_destroy)> component_{nullptr,strata_vk_native_hc_destroy};
    std::unique_ptr<StrataVkNativeHcMix,decltype(&strata_vk_native_hc_mix_destroy)> mix_{nullptr,strata_vk_native_hc_mix_destroy};
    std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> pending_{nullptr,strata_vk_native_hc_pending_destroy};
    const ggml_tensor *norm_,*weight_,*down_,*up_;
    size_t bytes_=0;
    uint64_t calls_=0,mix_calls_=0,pending_calls_=0;
    std::string receipt_dir_;
    bool first_recorded_=false;
public:
    ggml_tensor *pending_input=nullptr,*previous_output=nullptr,*previous_inject=nullptr,*pending_gate=nullptr,*pending_exp=nullptr;
    ggml_tensor *residual=nullptr,*normalized=nullptr,*rs=nullptr,*inject=nullptr;
    ggml_tensor *low_pre=nullptr,*low=nullptr,*low_exp=nullptr,*up_pre=nullptr,*gates=nullptr,*up_exp=nullptr,*mixed=nullptr;
    PortableNativeHc(ggml_backend_t backend,const ggml_tensor* norm,const ggml_tensor* weight,
                     size_t budget,const std::string& prefix,const ggml_tensor* down=nullptr,const ggml_tensor* up=nullptr,bool pending=false):norm_(norm),weight_(weight),down_(down),up_(up) {
        native_hc_environment();native_hc_need(backend&&!prefix.empty(),"native HC serial: backend/receipt prefix required");
        native_hc_need(bool(down)==bool(up),"native HC serial: paired down/up roles required");
        native_hc_need(!pending||down,"native HC pending: composed read required");
        context_.reset(ggml_init({16384,nullptr,true}));native_hc_need(bool(context_),"native HC serial: metadata");
        residual=ggml_new_tensor_3d(context_.get(),GGML_TYPE_F32,2560,4,1);
        normalized=ggml_new_tensor_2d(context_.get(),GGML_TYPE_F32,10240,1);
        rs=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
        inject=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
        for(auto* t:{residual,normalized,rs,inject}){ggml_set_input(t);ggml_set_output(t);}
        if(down_) {
            low_pre=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,320);
            low=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,320);
            low_exp=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,320);
            up_pre=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,10240);
            gates=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,10240);
            up_exp=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,10240);
            mixed=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,2560);
            for(auto* t:{low_pre,low,low_exp,up_pre,gates,up_exp,mixed}){ggml_set_input(t);ggml_set_output(t);}
        }
        if(pending) {
            pending_input=ggml_new_tensor_3d(context_.get(),GGML_TYPE_F32,2560,4,1);
            previous_output=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,2560);
            previous_inject=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
            pending_gate=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
            pending_exp=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
            for(auto* t:{pending_input,previous_output,previous_inject,pending_gate,pending_exp}){ggml_set_input(t);ggml_set_output(t);}
        }
        auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(backend));
        native_hc_need(bound<=budget,"native HC serial: scratch graph budget");
        static std::atomic<uint64_t> next{0};std::string dir;
        for(int i=0;i<1024;++i){auto candidate=prefix+".owner-"+std::to_string(next.fetch_add(1));if(std::filesystem::create_directory(candidate)){dir=candidate;break;}}
        native_hc_need(!dir.empty(),"native HC serial: fresh receipt namespace");
        receipt_dir_=dir;
        component_.reset(strata_vk_native_hc_create(backend,(dir+"/effective.spv").c_str(),budget-bound));
        native_hc_need(bool(component_),"native HC serial: component unavailable");
        size_t metadata=strata_vk_native_hc_bytes(component_.get());
        native_hc_need(metadata<=budget-bound,"native HC serial: norm metadata budget");
        if(down_) {
            mix_.reset(strata_vk_native_hc_mix_create(backend,(dir+"/mix-effective.spv").c_str(),budget-bound-metadata));
            const size_t mix_bytes=strata_vk_native_hc_mix_bytes(mix_.get());
            native_hc_need(mix_bytes<=budget-bound-metadata,"native HC serial: mix metadata budget");
            metadata+=mix_bytes;
        }
        if(pending) {
            pending_.reset(strata_vk_native_hc_pending_create(backend,(dir+"/pending-effective.spv").c_str(),budget-bound-metadata));
            const size_t extra=strata_vk_native_hc_pending_bytes(pending_.get());
            native_hc_need(extra<=budget-bound-metadata,"native HC pending: metadata graph budget");metadata+=extra;
        }
        native_hc_need(metadata<=budget-bound,"native HC serial: component graph budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));native_hc_need(bool(buffer_),"native HC serial: scratch allocation");
        const auto actual=ggml_backend_buffer_get_size(buffer_.get());native_hc_need(actual<=budget-metadata,"native HC serial: actual graph budget");
        bytes_=actual+metadata;
    }
    size_t bytes()const{return bytes_;}
    uint64_t calls()const{return calls_;}
    uint64_t mix_calls()const{return mix_calls_;}
    uint64_t pending_calls()const{return pending_calls_;}
    // Only the first successful attention graph is observed. This is a bounded
    // diagnostic readback, not a weight extraction or numerical fallback.
    void record_first() {
        if(first_recorded_ || calls_==0 || (mix_&&mix_calls_==0))return;
        for(auto pair:{std::pair{"input.f32le",residual},std::pair{"normalized.f32le",normalized},
                       std::pair{"rs.f32le",rs},std::pair{"inject.f32le",inject}}) {
            std::vector<float> values(size_t(ggml_nelements(pair.second)));
            ggml_backend_tensor_get(pair.second,values.data(),0,values.size()*sizeof(float));
            auto* f=std::fopen((receipt_dir_+"/first-"+pair.first).c_str(),"wbx");
            native_hc_need(f,"native HC serial: exclusive initial receipt");
            const bool written=std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();
            const bool closed=std::fclose(f)==0;
            native_hc_need(written&&closed,"native HC serial: initial receipt write failed");
        }
        if(mix_)for(auto pair:{std::pair{"low-pre",low_pre},std::pair{"low",low},std::pair{"low-exp",low_exp},std::pair{"up-pre",up_pre},std::pair{"gates",gates},std::pair{"up-exp",up_exp},std::pair{"mixed",mixed}}) {
            std::vector<float> v(size_t(ggml_nelements(pair.second)));ggml_backend_tensor_get(pair.second,v.data(),0,v.size()*4);
            auto* f=std::fopen((receipt_dir_+"/first-"+pair.first+".f32le").c_str(),"wbx");native_hc_need(f,"native HC serial: initial mix receipt");
            const bool written=std::fwrite(v.data(),4,v.size(),f)==v.size();const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native HC serial: mix receipt write");
        }
        if(pending_)for(auto pair:{std::pair{"unapplied-residual",pending_input},std::pair{"previous-output",previous_output},std::pair{"previous-injection",previous_inject},std::pair{"pending-gate",pending_gate},std::pair{"pending-exp",pending_exp}}) {
            std::vector<float> v(size_t(ggml_nelements(pair.second)));ggml_backend_tensor_get(pair.second,v.data(),0,v.size()*4);
            auto* f=std::fopen((receipt_dir_+"/first-"+pair.first+".f32le").c_str(),"wbx");native_hc_need(f,"native HC pending: exclusive tuple receipt");
            const bool written=std::fwrite(v.data(),4,v.size(),f)==v.size();const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native HC pending: tuple receipt write");
        }
        first_recorded_=true;
    }
private:
    void read(){
        native_hc_need(calls_!=UINT64_MAX&&mix_calls_!=UINT64_MAX,"native HC serial: call counter exhausted");
        strata_vk_native_hc_run(component_.get(),residual,norm_,weight_,normalized,rs,inject,1);++calls_;
        if(mix_){strata_vk_native_hc_mix_run(mix_.get(),normalized,down_,up_,low_pre,low,low_exp,up_pre,gates,up_exp,mixed);++mix_calls_;}
    }
public:
    void run(){native_hc_need(!pending_,"native HC pending: explicit tuple upload required");read();}
    void run_pending(){
        native_hc_need(pending_&&pending_calls_!=UINT64_MAX&&calls_!=UINT64_MAX&&mix_calls_!=UINT64_MAX,"native HC pending: owner/counter required");
        strata_vk_native_hc_pending_run(pending_.get(),pending_input,previous_output,previous_inject,residual,pending_gate,pending_exp);++pending_calls_;
        read();
    }
};
}
