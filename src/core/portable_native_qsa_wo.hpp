#pragma once
#include "native_qsa_wo_domain.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
#include "ggml-alloc.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>

namespace strata::core::portable_detail {
// Borrow the already admitted Q8_0 weight buffer and the native attention
// owner's live gated output. Only the packed/output banks and metadata are new.
class PortableNativeQsaWo {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> quant_{nullptr,strata_vk_native_stages_destroy};
 std::unique_ptr<StrataVkQ8Wo,decltype(&strata_vk_q8_wo_destroy)> wo_{nullptr,strata_vk_q8_wo_destroy};
 ggml_tensor *weight_=nullptr,*packed_=nullptr,*output_=nullptr;
 size_t bytes_=0;std::filesystem::path dir_;std::string role_;bool record_;std::array<bool,8>recorded_{};
 static Context context(){Context c(ggml_init({65536,nullptr,true}),ggml_free);if(!c)throw std::runtime_error("QSA WO: metadata");return c;}
 static size_t estimate_buffer(ggml_backend_t b,size_t words){auto c=context();ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,int64_t(words));return ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b));}
 static void save(const std::filesystem::path&p,const void* data,size_t n){std::ofstream o(p,std::ios::binary);o.write(static_cast<const char*>(data),n);if(!o)throw std::runtime_error("QSA WO: receipt write");}
public:
 struct Candidate{std::vector<float>output;std::array<uint8_t,6912>packed;};
 static size_t estimate(ggml_backend_t b){
    auto c=context();ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,6912/4);ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2560);
    size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b));
    return bank+estimate_buffer(b,128)+estimate_buffer(b,1);
 }
 PortableNativeQsaWo(ggml_backend_t b,ggml_tensor*weight,const void*role_bytes,size_t role_size,const std::string&directory,size_t budget,bool record):dir_(directory),record_(record){
    if(!b||!weight||weight->type!=GGML_TYPE_Q8_0||ggml_nbytes(weight)!=16711680||!weight->buffer||!ggml_is_contiguous(weight)||estimate(b)>budget)
        throw std::runtime_error("QSA WO: resident Q8_0 weight or aligned budget unavailable");
    role_=ggml_get_name(weight);if(!std::filesystem::create_directory(dir_))throw std::runtime_error("QSA WO: fresh receipt namespace required");
    ctx_=context();weight_=ggml_view_1d(ctx_.get(),weight,6144*2560,0);
    // The view retains the resident Q8_0 owner; reinterpret its exact bytes as
    // words for the existing descriptor-based consumer without another copy.
    weight_->type=GGML_TYPE_I32;weight_->ne[0]=16711680/4;
    weight_->nb[0]=4;weight_->nb[1]=weight_->nb[2]=weight_->nb[3]=16711680;
    if(ggml_backend_view_init(weight_)!=GGML_STATUS_SUCCESS)throw std::runtime_error("QSA WO: resident view binding");
    qsa_wo_weight_domain(role_bytes,role_size,role_);
    std::vector<uint8_t> check(65536);const auto*expected=static_cast<const uint8_t*>(role_bytes);
    for(size_t offset=0;offset<role_size;offset+=check.size()){
        size_t n=std::min(check.size(),role_size-offset);ggml_backend_tensor_get(weight_,check.data(),offset,n);
        if(std::memcmp(check.data(),expected+offset,n)!=0)throw std::runtime_error("QSA WO: resident weight bytes differ from model role");
    }
    packed_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_I32,6912/4);output_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,2560);
    size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(b));
    if(bank>=budget)throw std::runtime_error("QSA WO: no owner budget");
    quant_.reset(strata_vk_native_stages_create_quant_rcp_observation(b,(dir_/"quant-effective.spv").c_str(),budget-bank));
    const size_t qb=strata_vk_native_stages_bytes(quant_.get());
    if(qb>budget-bank)throw std::runtime_error("QSA WO: quantizer owner budget");
    // The WO owner's span guard counts the complete borrowed weight buffer.
    // That buffer is already charged to weight admission, so add only that
    // existing allocation to its local guard, without changing either cap.
    size_t owner_budget=budget-qb,weight_bytes=ggml_backend_buffer_get_size(weight->buffer);
    if(weight_bytes>SIZE_MAX-owner_budget)throw std::runtime_error("QSA WO: weight budget overflow");
    wo_.reset(strata_vk_q8_wo_create(b,(dir_/"wo-effective.spv").c_str(),owner_budget+weight_bytes));
    buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),b));
    if(!buffer_)throw std::runtime_error("QSA WO: output bank allocation");
    bytes_=ggml_backend_buffer_get_size(buffer_.get())+qb+strata_vk_q8_wo_bytes(wo_.get());
    if(bytes_>estimate(b)||bytes_>budget)throw std::runtime_error("QSA WO: actual allocation exceeds bound");
 }
 size_t bytes()const{return bytes_;}
 ggml_tensor* output()const{return output_;}
 uint64_t calls(unsigned i)const{switch(i){case 0:return strata_vk_native_stages_quant_submitted(quant_.get());case 1:return strata_vk_native_stages_quant_completed(quant_.get());case 2:return strata_vk_q8_wo_submitted(wo_.get());case 3:return strata_vk_q8_wo_completed(wo_.get());case 4:return strata_vk_native_stages_quant_submission_uncertain(quant_.get());case 5:return strata_vk_q8_wo_submission_uncertain(wo_.get());default:return 0;}}
 Candidate run(ggml_tensor* gated,const std::vector<float>& expected){
    qsa_wo_input_domain(expected);
    if(!gated||ggml_nbytes(gated)!=6144*4)throw std::runtime_error("QSA WO: actual gated tensor unavailable");
    std::vector<float> actual(6144);ggml_backend_tensor_get(gated,actual.data(),0,6144*4);
    if(std::memcmp(actual.data(),expected.data(),6144*4)!=0)throw std::runtime_error("QSA WO: gated tensor differs from attention candidate");
    const int32_t starts[]={0,1};strata_vk_native_stages_run(quant_.get(),0,gated,gated,gated,packed_,6144,starts,2);
    Candidate c;c.output.resize(2560);ggml_backend_tensor_get(packed_,c.packed.data(),0,c.packed.size());qsa_wo_packed_domain(c.packed.data(),c.packed.size());
    strata_vk_q8_wo_run(wo_.get(),weight_,packed_,output_,2560);
    ggml_backend_tensor_get(output_,c.output.data(),0,c.output.size()*sizeof(float));
    for(float v:c.output)if(!qsa_wo_normal_or_zero(v))throw std::runtime_error("QSA WO: unsupported output");
    return c;
 }
 void consumed(uint64_t position,const std::vector<float>&gated,ggml_tensor*leaf,const Candidate&c){
    std::vector<float>actual(2560);ggml_backend_tensor_get(leaf,actual.data(),0,actual.size()*4);
    if(std::memcmp(actual.data(),c.output.data(),actual.size()*4)!=0)throw std::runtime_error("QSA WO: pending consumer ignored native output");
    if(record_&&position<8&&!recorded_[position]){
        auto d=dir_/("first-"+std::to_string(position));if(!std::filesystem::create_directory(d))throw std::runtime_error("QSA WO: receipt exists");
        save(d/"gated.f32le",gated.data(),gated.size()*4);save(d/"qx.bin",c.packed.data(),c.packed.size());
        save(d/"projected.f32le",c.output.data(),c.output.size()*4);save(d/"consumed.f32le",actual.data(),actual.size()*4);
        std::ofstream o(d/"role.json");o<<"{\"name\":\""<<role_<<"\",\"type\":\"Q8_0\",\"shape\":[6144,2560],\"resident_bytes_match_role\":true,\"producer_buffer_consumed\":true,\"qualified\":false}\n";
        if(!o)throw std::runtime_error("QSA WO: role receipt write");recorded_[position]=true;
    }
 }
};
}
