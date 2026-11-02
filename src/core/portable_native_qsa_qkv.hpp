#pragma once
#include "native_qsa_qkv_layout.hpp"
#include "native_qsa_wo_domain.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
#include "ggml-alloc.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>

namespace strata::core::portable_detail {
// One quantizer owns the packed image consumed by all three native projections.
// Candidate outputs remain scratch until the existing layer transaction commits.
class PortableNativeQsaQkv {
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
 Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
 std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> quant_{nullptr,strata_vk_native_stages_destroy};
 std::unique_ptr<StrataVkQ8Wo,decltype(&strata_vk_q8_wo_destroy)> component_{nullptr,strata_vk_q8_wo_destroy};
 std::array<ggml_tensor*,3>weight_{},output_{};ggml_tensor*packed_=nullptr;
 size_t bytes_=0;std::filesystem::path dir_;bool record_;std::array<bool,8>recorded_{};
 static constexpr std::array<size_t,3> rows_{{12288,512,512}};
 static Context context(){Context c(ggml_init({65536,nullptr,true}),ggml_free);if(!c)throw std::runtime_error("QSA QKV: metadata");return c;}
 static size_t metadata(ggml_backend_t b,size_t words){auto c=context();ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,int64_t(words));return ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b));}
 static void save(const std::filesystem::path&p,const void* data,size_t n){std::ofstream o(p,std::ios::binary);o.write(static_cast<const char*>(data),n);if(!o)throw std::runtime_error("QSA QKV: receipt write");}
public:
 struct Candidate {std::array<std::vector<float>,3>output;std::array<uint8_t,qsa_qkv_packed_bytes>packed;};
 static size_t estimate(ggml_backend_t b){auto c=context();ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,qsa_qkv_packed_bytes/4);for(size_t n:rows_)ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,n);
    return ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b))+metadata(b,128)+metadata(b,1);}
 PortableNativeQsaQkv(ggml_backend_t b,const std::array<ggml_tensor*,3>& weights,const std::array<const void*,3>& role_bytes,
                      const std::array<size_t,3>& role_sizes,const std::string& directory,size_t budget,bool record):dir_(directory),record_(record){
    if(!b||estimate(b)>budget||!std::filesystem::create_directory(dir_))throw std::runtime_error("QSA QKV: backend, budget or fresh receipt required");
    ctx_=context();size_t borrowed=0;std::array<ggml_backend_buffer_t,3> seen{};
    for(size_t i=0;i<3;++i){auto*w=weights[i];const size_t n=rows_[i]*qsa_qkv_row_bytes;
        if(!w||w->type!=GGML_TYPE_Q8_0||ggml_nbytes(w)!=n||!w->buffer||!ggml_is_contiguous(w)||!role_bytes[i]||role_sizes[i]!=n)
            throw std::runtime_error("QSA QKV: resident weight shape or bytes unavailable");
        qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(role_bytes[i]),rows_[i]*qsa_qkv_blocks,34),"QSA QKV weight",true);
        weight_[i]=ggml_view_1d(ctx_.get(),w,2560*rows_[i],0);weight_[i]->type=GGML_TYPE_I32;weight_[i]->ne[0]=n/4;
        weight_[i]->nb[0]=4;weight_[i]->nb[1]=weight_[i]->nb[2]=weight_[i]->nb[3]=n;
        if(ggml_backend_view_init(weight_[i])!=GGML_STATUS_SUCCESS)throw std::runtime_error("QSA QKV: resident view binding");
        std::vector<uint8_t>check(65536);auto*expected=static_cast<const uint8_t*>(role_bytes[i]);
        for(size_t offset=0;offset<n;offset+=check.size()){size_t amount=std::min(check.size(),n-offset);ggml_backend_tensor_get(weight_[i],check.data(),offset,amount);
            if(std::memcmp(check.data(),expected+offset,amount)!=0)throw std::runtime_error("QSA QKV: resident bytes differ from role");}
        bool duplicate=false;for(size_t j=0;j<i;++j)duplicate|=seen[j]==w->buffer;seen[i]=w->buffer;
        if(!duplicate){size_t amount=ggml_backend_buffer_get_size(w->buffer);if(amount>SIZE_MAX-borrowed)throw std::runtime_error("QSA QKV: weight budget overflow");borrowed+=amount;}
    }
    packed_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_I32,qsa_qkv_packed_bytes/4);for(size_t i=0;i<3;++i){output_[i]=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,rows_[i]);ggml_set_output(output_[i]);}
    size_t bank=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(b));if(bank>=budget)throw std::runtime_error("QSA QKV: no owner budget");
    quant_.reset(strata_vk_native_stages_create_quant_rcp_observation(b,(dir_/"quant-effective.spv").c_str(),budget-bank));
    const size_t qb=strata_vk_native_stages_bytes(quant_.get());if(qb>budget-bank||borrowed>SIZE_MAX-(budget-qb))throw std::runtime_error("QSA QKV: owner budget");
    component_.reset(strata_vk_qsa_qkv_create(b,(dir_/"qkv-effective.spv").c_str(),budget-qb+borrowed));
    buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),b));if(!buffer_)throw std::runtime_error("QSA QKV: bank allocation");
    bytes_=ggml_backend_buffer_get_size(buffer_.get())+qb+strata_vk_q8_wo_bytes(component_.get());
    if(bytes_>estimate(b)||bytes_>budget)throw std::runtime_error("QSA QKV: allocation exceeds bound");
 }
 size_t bytes()const{return bytes_;}
 ggml_tensor* projection(size_t role) const { return output_.at(role); }
 uint64_t calls(unsigned i)const{switch(i){case 0:return strata_vk_native_stages_quant_submitted(quant_.get());case 1:return strata_vk_native_stages_quant_completed(quant_.get());case 2:return strata_vk_q8_wo_submitted(component_.get());case 3:return strata_vk_q8_wo_completed(component_.get());case 4:return strata_vk_native_stages_quant_submission_uncertain(quant_.get());case 5:return strata_vk_q8_wo_submission_uncertain(component_.get());default:return 0;}}
 Candidate run(ggml_tensor* mixed){
    if(!mixed||ggml_nbytes(mixed)!=2560*4)throw std::runtime_error("QSA QKV: actual mixed input unavailable");
    std::vector<float>actual(2560);ggml_backend_tensor_get(mixed,actual.data(),0,actual.size()*4);qsa_q8_input_domain(actual,qsa_qkv_blocks);
    const int32_t starts[]={0,1};strata_vk_native_stages_run(quant_.get(),0,mixed,mixed,mixed,packed_,2560,starts,2);
    Candidate c;ggml_backend_tensor_get(packed_,c.packed.data(),0,c.packed.size());
    qsa_wo_require_scales(qsa_wo_scale_classes(c.packed.data(),qsa_qkv_blocks,36),"producer Q8_1");
    for(size_t i=0;i<3;++i){c.output[i].resize(rows_[i]);strata_vk_q8_wo_run(component_.get(),weight_[i],packed_,output_[i],rows_[i]);
        ggml_backend_tensor_get(output_[i],c.output[i].data(),0,c.output[i].size()*4);
        for(float v:c.output[i])if(!qsa_wo_normal_or_zero(v))throw std::runtime_error("QSA QKV: unsupported output");}
    return c;
 }
 void consumed(uint64_t position,ggml_tensor*mixed,const std::array<ggml_tensor*,3>& leaves,const Candidate&c){
    std::array<std::vector<float>,3>actual;
    for(size_t i=0;i<3;++i){actual[i].resize(rows_[i]);ggml_backend_tensor_get(leaves[i],actual[i].data(),0,actual[i].size()*4);
        if(std::memcmp(actual[i].data(),c.output[i].data(),actual[i].size()*4))throw std::runtime_error("QSA QKV: downstream leaf differs from projection");}
    if(record_&&position<8&&!recorded_[position]){auto d=dir_/("first-"+std::to_string(position));if(!std::filesystem::create_directory(d))throw std::runtime_error("QSA QKV: receipt exists");
        std::vector<float>input(2560);ggml_backend_tensor_get(mixed,input.data(),0,input.size()*4);save(d/"mixed.f32le",input.data(),input.size()*4);save(d/"qx.bin",c.packed.data(),c.packed.size());
        const std::array<const char*,3>names{{"q.f32le","k.f32le","v.f32le"}},consumed{{"consumed-q.f32le","consumed-k.f32le","consumed-v.f32le"}};
        for(size_t i=0;i<3;++i){save(d/names[i],c.output[i].data(),c.output[i].size()*4);save(d/consumed[i],actual[i].data(),actual[i].size()*4);}
        std::ofstream report(d/"role.json");report<<"{\"roles\":[\"attn_q.weight\",\"attn_k.weight\",\"attn_v.weight\"],\"type\":\"Q8_0\",\"input\":2560,\"rows\":[12288,512,512],\"resident_bytes_match_roles\":true,\"one_packed_input\":true,\"qualified\":false}\n";
        if(!report)throw std::runtime_error("QSA QKV: role receipt write");recorded_[position]=true;}
 }
};
}
