#pragma once
#include "strata_vk_native_ple_conv.hpp"
#include "native_hc_layout.hpp"
#include "ggml-alloc.h"
#include <array>
#include <cstring>
#include <memory>
#include <vector>

namespace strata::core::portable_detail {
// The caller binds copies from the PLE prefix to persistent device banks.
// History and result are candidates: this owner never commits session state.
class PortableNativePleConv {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    Context ctx_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativePleConv,decltype(&strata_vk_native_ple_conv_destroy)> component_{nullptr,strata_vk_native_ple_conv_destroy};
    ggml_backend_t backend_;
    std::array<ggml_tensor*,10> tensors_{};
    std::array<const ggml_tensor*,4> copies_{},originals_{};
    size_t bytes_=0;
    bool ready_=false;
    static constexpr std::array<size_t,10> sizes_{92160,10240,20480,10240,10240,92160,10240,10240,10240,10240};
    static constexpr std::array<size_t,4> slots_{0,1,3,4};
    static void need(bool b,const char* message){if(!b)throw std::runtime_error(message);}
    static Context context(){Context c(ggml_init({16384,nullptr,true}),ggml_free);need(bool(c),"PLE conv: metadata");return c;}
    static void leaves(ggml_context* c,std::array<ggml_tensor*,10>& t){
        for(size_t i=0;i<t.size();++i){t[i]=ggml_new_tensor_1d(c,i==2?GGML_TYPE_I32:GGML_TYPE_F32,sizes_[i]);ggml_set_input(t[i]);}
    }
    void span(const ggml_tensor* t,size_t bytes) const {
        need(t&&t->type==GGML_TYPE_F32&&ggml_is_contiguous(t)&&t->buffer&&ggml_nbytes(t)==bytes,"PLE conv: producer span");
        auto type=ggml_backend_buffer_get_type(t->buffer);auto dev=ggml_backend_buft_get_device(type);auto owner=ggml_backend_get_device(backend_);
        need(type==ggml_backend_get_default_buffer_type(backend_)&&owner&&(dev?dev==owner:(ggml_backend_buft_is_host(type)&&ggml_backend_dev_type(owner)==GGML_BACKEND_DEVICE_TYPE_CPU)),"PLE conv: producer device");
    }
public:
    PortableNativePleConv(ggml_backend_t backend,const void* weights,size_t weight_bytes,const char* module,size_t budget):backend_(backend){
        need(backend&&weights&&weight_bytes==81920,"PLE conv: native F16 role required");
        const auto* p=static_cast<const uint8_t*>(weights);
        for(size_t i=0;i<weight_bytes;i+=2){uint16_t h;std::memcpy(&h,p+i,2);need((h&0x7c00u)!=0x7c00u,"PLE conv: nonfinite weight");}
        ctx_=context();leaves(ctx_.get(),tensors_);
        const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        need(bound<=budget,"PLE conv: allocation budget");
        component_.reset(strata_vk_native_ple_conv_create(backend,module,budget));need(bool(component_),"PLE conv: component");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));need(bool(buffer_),"PLE conv: allocation");
        const auto actual=ggml_backend_buffer_get_size(buffer_.get()),extra=strata_vk_native_ple_conv_bytes(component_.get());
        need(actual<=budget&&extra<=budget-actual,"PLE conv: actual budget");bytes_=actual+extra;
        ggml_backend_tensor_set(tensors_[2],weights,0,weight_bytes);
    }
    size_t bytes()const{return bytes_;}
    ggml_tensor* input(size_t i)const{return tensors_.at(slots_.at(i));}
    ggml_tensor* history()const{return tensors_[5];}
    ggml_tensor* convolution()const{return tensors_[6];}
    ggml_tensor* result()const{return tensors_[7];}
    bool ready()const{return ready_;}
    void bind(ggml_tensor* const copy[4],ggml_tensor* const original[4]){
        ready_=false;
        for(size_t i=0;i<4;++i){need(copy[i]&&original[i]&&copy[i]->op==GGML_OP_CPY&&copy[i]->src[0]==original[i]&&copy[i]->src[1]==input(i)&&copy[i]->view_src==input(i),"PLE conv: logical producer binding");copies_[i]=copy[i];originals_[i]=original[i];}
    }
    void run(ggml_tensor* const copy[4],ggml_tensor* const original[4]){
        ready_=false;
        std::array<std::vector<float>,4> before;
        for(size_t i=0;i<4;++i){
            need(copy[i]==copies_[i]&&original[i]==originals_[i]&&copy[i]->op==GGML_OP_CPY&&copy[i]->src[1]==input(i)&&copy[i]->view_src==input(i),"PLE conv: producer binding changed");
            span(copy[i]->src[0],sizes_[slots_[i]]*4);span(input(i),sizes_[slots_[i]]*4);
            before[i].resize(sizes_[slots_[i]]);ggml_backend_tensor_get(input(i),before[i].data(),0,before[i].size()*4);
            std::vector<float> source(before[i].size());ggml_backend_tensor_get(copy[i]->src[0],source.data(),0,source.size()*4);
            need(std::memcmp(source.data(),before[i].data(),source.size()*4)==0,"PLE conv: prefix copy differs");
            for(float x:before[i])native_hc_normal(x);
        }
        strata_vk_native_ple_conv_run(component_.get(),tensors_.data());
        for(size_t i=0;i<4;++i){std::vector<float> after(before[i].size());ggml_backend_tensor_get(input(i),after.data(),0,after.size()*4);need(std::memcmp(before[i].data(),after.data(),after.size()*4)==0,"PLE conv: input changed");}
        std::vector<float> next(sizes_[5]);ggml_backend_tensor_get(history(),next.data(),0,next.size()*4);
        for(size_t c=0;c<10240;++c){for(size_t j=0;j<8;++j)need(std::memcmp(&next[9*c+j],&before[0][9*c+j+1],4)==0,"PLE conv: candidate history differs");need(std::memcmp(&next[9*c+8],&before[1][c],4)==0,"PLE conv: appended history differs");}
        for(size_t i=6;i<10;++i){std::vector<float> out(sizes_[i]);ggml_backend_tensor_get(tensors_[i],out.data(),0,out.size()*4);for(float x:out)native_hc_normal(x);}
        ready_=true;
    }
};
}
