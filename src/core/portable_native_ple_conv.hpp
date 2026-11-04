#pragma once
#include "strata_vk_native_ple_conv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "native_hc_layout.hpp"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    uint64_t calls_=0;
    bool ready_=false,recorded_=false;
    std::filesystem::path directory_;
    std::array<std::vector<float>,4> before_;
    std::vector<float> next_,result_;
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
    // Native pack kind F16InF32 narrows the original F32 role with this exact
    // converter; raw F16 roles are copied. The packer refuses nonfinite and
    // F16-overflowing values instead of silently changing them.
    static std::vector<uint16_t> native_weights(const void* data,size_t bytes,uint32_t type){
        need(data&&((type==0&&bytes==40960*4)||(type==1&&bytes==40960*2)),"PLE conv: expected F32 or F16 role");
        const auto* p=static_cast<const uint8_t*>(data);
        std::vector<uint16_t> out(40960);
        for(size_t i=0;i<out.size();++i){
            if(type==0){
                float value;std::memcpy(&value,p+i*4,4);
                need(std::isfinite(value),"PLE conv: nonfinite F32 weight");
                out[i]=strata::kernels::f16_from_f32(value);
            } else std::memcpy(&out[i],p+i*2,2);
            need((out[i]&0x7c00u)!=0x7c00u,"PLE conv: nonfinite or overflowing F16 weight");
        }
        return out;
    }
    static size_t estimate(ggml_backend_t backend){
        need(backend,"PLE conv: backend");
        auto c=context();std::array<ggml_tensor*,10> t{};leaves(c.get(),t);
        auto type=ggml_backend_get_default_buffer_type(backend);
        const auto bank=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),type);
        auto metadata=context();ggml_new_tensor_1d(metadata.get(),GGML_TYPE_I32,1);
        const auto status=ggml_backend_alloc_ctx_tensors_from_buft_size(metadata.get(),type);
        need(bank<=SIZE_MAX-status,"PLE conv: budget overflow");
        return bank+status;
    }
    PortableNativePleConv(ggml_backend_t backend,const void* weights,size_t weight_bytes,uint32_t role_type,const char* module,size_t budget):backend_(backend),directory_(std::filesystem::path(module).parent_path()){
        need(backend,"PLE conv: backend required");
        const auto packed=native_weights(weights,weight_bytes,role_type);
        const auto admitted=estimate(backend);
        need(admitted<=budget,"PLE conv: allocation budget");
        ctx_=context();leaves(ctx_.get(),tensors_);
        const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        need(bound<=budget,"PLE conv: allocation budget");
        component_.reset(strata_vk_native_ple_conv_create(backend,module,budget));need(bool(component_),"PLE conv: component");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));need(bool(buffer_),"PLE conv: allocation");
        const auto actual=ggml_backend_buffer_get_size(buffer_.get()),extra=strata_vk_native_ple_conv_bytes(component_.get());
        need(actual<=budget&&extra<=budget-actual,"PLE conv: actual budget");bytes_=actual+extra;
        need(bytes_<=admitted,"PLE conv: allocation exceeds estimate");
        ggml_backend_tensor_set(tensors_[2],packed.data(),0,packed.size()*sizeof(uint16_t));
    }
    size_t bytes()const{return bytes_;}
    uint64_t calls()const{return calls_;}
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
            if(std::memcmp(source.data(),before[i].data(),source.size()*4)!=0){
                const char* names[]={"history","normalized","hidden","gated"};
                const auto* src=reinterpret_cast<const unsigned char*>(source.data());
                const auto* dst=reinterpret_cast<const unsigned char*>(before[i].data());
                size_t byte=0;while(byte<source.size()*4&&src[byte]==dst[byte])++byte;
                std::string logical="unavailable";
                if(original[i]==copy[i]->src[0])logical="scheduled-source";
                else if(original[i]->data&&(original[i]->flags&GGML_TENSOR_FLAG_OUTPUT)){
                    try{
                        span(original[i],source.size()*4);
                        std::vector<float> value(source.size());ggml_backend_tensor_get(original[i],value.data(),0,value.size()*4);
                        logical=std::memcmp(value.data(),before[i].data(),value.size()*4)==0?"destination":
                                std::memcmp(value.data(),source.data(),value.size()*4)==0?"scheduled-source":"neither";
                    }catch(const std::exception&){}
                }
                throw std::runtime_error(std::string("PLE conv: prefix copy differs: ")+names[i]+" byte "+std::to_string(byte)+" logical="+logical);
            }
            for(float x:before[i])native_hc_normal(x);
        }
        strata_vk_native_ple_conv_run(component_.get(),tensors_.data());
        for(size_t i=0;i<4;++i){std::vector<float> after(before[i].size());ggml_backend_tensor_get(input(i),after.data(),0,after.size()*4);need(std::memcmp(before[i].data(),after.data(),after.size()*4)==0,"PLE conv: input changed");}
        std::vector<float> next(sizes_[5]);ggml_backend_tensor_get(history(),next.data(),0,next.size()*4);
        for(size_t c=0;c<10240;++c){for(size_t j=0;j<8;++j)need(std::memcmp(&next[9*c+j],&before[0][9*c+j+1],4)==0,"PLE conv: candidate history differs");need(std::memcmp(&next[9*c+8],&before[1][c],4)==0,"PLE conv: appended history differs");}
        for(size_t i=6;i<10;++i){std::vector<float> out(sizes_[i]);ggml_backend_tensor_get(tensors_[i],out.data(),0,out.size()*4);for(float x:out)native_hc_normal(x);}
        before_=std::move(before);next_=std::move(next);result_.resize(sizes_[7]);
        ggml_backend_tensor_get(result(),result_.data(),0,result_.size()*4);
        ready_=true;++calls_;
    }
    // The suffix must read these same persistent banks. This is a candidate
    // receipt, emitted before the FFN and token transaction commit.
    void verify_and_record(const ggml_tensor* result_view,const ggml_tensor* history_view,
                           uint64_t position,int32_t token,const std::array<int32_t,2>& previous){
        need(ready_&&result_view&&history_view&&result_view->view_src==result()&&history_view->view_src==history(),
             "PLE conv: downstream producer identity");
        span(result_view,10240*4);span(history_view,92160*4);
        auto unchanged=[&](ggml_tensor* t,const std::vector<float>& expected){
            std::vector<float> got(expected.size());ggml_backend_tensor_get(t,got.data(),0,got.size()*4);
            need(std::memcmp(got.data(),expected.data(),got.size()*4)==0,"PLE conv: candidate changed");
        };
        for(size_t i=0;i<4;++i)unchanged(input(i),before_[i]);
        unchanged(history(),next_);unchanged(result(),result_);
        if(recorded_)return;
        const auto path=directory_/"first-candidate";
        need(std::filesystem::create_directory(path),"PLE conv: candidate receipt exists");
        const char* names[]={"history.f32le","normalized.f32le","weight.f16le","hidden.f32le","gated.f32le","next-history.f32le","conv.f32le","result.f32le","preactivation.f32le","exponent.f32le"};
        for(size_t i=0;i<10;++i){std::vector<char> data(ggml_nbytes(tensors_[i]));ggml_backend_tensor_get(tensors_[i],data.data(),0,data.size());std::ofstream f(path/names[i],std::ios::binary);f.write(data.data(),data.size());f.flush();need(bool(f),"PLE conv: receipt write");}
        std::ofstream meta(path/"metadata.json");
        meta<<"{\"position\":"<<position<<",\"token\":"<<token<<",\"previous\":["<<previous[0]<<","<<previous[1]<<"],\"producer_banks\":true,\"graph_completed\":true,\"committed\":false,\"qualified\":false}\n";
        meta.flush();need(bool(meta),"PLE conv: metadata write");recorded_=true;
    }
};
}
