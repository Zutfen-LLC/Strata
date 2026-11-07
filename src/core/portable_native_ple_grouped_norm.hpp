#pragma once
#include "strata_vk_native_ple_grouped_norm.hpp"
#include "portable_native_ple_host_bound.hpp"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core::portable_detail {
// Three separate candidate banks, in native key/query/gated-value order.
// This owner does not advance PLE history or commit any model state.
class PortableNativePleGroupedNorm {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    Context context_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativePleGroupedNorm,decltype(&strata_vk_native_ple_grouped_norm_destroy)> component_{nullptr,strata_vk_native_ple_grouped_norm_destroy};
    ggml_backend_t backend_;
    std::array<ggml_tensor*,3> inputs_{},gammas_{},outputs_{},diagnostics_{};
    std::array<const ggml_tensor*,3> copies_{},producers_{};
    std::array<bool,3> ready_{};
    size_t bytes_=0, rows_=1;
    static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
    static Context context(){Context c(ggml_init({16384,nullptr,true}),ggml_free);need(bool(c),"PLE grouped norm: metadata");return c;}
    static void leaves(ggml_context* c,std::array<ggml_tensor*,3>& in,std::array<ggml_tensor*,3>& gamma,
                       std::array<ggml_tensor*,3>& out,std::array<ggml_tensor*,3>& diag,size_t rows){
        for(size_t i=0;i<3;++i){
            in[i]=ggml_new_tensor_3d(c,GGML_TYPE_F32,2560,4,rows);
            gamma[i]=ggml_new_tensor_2d(c,GGML_TYPE_F32,2560,4);
            out[i]=ggml_new_tensor_3d(c,GGML_TYPE_F32,2560,4,rows);
            diag[i]=ggml_new_tensor_3d(c,GGML_TYPE_F32,3,4,rows);
            for(auto* t:{in[i],gamma[i],out[i],diag[i]})ggml_set_input(t);
        }
    }
    void span(const ggml_tensor* t)const{
        need(t&&t->type==GGML_TYPE_F32&&ggml_is_contiguous(t)&&t->buffer&&ggml_nbytes(t)==40960*rows_,"PLE grouped norm: producer span");
        auto type=ggml_backend_buffer_get_type(t->buffer);auto dev=ggml_backend_buft_get_device(type);auto owner=ggml_backend_get_device(backend_);
        need(type==ggml_backend_get_default_buffer_type(backend_)&&owner&&(dev?dev==owner:(ggml_backend_buft_is_host(type)&&ggml_backend_dev_type(owner)==GGML_BACKEND_DEVICE_TYPE_CPU)),"PLE grouped norm: producer device");
    }
public:
    // Persistent context and two simultaneous scheduled-copy comparison spans.
    static constexpr size_t host_bound(size_t rows){return ple_host_add(16384,ple_host_mul(rows,2*40960));}
    static size_t estimate(ggml_backend_t backend,size_t rows=1){
        need(backend&&(rows==1||rows==2),"PLE grouped norm: backend or rows");auto c=context();std::array<ggml_tensor*,3> a{},b{},d{},e{};leaves(c.get(),a,b,d,e,rows);
        auto type=ggml_backend_get_default_buffer_type(backend);const size_t banks=ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),type);
        auto status=context();ggml_new_tensor_1d(status.get(),GGML_TYPE_I32,1);const size_t extra=ggml_backend_alloc_ctx_tensors_from_buft_size(status.get(),type);
        need(banks<=SIZE_MAX-extra,"PLE grouped norm: budget overflow");return banks+extra;
    }
    PortableNativePleGroupedNorm(ggml_backend_t backend,const std::array<const float*,3>& gamma,const char* module,size_t budget,size_t rows=1):backend_(backend),rows_(rows){
        need(backend&&module&&gamma[0]&&gamma[1]&&gamma[2]&&gamma[0]!=gamma[1]&&gamma[0]!=gamma[2]&&gamma[1]!=gamma[2],"PLE grouped norm: distinct gamma roles required");
        for(auto* role:gamma)for(size_t i=0;i<10240;++i)need(std::isfinite(role[i])&&(std::fpclassify(role[i])!=FP_SUBNORMAL),"PLE grouped norm: unsupported gamma");
        const auto admitted=estimate(backend,rows_);need(admitted<=budget,"PLE grouped norm: allocation budget");
        context_=context();leaves(context_.get(),inputs_,gammas_,outputs_,diagnostics_,rows_);
        component_.reset(strata_vk_native_ple_grouped_norm_create(backend,module,budget));need(bool(component_),"PLE grouped norm: component");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));need(bool(buffer_),"PLE grouped norm: allocation");
        const size_t actual=ggml_backend_buffer_get_size(buffer_.get()),extra=strata_vk_native_ple_grouped_norm_bytes(component_.get());
        need(actual<=budget&&extra<=budget-actual&&actual+extra<=admitted,"PLE grouped norm: actual allocation budget");bytes_=actual+extra;
        ggml_backend_buffer_clear(buffer_.get(),0);
        for(size_t i=0;i<3;++i)ggml_backend_tensor_set(gammas_[i],gamma[i],0,40960);
    }
    size_t bytes()const{return bytes_;}
    uint64_t calls()const{return strata_vk_native_ple_grouped_norm_calls(component_.get());}
    ggml_tensor* input(size_t i)const{return inputs_.at(i);}
    ggml_tensor* gamma(size_t i)const{return gammas_.at(i);}
    ggml_tensor* output(size_t i)const{return outputs_.at(i);}
    bool ready(size_t i)const{return ready_.at(i);}
    void begin(){ready_.fill(false);}
    void bind(size_t i,const ggml_tensor* copy,const ggml_tensor* producer){
        need(i<3&&copy&&producer&&copy->op==GGML_OP_CPY&&copy->src[0]==producer&&copy->src[1]==inputs_[i]&&copy->view_src==inputs_[i],"PLE grouped norm: logical producer binding");
        copies_[i]=copy;producers_[i]=producer;
    }
    void run(size_t i,const ggml_tensor* copy,const ggml_tensor* producer){
        need(i<3,"PLE grouped norm: role index");ready_[i]=false;
        need(copy&&copy==copies_[i]&&producer==producers_[i]&&copy->op==GGML_OP_CPY&&copy->view_src==inputs_[i]&&copy->src[0],"PLE grouped norm: producer binding differs");
        span(inputs_[i]);span(copy->src[0]);std::vector<unsigned char> actual(40960*rows_),expected(40960*rows_);
        ggml_backend_tensor_get(inputs_[i],actual.data(),0,actual.size());ggml_backend_tensor_get(copy->src[0],expected.data(),0,expected.size());
        need(actual==expected,"PLE grouped norm: scheduled copy differs");
        need(strata_vk_native_ple_grouped_norm_run(component_.get(),inputs_[i],gammas_[i],outputs_[i],diagnostics_[i])==0,"PLE grouped norm: finite domain refused");ready_[i]=true;
    }
};
}
