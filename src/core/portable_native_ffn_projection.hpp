#pragma once
// Standalone fixed T1/T2 router matrix projection. No model admission.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core::portable_detail {
enum class PortableRouterMethod { legacy_f32, single_bf16 };
class PortableNativeFfnProjection {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    ggml_backend_t backend_;
    Context ctx_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    ggml_tensor *weight_=nullptr,*input_[2]{},*operand_[2]{},*output_[2]{};
    PortableRouterMethod method_;
    size_t graph_attempts_=0;
    ggml_cgraph* graph_[2]{};
    size_t bytes_=0,ready_=0;
    static void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
public:
    // The source role is copied once; backend must outlive this owner. This
    // budget covers the backend tensor buffer only, not metadata or workspace.
    PortableNativeFfnProjection(ggml_backend_t backend,const ggml_tensor* role,size_t budget,
                                PortableRouterMethod method=PortableRouterMethod::legacy_f32):backend_(backend),method_(method){
        need(method==PortableRouterMethod::legacy_f32 || method==PortableRouterMethod::single_bf16,"FFN projection: method");
        need(backend && role && role->buffer && ggml_is_contiguous(role) &&
             role->ne[0]==2560 && role->ne[1]==512 && role->ne[2]==1 && role->ne[3]==1 &&
             (role->type==GGML_TYPE_F32 || (method==PortableRouterMethod::single_bf16 && role->type==GGML_TYPE_BF16)) &&
             ggml_backend_buffer_get_type(role->buffer)==ggml_backend_get_default_buffer_type(backend),
             "FFN projection: invalid router role");
        const auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(role->buffer));
        const auto data=reinterpret_cast<uintptr_t>(role->data);
        const size_t size=ggml_backend_buffer_get_size(role->buffer), span=ggml_nbytes(role);
        need(base && data && data>=base && data-base<=size && span<=size-(data-base),
             "FFN projection: role outside buffer");
        if(role->view_src) {
            const auto source=reinterpret_cast<uintptr_t>(role->view_src->data);
            const size_t source_span=ggml_nbytes(role->view_src);
            need(role->view_src->buffer==role->buffer && source && source>=base &&
                 source-base<=size && source_span<=size-(source-base) &&
                 role->view_offs<=source_span && span<=source_span-role->view_offs &&
                 data-base==(source-base)+role->view_offs,
                 "FFN projection: inconsistent role view");
        }
        ctx_.reset(ggml_init({512u<<10,nullptr,true}));need(bool(ctx_),"FFN projection: metadata");
        weight_=method==PortableRouterMethod::legacy_f32 ? ggml_dup_tensor(ctx_.get(),role) :
            ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_BF16,2560,512);
        need(weight_,"FFN projection: weight metadata");
        // In the opt-in method, the T1 views reuse the T2 banks. Shared work
        // reads input_, while only operand_ contains rounded BF16 images.
        if(method==PortableRouterMethod::single_bf16){
            input_[1]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,2560,2);
            operand_[1]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,2560,2);
            input_[0]=ggml_view_2d(ctx_.get(),input_[1],2560,1,input_[1]->nb[1],0);
            operand_[0]=ggml_view_2d(ctx_.get(),operand_[1],2560,1,operand_[1]->nb[1],0);
        }
        for(size_t i=0;i<2;++i){
            if(method==PortableRouterMethod::legacy_f32){input_[i]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,2560,i+1);operand_[i]=input_[i];}
            output_[i]=ggml_mul_mat(ctx_.get(),weight_,operand_[i]);
            graph_[i]=ggml_new_graph(ctx_.get());
            need(input_[i]&&output_[i]&&graph_[i],"FFN projection: graph metadata");
            if(method==PortableRouterMethod::single_bf16)
                need(ggml_backend_supports_op(backend,output_[i]),"FFN projection: BF16 operation unsupported");
            ggml_set_input(input_[i]);ggml_set_output(output_[i]);ggml_build_forward_expand(graph_[i],output_[i]);
        }
        const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        need(bound<=budget,"FFN projection: budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));need(bool(buffer_),"FFN projection: allocation");
        bytes_=ggml_backend_buffer_get_size(buffer_.get());need(bytes_<=budget,"FFN projection: actual budget");
        if(method==PortableRouterMethod::legacy_f32) ggml_backend_tensor_copy(role,weight_);
        else {
            // Read and validate in bounded chunks, including actual BF16 roles.
            std::array<uint32_t,1024> raw{};std::array<uint16_t,1024> packed{};
            constexpr size_t count=2560u*512;
            for(size_t pos=0;pos<count;pos+=raw.size()){
                const size_t n=std::min(raw.size(),count-pos), stride=role->type==GGML_TYPE_F32?4:2;
                ggml_backend_tensor_get(role,raw.data(),pos*stride,n*stride);
                for(size_t j=0;j<n;++j){
                    uint16_t h;
                    if(stride==4){const uint32_t u=raw[j];need((u&0xffffu)==0,"FFN projection: inexact F32 weight");h=uint16_t(u>>16);}
                    else std::memcpy(&h,reinterpret_cast<const uint8_t*>(raw.data())+2*j,2);
                    need((h&0x7f80u)!=0x7f80u,"FFN projection: nonfinite weight");packed[j]=h;
                }
                ggml_backend_tensor_set(weight_,packed.data(),pos*2,n*2);
            }
        }
    }
    PortableNativeFfnProjection(const PortableNativeFfnProjection&)=delete;
    PortableNativeFfnProjection& operator=(const PortableNativeFfnProjection&)=delete;
    size_t bytes()const{return bytes_;}
    // Graph attempts, not measured device dispatches.
    size_t graph_attempts()const{return graph_attempts_;}
    PortableRouterMethod method()const{return method_;}
    // Fixed metadata plus bounded conversion/readback scratch; excludes backend-private workspace.
    static constexpr size_t host_bound(){return (512u<<10)+64u*1024;}
    ggml_tensor* output(size_t rows)const{need(rows==ready_ && ready_!=0,"FFN projection: stale output");return output_[rows-1];}
    ggml_tensor* mixed(size_t rows)const{need(rows==ready_ && ready_!=0,"FFN projection: stale input");return input_[rows-1];}
    void run(const std::vector<float>& mixed,size_t rows){
        ready_=0;need((rows==1||rows==2)&&mixed.size()==rows*2560,"FFN projection: row shape");
        for(float x:mixed)need(std::isfinite(x),"FFN projection: nonfinite input");
        ggml_backend_tensor_set(input_[rows-1],mixed.data(),0,mixed.size()*sizeof(float));
        if(method_==PortableRouterMethod::single_bf16){
            std::vector<float> rounded(mixed.size());
            for(size_t i=0;i<mixed.size();++i){
                uint32_t u;std::memcpy(&u,&mixed[i],4);u+=0x7fffu+((u>>16)&1u);u&=0xffff0000u;
                need((u&0x7f800000u)!=0x7f800000u,"FFN projection: BF16 overflow");
                std::memcpy(&rounded[i],&u,4);
            }
            ggml_backend_tensor_set(operand_[rows-1],rounded.data(),0,rounded.size()*sizeof(float));
        }
        ++graph_attempts_;
        const auto status=ggml_backend_graph_compute(backend_,graph_[rows-1]);ggml_backend_synchronize(backend_);
        need(status==GGML_STATUS_SUCCESS,"FFN projection: graph failure");
        std::vector<float> logits(rows*512);ggml_backend_tensor_get(output_[rows-1],logits.data(),0,logits.size()*sizeof(float));
        for(float x:logits)need(std::isfinite(x),"FFN projection: nonfinite logits");
        ready_=rows;
    }
};
}
