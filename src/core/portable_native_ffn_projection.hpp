#pragma once
// Standalone fixed T1/T2 router matrix projection. No model admission.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core::portable_detail {
class PortableNativeFfnProjection {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    ggml_backend_t backend_;
    Context ctx_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    ggml_tensor *weight_=nullptr,*input_[2]{},*output_[2]{};
    ggml_cgraph* graph_[2]{};
    size_t bytes_=0,ready_=0;
    static void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
public:
    // The source role is copied once; backend must outlive this owner. This
    // budget covers the backend tensor buffer only, not metadata or workspace.
    PortableNativeFfnProjection(ggml_backend_t backend,const ggml_tensor* role,size_t budget):backend_(backend){
        need(backend && role && role->buffer && ggml_is_contiguous(role) &&
             role->ne[0]==2560 && role->ne[1]==512 && role->ne[2]==1 && role->ne[3]==1 &&
             role->type==GGML_TYPE_F32 &&
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
        weight_=ggml_dup_tensor(ctx_.get(),role);need(weight_,"FFN projection: weight metadata");
        for(size_t i=0;i<2;++i){
            input_[i]=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,2560,i+1);
            output_[i]=ggml_mul_mat(ctx_.get(),weight_,input_[i]);
            graph_[i]=ggml_new_graph(ctx_.get());
            need(input_[i]&&output_[i]&&graph_[i],"FFN projection: graph metadata");
            ggml_set_input(input_[i]);ggml_set_output(output_[i]);ggml_build_forward_expand(graph_[i],output_[i]);
        }
        const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        need(bound<=budget,"FFN projection: budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));need(bool(buffer_),"FFN projection: allocation");
        bytes_=ggml_backend_buffer_get_size(buffer_.get());need(bytes_<=budget,"FFN projection: actual budget");
        ggml_backend_tensor_copy(role,weight_);
    }
    PortableNativeFfnProjection(const PortableNativeFfnProjection&)=delete;
    PortableNativeFfnProjection& operator=(const PortableNativeFfnProjection&)=delete;
    size_t bytes()const{return bytes_;}
    ggml_tensor* output(size_t rows)const{need(rows==ready_ && ready_!=0,"FFN projection: stale output");return output_[rows-1];}
    ggml_tensor* mixed(size_t rows)const{need(rows==ready_ && ready_!=0,"FFN projection: stale input");return input_[rows-1];}
    void run(const std::vector<float>& mixed,size_t rows){
        ready_=0;need((rows==1||rows==2)&&mixed.size()==rows*2560,"FFN projection: row shape");
        for(float x:mixed)need(std::isfinite(x),"FFN projection: nonfinite input");
        ggml_backend_tensor_set(input_[rows-1],mixed.data(),0,mixed.size()*sizeof(float));
        const auto status=ggml_backend_graph_compute(backend_,graph_[rows-1]);ggml_backend_synchronize(backend_);
        need(status==GGML_STATUS_SUCCESS,"FFN projection: graph failure");
        std::vector<float> logits(rows*512);ggml_backend_tensor_get(output_[rows-1],logits.data(),0,logits.size()*sizeof(float));
        for(float x:logits)need(std::isfinite(x),"FFN projection: nonfinite logits");
        ready_=rows;
    }
};
}
