#pragma once
#ifndef STRATA_PLE_CONV_COPY_TEST_COMPUTE
#define STRATA_PLE_CONV_COPY_TEST_COMPUTE ggml_backend_graph_compute
#endif
#include "portable_native_ple_conv.hpp"

namespace strata::core::portable_detail {
// Borrows the backend and producer tensors, including their contexts and buffers.
// They and the executable producer dependencies must outlive this connector and
// remain quiescent during run. Weights are packed by the owned convolution.
class PortableNativePleConvCopyGraph {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    ggml_backend_t backend_;
    size_t rows_;
    PortableNativePleConv conv_;
    Context context_{nullptr,ggml_free};
    ggml_cgraph* graph_=nullptr;
    std::array<ggml_tensor*,4> producers_{},copies_{};
    bool ready_=false;
    static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
    void validate() const {
        for(size_t i=0;i<4;++i){
            auto* p=producers_[i]; auto* c=copies_[i];
            need(p&&p->type==GGML_TYPE_F32&&ggml_is_contiguous(p)&&p->buffer&&p->data&&ggml_nbytes(p)==(i?10240*rows_:92160)*sizeof(float),"PLE conv copy graph: producer span");
            auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(p->buffer));
            auto data=reinterpret_cast<uintptr_t>(p->data);
            auto size=ggml_backend_buffer_get_size(p->buffer), span=ggml_nbytes(p);
            need(base&&data>=base&&data-base<=size&&span<=size-(data-base),"PLE conv copy graph: out of buffer");
            auto type=ggml_backend_buffer_get_type(p->buffer); auto dev=ggml_backend_buft_get_device(type); auto owner=ggml_backend_get_device(backend_);
            need(type==ggml_backend_get_default_buffer_type(backend_)&&owner&&(dev?dev==owner:(ggml_backend_buft_is_host(type)&&ggml_backend_dev_type(owner)==GGML_BACKEND_DEVICE_TYPE_CPU)),"PLE conv copy graph: producer device");
            need(c&&c->op==GGML_OP_CPY&&c->src[0]==p&&c->src[1]==conv_.input(i)&&c->view_src==conv_.input(i),"PLE conv copy graph: binding changed");
        }
    }
    // Upper bound on simultaneously live diagnostic vectors in conv::run,
    // including retained diagnostics from a previous successful invocation.
    static constexpr size_t host_bytes(size_t rows){return ple_host_add(1024*1024, ple_host_add(40960*sizeof(uint16_t), ple_host_mul(ple_host_add(5*92160, ple_host_mul(20*10240, rows)), sizeof(float))));}
public:
    // The mixed estimate includes this host contribution. Expose it so a
    // composing admission calculation can remove it before adding host_bound.
    static constexpr size_t estimate_host_bound(size_t rows) { return host_bytes(rows); }
    // This also includes the convolution context absent from the mixed estimate.
    static constexpr size_t host_bound(size_t rows) { return ple_host_add(1024*1024, PortableNativePleConv::host_bound(rows)); }
    static size_t estimate(ggml_backend_t backend,size_t rows=1){
        auto device=PortableNativePleConv::estimate(backend,rows); auto host=host_bytes(rows);
        need(device<=SIZE_MAX-host,"PLE conv copy graph: budget overflow"); return device+host;
    }
    PortableNativePleConvCopyGraph(ggml_backend_t backend,const std::array<ggml_tensor*,4>& producers,
        const void* weights,size_t weight_bytes,uint32_t role_type,const char* module,size_t budget,size_t rows=1)
        :backend_(backend),rows_(rows),conv_(backend,weights,weight_bytes,role_type,module,
          budget>=estimate(backend,rows)?budget-host_bytes(rows):0,rows),producers_(producers){
        need(budget>=estimate(backend,rows),"PLE conv copy graph: budget");
        for(size_t i=0;i<4;++i){need(producers[i],"PLE conv copy graph: null producer");for(size_t j=0;j<i;++j)need(producers[i]!=producers[j],"PLE conv copy graph: distinct producers required");}
        context_.reset(ggml_init({1024*1024,nullptr,true}));need(bool(context_),"PLE conv copy graph: metadata");
        graph_=ggml_new_graph(context_.get());need(graph_,"PLE conv copy graph: graph");
        for(size_t i=0;i<4;++i){copies_[i]=ggml_cpy(context_.get(),producers_[i],conv_.input(i));need(copies_[i],"PLE conv copy graph: copy");ggml_build_forward_expand(graph_,copies_[i]);}
        conv_.bind(copies_.data(),producers_.data());validate();
    }
    PortableNativePleConvCopyGraph(const PortableNativePleConvCopyGraph&)=delete;
    PortableNativePleConvCopyGraph& operator=(const PortableNativePleConvCopyGraph&)=delete;
    size_t bytes()const{return conv_.bytes()+host_bytes(rows_);}
    uint64_t calls()const{return conv_.calls();}
    bool ready()const{return ready_;}
    const ggml_tensor* copy(size_t i)const{return copies_.at(i);}
    ggml_tensor* result()const{need(ready_,"PLE conv copy graph: not ready");return conv_.result();}
    ggml_tensor* history()const{need(ready_,"PLE conv copy graph: not ready");return conv_.history();}
    void abort(){ready_=false;}
    void run(const std::array<ggml_tensor*,4>& producers,const std::vector<float>& residual){
        ready_=false;need(producers==producers_,"PLE conv copy graph: producer identity changed");validate();
        auto status=STRATA_PLE_CONV_COPY_TEST_COMPUTE(backend_,graph_);ggml_backend_synchronize(backend_);
        need(status==GGML_STATUS_SUCCESS,"PLE conv copy graph: execution failed");
        conv_.run(copies_.data(),producers_.data(),residual);ready_=true;
    }
};
}
