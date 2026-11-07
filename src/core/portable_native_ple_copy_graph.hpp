#pragma once
#include "portable_native_ple_grouped_norm.hpp"

namespace strata::core::portable_detail {
// Borrows backend and the three producer tensors (including their contexts and
// buffers); they must outlive this object and be quiescent during each phase. The
// graph expands producer dependencies, so they must be executable on backend.
// Budget covers grouped-norm device allocations and retained graph metadata;
// producer/dependency allocations are borrowed and excluded.
class PortableNativePleCopyGraph {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    static constexpr size_t metadata = 1024 * 1024;
    ggml_backend_t backend_;
    size_t rows_;
    PortableNativePleGroupedNorm norm_;
    Context context_{nullptr,ggml_free};
    std::array<ggml_cgraph*,2> graphs_{};
    std::array<ggml_tensor*,3> producers_{}, copies_{};
    bool ready_=false, first_ready_=false;
    static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
    void validate() const {
        for(size_t i=0;i<3;++i){
            auto* p=producers_[i]; auto* c=copies_[i];
            need(p&&p->type==GGML_TYPE_F32&&ggml_is_contiguous(p)&&p->buffer&&
                 ggml_nbytes(p)==40960*rows_,"PLE copy graph: producer span");
            auto type=ggml_backend_buffer_get_type(p->buffer);
            auto dev=ggml_backend_buft_get_device(type), owner=ggml_backend_get_device(backend_);
            need(type==ggml_backend_get_default_buffer_type(backend_)&&owner&&
                 (dev?dev==owner:(ggml_backend_buft_is_host(type)&&ggml_backend_dev_type(owner)==GGML_BACKEND_DEVICE_TYPE_CPU)),"PLE copy graph: producer device");
            need(c&&c->op==GGML_OP_CPY&&c->src[0]==p&&c->src[1]==norm_.input(i)&&c->view_src==norm_.input(i),"PLE copy graph: binding changed");
        }
    }
public:
    // Graph context is already included in estimate(); norm host payload is not.
    // Producer tensors and dependencies are borrowed.
    static constexpr size_t host_bound(size_t rows){return ple_host_add(metadata,PortableNativePleGroupedNorm::host_bound(rows));}
    static size_t estimate(ggml_backend_t backend,size_t rows=1){
        auto n=PortableNativePleGroupedNorm::estimate(backend,rows);
        need(n<=SIZE_MAX-metadata,"PLE copy graph: budget overflow");
        return n+metadata;
    }
    PortableNativePleCopyGraph(ggml_backend_t backend,const std::array<ggml_tensor*,3>& producers,
                               const std::array<const float*,3>& gamma,const char* module,size_t budget,size_t rows=1)
        :backend_(backend),rows_(rows),norm_(backend,gamma,module,budget>=metadata?budget-metadata:0,rows),producers_(producers){
        need(budget>=estimate(backend,rows),"PLE copy graph: budget");
        need(producers[0]&&producers[1]&&producers[2]&&producers[0]!=producers[1]&&producers[0]!=producers[2]&&producers[1]!=producers[2],"PLE copy graph: distinct producers required");
        for(auto* p:producers) need(p->type==GGML_TYPE_F32&&ggml_is_contiguous(p)&&p->buffer&&ggml_nbytes(p)==40960*rows_,"PLE copy graph: producer span");
        context_.reset(ggml_init({metadata,nullptr,true}));need(bool(context_),"PLE copy graph: metadata");
        for(auto& graph:graphs_){graph=ggml_new_graph(context_.get());need(graph,"PLE copy graph: graph");}
        for(size_t i=0;i<3;++i){
            copies_[i]=ggml_cpy(context_.get(),producers_[i],norm_.input(i));
            need(copies_[i],"PLE copy graph: copy");
            norm_.bind(i,copies_[i],producers_[i]);
            ggml_build_forward_expand(graphs_[i==2?1:0],copies_[i]);
        }
        validate();
        // A key/query dependency on the gated-value producer would read it early.
        for(int n=0;n<ggml_graph_n_nodes(graphs_[0]);++n){
            auto* node=ggml_graph_node(graphs_[0],n);
            need(node!=producers_[2]&&node->view_src!=producers_[2],"PLE copy graph: early gated-value dependency");
            for(auto* src:node->src) need(src!=producers_[2],"PLE copy graph: early gated-value dependency");
        }
    }
    PortableNativePleCopyGraph(const PortableNativePleCopyGraph&)=delete;
    PortableNativePleCopyGraph& operator=(const PortableNativePleCopyGraph&)=delete;
    size_t bytes()const{return norm_.bytes()+metadata;}
    uint64_t calls()const{return norm_.calls();}
    bool ready()const{return ready_;}
    // Exposed for identity inspection; callers must not mutate the graph.
    const ggml_tensor* copy(size_t i)const{return copies_.at(i);}
    // Stable identity for binding downstream owners before the first run.
    // Contents are only valid after the corresponding phase succeeds.
    ggml_tensor* output_bank(size_t i)const{return norm_.output(i);}
    ggml_tensor* output(size_t i)const{need(i<3&&norm_.ready(i),"PLE copy graph: output not ready");return norm_.output(i);}
    void reset(){ready_=false;first_ready_=false;norm_.begin();}
    // Complete key/query before the caller runs the gate and produces role 2.
    void run_key_query(const std::array<ggml_tensor*,3>& producers){
        reset();
        try {
            need(producers==producers_,"PLE copy graph: producer identity changed");
            validate();
            const auto status=ggml_backend_graph_compute(backend_,graphs_[0]);
            ggml_backend_synchronize(backend_);
            need(status==GGML_STATUS_SUCCESS,"PLE copy graph: execution failed");
            for(size_t i=0;i<2;++i)norm_.run(i,copies_[i],producers_[i]);
            first_ready_=true;
        } catch(...) {reset();throw;}
    }
    // Call only after the gate and gated-value producer have completed.
    void run_gated_value(const std::array<ggml_tensor*,3>& producers){
        try {
            need(first_ready_&&!ready_,"PLE copy graph: key/query phase required");
            need(producers==producers_,"PLE copy graph: producer identity changed");
            validate();
            const auto status=ggml_backend_graph_compute(backend_,graphs_[1]);
            ggml_backend_synchronize(backend_);
            need(status==GGML_STATUS_SUCCESS,"PLE copy graph: execution failed");
            norm_.run(2,copies_[2],producers_[2]);
            ready_=true;
        } catch(...) {reset();throw;}
    }
};
}
