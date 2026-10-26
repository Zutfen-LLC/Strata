#pragma once
// Fixed one-expert graph shared by the opt-in adapter and offline diagnostic.
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::worker::vulkan_detail {
namespace cpu = strata::kernels::cpu;
inline constexpr int H = 2560, F = 640;
inline void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
inline void finite(const std::vector<float>& v) {
    for (float x : v) require(std::isfinite(x), "nonfinite expert value");
}
struct Rows { std::vector<float> gate, up, hidden, out; };
inline float host_swiglu(float gate, float up) { return (gate/(1.f+std::exp(-gate)))*up; }
struct Boundaries {
    std::vector<uint8_t> input_q, hidden_q;
    std::vector<float> input_f, down_f;
};
inline void validate(const Rows& r) {
    require(r.gate.size() == F && r.up.size() == F && r.hidden.size() == F && r.out.size() == H,
            "expert result shape mismatch");
    finite(r.gate); finite(r.up); finite(r.hidden); finite(r.out);
}
class ExpertGraph {
    Context ctx_{nullptr,ggml_free};
    Buffer buf_{nullptr,ggml_backend_buffer_free};
    ggml_cgraph *graph_ = nullptr, *down_graph_ = nullptr;
    ggml_tensor *x_ = nullptr, *g_ = nullptr, *u_ = nullptr, *hidden_ = nullptr, *out_ = nullptr;
    ggml_backend_t backend_;
    cpu::NativeFmt fmt_;
    bool native_boundaries_;
    Boundaries boundaries_;
    std::vector<float> read(ggml_tensor* t) {
        std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t,v.data(),0,v.size()*sizeof(float)); return v;
    }
public:
    ExpertGraph(ggml_backend_t backend, const cpu::NativeFmt& fmt, const std::vector<uint8_t>& blob,
                bool native_boundaries = false, size_t tensor_budget = 16u<<20):backend_(backend),fmt_(fmt),native_boundaries_(native_boundaries) {
        require(blob.size() == fmt.bytes && fmt.bytes == 5222400 && fmt.up_off == 1740800 &&
                fmt.down_off == 3481600 && fmt.gu_act == 8 && fmt.d_act == 8,
                "unsupported fixed Q8 expert layout");
        ggml_init_params p{2u<<20,nullptr,true};
        ctx_.reset(ggml_init(p)); require(bool(ctx_), "ggml metadata allocation failed");
        auto* gate = ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_Q8_0,H,F);
        auto* up = ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_Q8_0,H,F);
        auto* down = ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_Q8_0,F,H);
        x_ = ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,H,1);
        g_ = ggml_mul_mat(ctx_.get(),gate,x_); u_ = ggml_mul_mat(ctx_.get(),up,x_);
        hidden_ = native_boundaries_ ? ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,F,1) :
                  ggml_mul(ctx_.get(),ggml_silu(ctx_.get(),g_),u_);
        out_ = ggml_mul_mat(ctx_.get(),down,hidden_);
        graph_ = ggml_new_graph(ctx_.get());
        if (native_boundaries_) {
            ggml_build_forward_expand(graph_,g_); ggml_build_forward_expand(graph_,u_);
            down_graph_ = ggml_new_graph(ctx_.get()); ggml_build_forward_expand(down_graph_,out_);
        } else ggml_build_forward_expand(graph_,out_);
        // Direct backend execution only. Unsupported operations stop; no scheduler/CPU fallback.
        for (auto* graph : {graph_,down_graph_}) if (graph)
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
                require(ggml_backend_supports_op(backend_,ggml_graph_node(graph,i)), "backend declines expert operation");
        // Pinned ggml dry-run API accounts for backend tensor padding/alignment before allocation.
        const size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(
            ctx_.get(),ggml_backend_get_default_buffer_type(backend_));
        require(needed > 0 && needed < (16u<<20),"unexpected backend tensor size above 16 MiB");
        require(needed <= tensor_budget,"Vulkan resident tensor budget exceeded");
        buf_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend_));
        require(bool(buf_), "backend expert allocation failed");
        require(ggml_backend_buffer_get_size(buf_.get()) == needed,
                "backend allocation differs from budgeted tensor size");
        ggml_backend_tensor_set(gate,blob.data(),0,fmt.up_off);
        ggml_backend_tensor_set(up,blob.data()+fmt.up_off,0,fmt.up_off);
        ggml_backend_tensor_set(down,blob.data()+fmt.down_off,0,fmt.bytes-fmt.down_off);
    }
    size_t buffer_bytes() const { return ggml_backend_buffer_get_size(buf_.get()); }
    const Boundaries& boundaries() const { return boundaries_; }
    Rows run(const std::vector<float>& x) {
        require(x.size() == H,"expert input shape mismatch"); finite(x);
        if (native_boundaries_) {
            boundaries_.input_q.resize(fmt_.act_bytes); boundaries_.input_f.resize(H);
            cpu::native_quant_act(fmt_,x.data(),boundaries_.input_q.data());
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(boundaries_.input_q.data(),boundaries_.input_f.data(),H);
            finite(boundaries_.input_f);
            ggml_backend_tensor_set(x_,boundaries_.input_f.data(),0,H*sizeof(float));
            require(ggml_backend_graph_compute(backend_,graph_) == GGML_STATUS_SUCCESS,"hybrid gate/up compute failed");
            ggml_backend_synchronize(backend_);
            Rows r{read(g_),read(u_),std::vector<float>(F),{}};
            finite(r.gate); finite(r.up);
            for (int i = 0; i < F; ++i) r.hidden[i] = host_swiglu(r.gate[i],r.up[i]);
            finite(r.hidden);
            boundaries_.hidden_q.resize(fmt_.h_bytes); boundaries_.down_f.resize(F);
            cpu::native_quant_h(fmt_,r.hidden.data(),boundaries_.hidden_q.data());
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(boundaries_.hidden_q.data(),boundaries_.down_f.data(),F);
            finite(boundaries_.down_f);
            ggml_backend_tensor_set(hidden_,boundaries_.down_f.data(),0,F*sizeof(float));
            require(ggml_backend_graph_compute(backend_,down_graph_) == GGML_STATUS_SUCCESS,"hybrid down compute failed");
            ggml_backend_synchronize(backend_);
            r.out = read(out_); validate(r); return r;
        }
        ggml_backend_tensor_set(x_,x.data(),0,x.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend_,graph_) == GGML_STATUS_SUCCESS,"backend expert compute failed");
        ggml_backend_synchronize(backend_);
        Rows r{read(g_),read(u_),read(hidden_),read(out_)}; validate(r); return r;
    }
};
} // namespace strata::worker::vulkan_detail
