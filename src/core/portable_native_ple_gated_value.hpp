#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace strata::core::portable_detail {
// The borrowed producer tensors, their buffers and backend must outlive this owner.
// The caller must complete and synchronize the native gate before calling run().
// No producer graph is expanded: the graph reads the already-produced buffers.
class PortableNativePleGatedValue {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    static constexpr size_t metadata = 1024 * 1024;
    ggml_backend_t backend_;
    size_t rows_, bytes_ = 0;
    ggml_tensor *value_, *gate_, *vleaf_ = nullptr, *gleaf_ = nullptr, *output_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    Context context_{nullptr, ggml_free};
    Buffer buffer_{nullptr, ggml_backend_buffer_free};
    bool ready_ = false;
    static void need(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
    static Context context() {
        Context c(ggml_init({metadata, nullptr, true}), ggml_free);
        need(bool(c), "PLE gated value: metadata allocation"); return c;
    }
    static void build(ggml_context* c, size_t rows, ggml_tensor*& v, ggml_tensor*& g,
                      ggml_tensor*& out, ggml_cgraph*& graph) {
        v = ggml_new_tensor_3d(c, GGML_TYPE_F32, 2560, 1, rows);
        g = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 4, rows);
        auto* shaped = ggml_reshape_3d(c, v, 2560, 1, rows);
        auto* repeated = ggml_repeat_4d(c, shaped, 2560, 4, rows, 1);
        out = ggml_mul(c, repeated, g);
        graph = ggml_new_graph(c);
        need(v && g && shaped && repeated && out && graph, "PLE gated value: graph allocation");
        ggml_build_forward_expand(graph, out);
    }
    void span(const ggml_tensor* t, int64_t n0, int64_t n1) const {
        need(t && t->type == GGML_TYPE_F32 && t->ne[0] == n0 && t->ne[1] == n1 &&
             t->ne[2] == int64_t(rows_) && t->ne[3] == 1 && ggml_is_contiguous(t) &&
             t->buffer && t->data, "PLE gated value: producer shape or span");
        auto type = ggml_backend_buffer_get_type(t->buffer);
        auto dev = ggml_backend_buft_get_device(type), owner = ggml_backend_get_device(backend_);
        need(owner && type == ggml_backend_get_default_buffer_type(backend_) &&
             (dev ? dev == owner : (ggml_backend_buft_is_host(type) && ggml_backend_dev_type(owner) == GGML_BACKEND_DEVICE_TYPE_CPU)),
             "PLE gated value: producer device");
        auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(t->buffer));
        auto p = reinterpret_cast<uintptr_t>(t->data);
        auto size = ggml_backend_buffer_get_size(t->buffer), n = ggml_nbytes(t);
        need(base && p >= base && p - base <= size && n <= size - (p - base), "PLE gated value: out of buffer");
    }
    void validate() const {
        span(value_, 2560, 1); span(gate_, 1, 4);
        auto a = reinterpret_cast<uintptr_t>(value_->data), b = reinterpret_cast<uintptr_t>(gate_->data);
        need(a <= b ? ggml_nbytes(value_) <= b-a : ggml_nbytes(gate_) <= a-b,
             "PLE gated value: overlapping producers");
    }
public:
    // Graph context only, already included in estimate(); producer banks borrowed.
    static constexpr size_t host_bound(size_t) { return metadata; }
    static size_t estimate(ggml_backend_t backend, size_t rows = 1) {
        need(backend && (rows == 1 || rows == 2), "PLE gated value: backend or rows");
        auto c = context(); ggml_tensor *v, *g, *out; ggml_cgraph* graph;
        build(c.get(), rows, v, g, out, graph);
        auto n = ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(), ggml_backend_get_default_buffer_type(backend));
        need(n <= SIZE_MAX - metadata, "PLE gated value: budget overflow"); return n + metadata;
    }
    PortableNativePleGatedValue(ggml_backend_t backend, ggml_tensor* value, ggml_tensor* gate,
                                size_t budget, size_t rows = 1)
        : backend_(backend), rows_(rows), value_(value), gate_(gate) {
        auto admitted = estimate(backend, rows); need(admitted <= budget, "PLE gated value: budget");
        validate(); context_ = context(); build(context_.get(), rows, vleaf_, gleaf_, output_, graph_);
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend));
        need(bool(buffer_), "PLE gated value: allocation");
        auto actual = ggml_backend_buffer_get_size(buffer_.get());
        need(actual <= admitted - metadata, "PLE gated value: actual budget"); bytes_ = actual + metadata;
    }
    PortableNativePleGatedValue(const PortableNativePleGatedValue&) = delete;
    PortableNativePleGatedValue& operator=(const PortableNativePleGatedValue&) = delete;
    size_t bytes() const { return bytes_; }
    bool ready() const { return ready_; }
    void reset() { ready_ = false; }
    // Stable identity for constructing retained downstream graphs. Its contents
    // are not valid until run() succeeds; output() enforces that at consumption.
    ggml_tensor* output_bank() const { return output_; }
    ggml_tensor* output() const { need(ready_, "PLE gated value: output not ready"); return output_; }
    void run(ggml_tensor* value, ggml_tensor* gate) {
        reset();
        need(value == value_ && gate == gate_, "PLE gated value: producer identity"); validate();
        // Bind graph leaves directly to the already-computed producers. Their op and
        // dependencies remain absent from this graph, so the gate cannot be rerun.
        vleaf_->buffer = value_->buffer; vleaf_->data = value_->data;
        gleaf_->buffer = gate_->buffer; gleaf_->data = gate_->data;
        auto* shaped = output_->src[0]->src[0];
        shaped->buffer = value_->buffer; shaped->data = value_->data;
#ifdef STRATA_PLE_GATED_VALUE_TEST_COMPUTE
        auto status = STRATA_PLE_GATED_VALUE_TEST_COMPUTE(backend_, graph_);
#else
        auto status = ggml_backend_graph_compute(backend_, graph_);
#endif
        ggml_backend_synchronize(backend_);
        need(status == GGML_STATUS_SUCCESS, "PLE gated value: execution failed"); ready_ = true;
    }
};
}
