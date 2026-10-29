#include "strata/core/portable_qsa.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "portable_graph.hpp"
#include "portable_qsa_graph.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace strata::core {
namespace {
using portable_detail::require;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr int64_t H = 2560, HC = 4, R = 320, D = 256, QH = 24, KH = 2, ID = 128, IH = 4;
constexpr size_t RES = H * HC, KV = D * KH;
const std::map<std::string, std::vector<int64_t>> &shapes() {
    static const std::map<std::string, std::vector<int64_t>> s{{"hc_attn_norm.weight", {RES}},
                                                               {"hc_attn_down.weight", {RES, R}},
                                                               {"hc_attn_up.weight", {R, RES}},
                                                               {"hc_attn_inject.weight", {RES, HC}},
                                                               {"attn_q.weight", {H, 2 * D * QH}},
                                                               {"attn_k.weight", {H, KV}},
                                                               {"attn_v.weight", {H, KV}},
                                                               {"attn_output.weight", {D * QH, H}},
                                                               {"attn_q_norm.weight", {D}},
                                                               {"attn_k_norm.weight", {D}},
                                                               {"indexer.q_proj.weight", {H, ID * IH}},
                                                               {"indexer.k_proj.weight", {H, ID}},
                                                               {"indexer.q_norm.weight", {ID}},
                                                               {"indexer.k_norm.weight", {ID}}};
    return s;
}
Context context() {
    Context c(ggml_init({4u << 20, nullptr, true}), ggml_free);
    require(bool(c), "portable QSA: metadata allocation failed");
    return c;
}
void finite(const std::vector<float> &x) {
    for (auto v : x)
        if (!std::isfinite(v))
            require(false, "portable QSA: nonfinite input/result");
}
std::vector<float> read(ggml_tensor *t) {
    std::vector<float> x(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, x.data(), 0, x.size() * sizeof(float));
    finite(x);
    return x;
}
void write(ggml_tensor *t, const std::vector<float> &x) {
    require(size_t(ggml_nelements(t)) == x.size(), "portable QSA: input size mismatch");
    ggml_backend_tensor_set(t, x.data(), 0, x.size() * sizeof(float));
}
std::vector<uint16_t> half(const std::vector<float> &x) {
    std::vector<uint16_t> h(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        h[i] = ggml_fp32_to_fp16(x[i]);
        if (!std::isfinite(ggml_fp16_to_fp32(h[i])))
            require(false, "portable QSA: FP16 cache overflow");
    }
    return h;
}
// One schema for preflight and allocation, in allocator order. No layer,
// scheduler, weight or backend buffer is created by the estimator.
std::array<ggml_tensor *, 7> state_tensors(ggml_context *c, size_t capacity) {
    require(capacity > 0 && capacity <= 262144, "portable QSA: capacity must be in [1,262144]");
    return {{ggml_new_tensor_3d(c, GGML_TYPE_F16, D, KH, capacity),
             ggml_new_tensor_3d(c, GGML_TYPE_F16, D, KH, capacity),
             ggml_new_tensor_2d(c, GGML_TYPE_F32, ID, capacity / 4 + 1),
             ggml_new_tensor_2d(c, GGML_TYPE_F32, ID, 1),
             ggml_new_tensor_3d(c, GGML_TYPE_F32, ID, 4, 1),
             ggml_new_tensor_1d(c, GGML_TYPE_I32, capacity),
             ggml_new_tensor_1d(c, GGML_TYPE_F32, capacity)}};
}
size_t state_allocation_size(ggml_context *c, ggml_backend_t backend) {
    require(backend != nullptr, "portable QSA: state estimate requires selected backend");
    auto *buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    require(alignment > 0 && (alignment & (alignment - 1)) == 0,
            "portable QSA: invalid device state alignment");
    const size_t maximum = ggml_backend_buft_get_max_size(buft);
    size_t total = 0;
    for (auto *t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, t);
        require(bytes <= SIZE_MAX - (alignment - 1), "portable QSA: device state alignment overflow");
        const size_t padded = (bytes + alignment - 1) & ~(alignment - 1);
        require(padded <= maximum && padded <= SIZE_MAX - total,
                "portable QSA: device state allocation overflow/limit");
        total += padded;
    }
    // Pinned GGML pads each tensor and sums buffer chunks. Check all arithmetic
    // before calling its estimator, which otherwise uses unchecked additions.
    require(total == ggml_backend_alloc_ctx_tensors_from_buft_size(c, buft),
            "portable QSA: device state estimator mismatch");
    return total;
}
ggml_tensor *rope(ggml_context *c, ggml_tensor *x, ggml_tensor *position) {
    int sections[]{11, 11, 10, 0};
    return ggml_rope_multi(c, x, position, nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE, 0, 1e7f, 1.0f, 0.0f, 1.0f,
                           32.0f, 1.0f);
}
struct Phase {
    Context ctx{context()};
    ggml_cgraph *graph = ggml_new_graph(ctx.get());
    portable_detail::Scheduler scheduler{nullptr, ggml_backend_sched_free};
    size_t workspace_highwater = 0;
    ggml_tensor *input(std::initializer_list<int64_t> dims, ggml_type type = GGML_TYPE_F32) {
        auto *t = ggml_new_tensor(ctx.get(), type, int(dims.size()), dims.begin());
        ggml_set_input(t);
        return t;
    }
    void output(ggml_tensor *t, const char *name) {
        ggml_set_name(t, name);
        ggml_set_output(t);
        ggml_build_forward_expand(graph, t);
    }
    void schedule(const char *name, const PortableFfnOptions &opts, portable_detail::Backend &cpu,
                  std::vector<PortablePlacement> &placement, size_t &bytes, bool reuse = false) {
        // Public CPU plan exposes backend work_data (TOP_K per-thread index
        // scratch, quantized dot staging). Price it even for a strict device run,
        // so an admitted fallback cannot acquire unpriced host workspace.
        const size_t planned = ggml_graph_plan(graph, opts.threads, nullptr).work_size;
        const size_t workspace = reuse ? std::max(workspace_highwater, planned) : planned;
        require(bytes <= opts.graph_budget && workspace <= opts.graph_budget - bytes,
                "portable QSA: CPU workspace budget exceeded");
        bytes += workspace;
        workspace_highwater = workspace;
        if (reuse)
            portable_detail::schedule_reusing(graph, name, opts, cpu, placement, bytes, scheduler);
        else
            scheduler = portable_detail::schedule(graph, name, opts, cpu, placement, bytes);
    }
    void rebuild() {
        // Execution is synchronous. Forget scheduler tensor identities before
        // reusing their addresses in the same bounded metadata arena.
        if (scheduler)
            ggml_backend_sched_reset(scheduler.get());
        ggml_reset(ctx.get());
        graph = ggml_new_graph(ctx.get());
    }
    void run() {
        require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS,
                "portable QSA: graph execution failed");
    }
};
} // namespace

struct PortableQsaLayer::Impl {
    PortableQsaOptions options;
    portable_detail::Backend cpu{nullptr, ggml_backend_free};
    Context weights_ctx{nullptr, ggml_free};
    Buffer weights_buffer{nullptr, ggml_backend_buffer_free};
    std::map<std::string, ggml_tensor *> w;
    Context state_ctx{nullptr, ggml_free};
    Context copy_ctx{nullptr, ggml_free};
    Buffer state_buffer{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<PortableFfn> ffn, chunk_ffn;
    bool chunk_active = false;
    size_t chunk_host_bound = 0;
    PortableLayerChunkWork chunk_work;
    size_t chunk_graphs() const { return chunk_ffn ? chunk_ffn->graph_bytes() : 0; }

    void prepare(size_t T,std::string& error) {
        auto& s=*this;
        require(T>=1&&T<=8,"portable layer: chunk size outside [1,8]");
        const size_t extra=(s.options.production?0:s.options.capacity*2*KV*sizeof(uint16_t)+
                            (s.options.capacity/4+6)*ID*sizeof(float))+(3*T*RES+RES)*sizeof(float);
        const size_t device_state=s.state_buffer?ggml_backend_buffer_get_size(s.state_buffer.get()):0;
        require(s.states_size+device_state<=s.options.state_budget &&
                extra<=s.options.state_budget-s.states_size-device_state,"portable QSA: chunk working state budget exceeded");
        s.chunk_host_bound=extra;
        if(T==1 || (s.chunk_ffn && s.chunk_ffn->tokens()!=int64_t(T))) s.chunk_ffn.reset();
        if(T>1 && !s.chunk_ffn) {
            const size_t used=s.fixed_graphs+s.retained_attention_bytes;
            require(used<s.options.execution.graph_budget,"portable QSA: no chunk graph budget");
            s.chunk_ffn=s.ffn->create_chunk_execution(int64_t(T),error,s.options.execution.graph_budget-used);
            require(bool(s.chunk_ffn),error);
        }
    }
    struct CursorView {
        const std::vector<uint16_t>& keys;
        const std::vector<uint16_t>& values;
        const std::vector<float>& pooled;
        const std::vector<float>& tail;
    };
    struct AttentionResult {
        std::vector<float> residual;
        std::vector<uint16_t> key,value;
        std::vector<float> pooled,tail;
        std::vector<PortablePlacement> placement;
        size_t graphs;
        PortableQsaTrace trace;
        std::unique_ptr<Phase> temporary; // legacy T1 scratch stays live through FFN
    };
    AttentionResult attend(uint64_t position,const std::vector<float>& residual,CursorView cursor,bool trace) {
        auto& s = *this;
        finite(residual);
        const int32_t p = int32_t(position);
        std::array<int32_t, 4> positions{{p, p, p, p}};
        ggml_backend_tensor_set(s.pos, positions.data(), 0, sizeof(positions));
        write(s.input, residual);
        s.project->run();
        auto q = read(s.q), k = read(s.k), v = read(s.v), gate = read(s.gate), iq = read(s.iq), ik = read(s.ik),
             inject = read(s.inject);
        auto hk = half(k), hv = half(v);
        const size_t n = size_t(position) + 1, complete = n / 4;
        std::vector<float> next_tail, new_pool, next_pool;
        std::vector<uint16_t> next_keys, next_values;
        if (s.options.production) {
            // Preserve the first-token four-add/scale pool calculation exactly.
            if (position == 0) {
                std::vector<float> first(4 * ID);
                for (size_t row = 0; row < 4; ++row)
                    std::copy(ik.begin(), ik.end(), first.begin() + row * ID);
                write(s.device_tail, first);
                ggml_backend_tensor_set(s.pool_pos, positions.data(), 0, sizeof(positions));
                s.pool->run();
                ggml_backend_tensor_copy(s.pool_output, s.device_spare);
                ggml_backend_tensor_memset(s.device_tail, 0, 0, ggml_nbytes(s.device_tail));
            }
            ggml_backend_tensor_set(s.device_k, hk.data(), size_t(position) * KV * sizeof(uint16_t), KV * sizeof(uint16_t));
            ggml_backend_tensor_set(s.device_v, hv.data(), size_t(position) * KV * sizeof(uint16_t), KV * sizeof(uint16_t));
            ggml_backend_tensor_set(s.device_tail, ik.data(), size_t(position % 4) * ID * sizeof(float), ID * sizeof(float));
            // Completion overwrites the old spare; the immutable spare is copied
            // into its new slot. Only this block's bias changes.
            ggml_reset(s.copy_ctx.get());
            if (n % 4 == 0) {
                const int32_t start = int32_t(n - 4);
                positions = {{start, start, start, start}};
                ggml_backend_tensor_set(s.pool_pos, positions.data(), 0, sizeof(positions));
                s.pool->run();
                auto *row = Impl::view(s.copy_ctx.get(), s.device_pool, 1, (complete - 1) * ID * sizeof(float));
                ggml_backend_tensor_copy(s.pool_output, row);
                ggml_backend_tensor_memset(s.device_tail, 0, 0, ggml_nbytes(s.device_tail));
            }
            if (position == 0 || n % 4 == 0) {
                auto *row = Impl::view(s.copy_ctx.get(), s.device_pool, 1, complete * ID * sizeof(float));
                ggml_backend_tensor_copy(s.device_spare, row);
            }
            const std::array<float, 4> biases = n % 4 == 0 ? std::array<float, 4>{{0, 0, 0, 0}}
                                                          : std::array<float, 4>{{1e9f, 1e9f, 1e9f, 1e9f}};
            const size_t start = size_t(position / 4) * 4, active = n - start;
            ggml_backend_tensor_set(s.device_bias, biases.data(), start * sizeof(float), active * sizeof(float));
        } else {
            next_tail = cursor.tail;
            std::copy(ik.begin(), ik.end(), next_tail.begin() + size_t(position % 4) * ID);
            // The spare is normalized raw cell0 at RoPE position0, not a zero
            // vector. Keep it in the final pooled row and move it on completion.
            // Everything here is candidate-only until the existing FFN commit.
            std::vector<float> spare(cursor.pooled.end() - ID, cursor.pooled.end());
            if (position == 0) {
                std::vector<float> first_block(4 * ID);
                for (size_t member = 0; member < 4; ++member)
                    std::copy(ik.begin(), ik.end(), first_block.begin() + member * ID);
                write(s.pool_input, first_block);
                ggml_backend_tensor_set(s.pool_pos, positions.data(), 0, sizeof(positions));
                s.pool->run();
                spare = read(s.pool_output);
            }
            if (n % 4 == 0) {
                write(s.pool_input, next_tail);
                const int32_t start = int32_t(n - 4);
                positions = {{start, start, start, start}};
                ggml_backend_tensor_set(s.pool_pos, positions.data(), 0, sizeof(positions));
                s.pool->run();
                new_pool = read(s.pool_output);
                std::fill(next_tail.begin(), next_tail.end(), 0);
            }
            // Only active history is staged; no maximum-capacity copy per token.
            next_pool.assign((complete + 1) * ID, 0);
            std::copy(cursor.pooled.begin(), cursor.pooled.end(), next_pool.begin());
            if (!new_pool.empty()) {
                next_pool.resize((complete + 1) * ID, 0);
                std::copy(new_pool.begin(), new_pool.end(), next_pool.begin() + (complete - 1) * ID);
            }
            std::copy(spare.begin(), spare.end(), next_pool.begin() + complete * ID);
            next_keys.resize(n * KV);
            next_values.resize(n * KV);
            std::copy(cursor.keys.begin(), cursor.keys.end(), next_keys.begin());
            std::copy(cursor.values.begin(), cursor.values.end(), next_values.begin());
            std::copy(hk.begin(), hk.end(), next_keys.begin() + (n - 1) * KV);
            std::copy(hv.begin(), hv.end(), next_values.begin() + (n - 1) * KV);
        }
        std::unique_ptr<Phase> temporary;
        if (!s.attention)
            temporary = std::make_unique<Phase>();
        auto &attention = s.attention ? *s.attention : *temporary;
        if (s.attention)
            attention.rebuild();
        auto [ki, vi, qi, idxq, pk, mapping, bias, scores, selected, out] = s.build_attention(attention, n);
        auto next_placement = s.fixed_placement;
        size_t next_graphs = s.fixed_graphs+s.chunk_graphs();
        attention.schedule("qsa-attend", s.options.execution, s.cpu, next_placement, next_graphs, bool(s.attention));
        if (s.attention) {
            // Replanning may change offsets/topology (notably at 2051 cells).
            // Account the retained high water even after a rejected candidate.
            s.retained_attention_bytes = std::max(s.retained_attention_bytes, next_graphs - s.fixed_graphs-s.chunk_graphs());
            next_graphs = s.fixed_graphs+s.retained_attention_bytes+s.chunk_graphs();
        }
        if (!s.options.production) {
            ggml_backend_tensor_set(ki, next_keys.data(), 0, next_keys.size() * sizeof(uint16_t));
            ggml_backend_tensor_set(vi, next_values.data(), 0, next_values.size() * sizeof(uint16_t));
            write(pk, next_pool);
            std::vector<int32_t> map(n);
            std::vector<float> biases(n, 0);
            for (size_t i = 0; i < n; ++i) {
                map[i] = int32_t(i / 4);
                if (i >= complete * 4)
                    biases[i] = 1e9f;
            }
            ggml_backend_tensor_set(mapping, map.data(), 0, n * sizeof(int32_t));
            write(bias, biases);
        }
        if (!s.options.device_combine) {
            write(qi, q);
            write(idxq, iq);
        }
        attention.run();
        ++s.graph_work.attention_submissions;
        if (s.options.device_combine) ++s.graph_work.device_handoffs;
        auto attn = read(out);
        auto cell_scores = s.options.production ? std::vector<float>{} : read(scores);
        auto selection = s.options.production ? std::vector<float>{} : read(selected);
        if (!s.options.device_combine) {
            write(s.out_input, attn);
            write(s.out_gate, gate);
            write(s.out_residual, residual);
            write(s.out_inject, inject);
            s.combine->run();
            ++s.graph_work.combine_submissions;
        }
        auto next_residual = read(s.attn_residual);
        auto gated = read(s.gated);
        PortableQsaTrace next_trace;
        if (trace)
            next_trace = {q, k, v, gate, iq, ik, new_pool, cell_scores, selection, attn, gated, next_residual};
        return {std::move(next_residual),std::move(hk),std::move(hv),std::move(next_pool),
                std::move(next_tail),std::move(next_placement),next_graphs-s.chunk_graphs(),std::move(next_trace),std::move(temporary)};
    }

    std::unique_ptr<Phase> project, pool, combine, attention;
    size_t retained_attention_bytes = 0;
    ggml_tensor *input, *pos, *q, *k, *v, *gate, *iq, *ik, *inject;
    ggml_tensor *pool_input, *pool_pos, *pool_output;
    ggml_tensor *out_input, *out_residual, *out_inject, *out_gate, *gated, *attn_residual;
    std::vector<uint16_t> keys, values;
    std::vector<float> pooled, tail, last;
    std::vector<PortablePlacement> fixed_placement, placement;
    size_t attention_device_weights = 0;
    size_t weights_size = 0, fixed_graphs = 0, graphs_size = 0, states_size = 0, conversion_scratch = 0;
    uint64_t count = 0;
    PortableQsaGraphWork graph_work;
    bool healthy = true;
    ggml_tensor *device_k = nullptr, *device_v = nullptr, *device_pool = nullptr, *device_spare = nullptr,
                *device_tail = nullptr, *device_mapping = nullptr, *device_bias = nullptr;

    // Views alias externally allocated device state; no scheduled prefix inputs.
    static ggml_tensor *view(ggml_context *c, ggml_tensor *src, int64_t rows, size_t offset = 0) {
        auto *t = ggml_view_2d(c, src, src->ne[0], rows, src->nb[1], offset);
        require(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS, "portable QSA: state view binding failed");
        return t;
    }
    void initialize_state() {
        ggml_backend_buffer_clear(state_buffer.get(), 0);
        std::vector<int32_t> map(options.capacity);
        for (size_t i = 0; i < map.size(); ++i)
            map[i] = int32_t(i / 4);
        ggml_backend_tensor_set(device_mapping, map.data(), 0, map.size() * sizeof(int32_t));
        ggml_backend_synchronize(options.execution.accelerator);
    }
    void allocate_state() {
        state_ctx = context();
        copy_ctx.reset(ggml_init({4096, nullptr, true}));
        require(bool(copy_ctx), "portable QSA: copy metadata allocation failed");
        auto *c = state_ctx.get();
        auto tensors = state_tensors(c, options.capacity);
        device_k = tensors[0]; device_v = tensors[1]; device_pool = tensors[2];
        device_spare = tensors[3]; device_tail = tensors[4];
        device_mapping = tensors[5]; device_bias = tensors[6];
        const size_t aligned = state_allocation_size(c, options.execution.accelerator);
        require(aligned <= options.state_budget - states_size, "portable QSA: aligned device state budget exceeded");
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(c, options.execution.accelerator));
        require(bool(state_buffer), "portable QSA: device state allocation failed");
        require(ggml_backend_buffer_get_size(state_buffer.get()) <= options.state_budget - states_size,
                "portable QSA: actual device state budget exceeded");
        initialize_state();
    }

    // The project scheduler owns these already allocated outputs/inputs. A leaf
    // view shares their backend storage without importing the project DAG into
    // this graph. Project execution and CPU half(k/v) validation finish first.
    ggml_tensor *boundary(ggml_context *c, ggml_tensor *source) {
        require(source->buffer && ggml_backend_buffer_get_type(source->buffer) ==
                    ggml_backend_get_default_buffer_type(options.execution.accelerator),
                "portable QSA: project handoff must stay on selected backend");
        auto *t = ggml_view_tensor(c, source);
        t->op = GGML_OP_NONE;
        std::fill(std::begin(t->src), std::end(t->src), nullptr);
        require(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS, "portable QSA: project view binding failed");
        return t;
    }
    void build_combine(Phase &phase, ggml_tensor *attention_output, ggml_tensor *raw_gate,
                       ggml_tensor *residual, ggml_tensor *injection) {
        auto outputs = portable_detail::qsa_combine(phase.ctx.get(), attention_output, raw_gate,
            w.at("attn_output.weight"), residual, injection);
        gated = outputs.first;
        attn_residual = outputs.second;
        phase.output(gated, "qsa-gated-attention");
        phase.output(attn_residual, "qsa-attention-residual");
    }

    struct AttentionInputs {
        ggml_tensor *ki, *vi, *qi, *idxq, *pk, *mapping, *bias, *scores, *selected, *out;
    };
    AttentionInputs build_attention(Phase &phase, size_t n) {
        auto *c = phase.ctx.get();
        const int64_t cells = int64_t(n), blocks = int64_t(n / 4 + 1);
        auto *ki = options.production ? ggml_view_3d(c, device_k, D, KH, cells, device_k->nb[1], device_k->nb[2], 0)
                                      : phase.input({D, KH, cells}, GGML_TYPE_F16);
        auto *vi = options.production ? ggml_view_3d(c, device_v, D, KH, cells, device_v->nb[1], device_v->nb[2], 0)
                                      : phase.input({D, KH, cells}, GGML_TYPE_F16);
        auto *qi = options.device_combine ? ggml_reshape_3d(c, boundary(c, q), D, 12, KH) : phase.input({D, 12, KH});
        auto *idxq = options.device_combine ? ggml_reshape_2d(c, boundary(c, iq), ID, IH) : phase.input({ID, IH});
        auto *pk = options.production ? view(c, device_pool, blocks) : phase.input({ID, blocks});
        auto *mapping = options.production ? ggml_view_1d(c, device_mapping, cells, 0) : phase.input({cells}, GGML_TYPE_I32);
        auto *bias = options.production ? ggml_view_1d(c, device_bias, cells, 0) : phase.input({cells});
        if (options.production)
            for (auto *t : {ki, vi, mapping, bias})
                require(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS, "portable QSA: active view binding failed");
        auto *scores = portable_detail::qsa_cell_scores(c, pk, idxq, mapping, bias);
        auto *selected = portable_detail::qsa_selection(c, scores);
        auto *out = portable_detail::qsa_dense_attention(c, ki, vi, qi, selected);
        phase.output(scores, "qsa-cell-scores");
        phase.output(selected, "qsa-selected-mask");
        phase.output(out, "qsa-attention");
        if (options.device_combine)
            build_combine(phase, out, boundary(c, gate), boundary(c, input), boundary(c, inject));
        return {ki, vi, qi, idxq, pk, mapping, bias, scores, selected, out};
    }

    Impl(int64_t layer, const std::vector<PortableTensorView> &tensors, PortableQsaOptions opts) : options(opts) {
        require(layer >= 0 && layer < 48 && layer % 4 == 3, "portable QSA: expected a QSA layer 3/7/.../47");
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        require(options.capacity > 0 && options.capacity <= 262144, "portable QSA: capacity must be in [1,262144]");
        require(!options.production || (options.reuse_attention && options.execution.request_accelerator &&
                    options.execution.accelerator && !options.execution.allow_cpu_fallback),
                "portable QSA: production requires reuse and strict selected accelerator");
        require(!options.device_combine || options.production,
                "portable QSA: device combine requires production");
        // Capacity bound also keeps float prefix counts exact. Price committed
        // and active candidate FP16 caches, pooled state, F32 traces, readbacks,
        // residual/FFN staging and fixed metadata. Scheduler storage is separate.
        states_size = options.capacity * (4 * KV * sizeof(uint16_t) + 8 * sizeof(float)) +
                      (options.capacity / 4 + 2) * ID * 3 * sizeof(float) + (32 * RES + 16384) * sizeof(float) +
                      (32u << 20);
        if (options.production)
            states_size -= options.capacity * (4 * KV * sizeof(uint16_t) + 8 * sizeof(float)) +
                           (options.capacity / 4 + 2) * ID * 3 * sizeof(float);
        require(states_size <= options.state_budget, "portable QSA: host state budget exceeded");
        std::vector<PortableTensorView> aw, fw;
        for (const auto &t : tensors) {
            require(t.name.rfind(prefix, 0) == 0, "portable QSA: tensor belongs to another layer");
            (shapes().count(t.name.substr(prefix.size())) ? aw : fw).push_back(t);
        }
        require(aw.size() == shapes().size() && fw.size() == 9, "portable QSA: missing/extra attention or FFN tensors");
        std::string error;
        ffn = PortableFfn::create({H, HC, R, 640, 512, 10, 1e-6f}, layer, fw, options.execution, error);
        require(bool(ffn), error);
        weights_size = ffn->weight_bytes();
        fixed_graphs = ffn->graph_bytes();
        if (options.execution.request_accelerator && options.execution.accelerator) {
            // Pinned Vulkan has private X/Y conversion, split-K (capped at 8),
            // radix-selection and prefix-scan buffers outside scheduler buffers.
            // A full F32 image of the largest weight plus 8 MiB and 64 B/cell
            // bounds those tensor workspaces for this fixed graph family. Driver
            // pipelines/command pools and unrelated users of the borrowed backend
            // remain caller-owned and outside these per-layer limits.
            const size_t workspace = size_t(H * 2 * D * QH) * sizeof(float) + (8u << 20) + options.capacity * 64;
            require(fixed_graphs <= options.execution.graph_budget &&
                        workspace <= options.execution.graph_budget - fixed_graphs,
                    "portable QSA: accelerator workspace budget exceeded");
            fixed_graphs += workspace;
        }
        fixed_placement = ffn->placements();
        cpu.reset(ggml_backend_cpu_init());
        require(bool(cpu), "portable QSA: CPU unavailable");
        ggml_backend_cpu_set_n_threads(cpu.get(), options.execution.threads);
        weights_ctx = context();
        for (const auto &view : aw) {
            const auto key = view.name.substr(prefix.size());
            const auto &shape = shapes().at(key);
            require(!w.count(key) && view.shape == shape, "portable QSA: bad/duplicate tensor shape: " + view.name);
            const bool indexer = key == "indexer.q_proj.weight" || key == "indexer.k_proj.weight";
            const bool widen = indexer && view.type == 30;
            // Real fleet indexer projections are BF16; legacy Q8 fixtures and
            // exact F32 references remain supported for these two roles only.
            require(view.data && (view.type == (shape.size() == 2 ? 8u : 0u) || widen || (indexer && view.type == 0)),
                    "portable QSA: unsupported encoding for role: " + view.name);
            const size_t rows = shape.size() == 2 ? size_t(shape[1]) : 1;
            const size_t source_bytes = ggml_row_size(ggml_type(view.type), shape[0]) * rows;
            auto *t = ggml_new_tensor(weights_ctx.get(), widen ? GGML_TYPE_F32 : ggml_type(view.type),
                                      int(shape.size()), shape.data());
            ggml_set_name(t, view.name.c_str());
            const size_t storage_bytes = ggml_nbytes(t);
            require(view.bytes == source_bytes && storage_bytes <= options.execution.weight_budget - weights_size,
                    "portable QSA: weight size/budget mismatch");
            weights_size += storage_bytes;
            if (widen)
                conversion_scratch = std::max(conversion_scratch, size_t(shape[0]) * sizeof(float));
            w[key] = t;
        }
        auto *weight_backend = portable_detail::weight_backend(options.execution, cpu.get());
        const size_t allocation = ggml_backend_alloc_ctx_tensors_from_buft_size(
            weights_ctx.get(), ggml_backend_get_default_buffer_type(weight_backend));
        require(allocation <= options.execution.weight_budget - ffn->weight_bytes(),
                "portable QSA: aligned weight budget exceeded");
        weights_buffer.reset(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), weight_backend));
        require(bool(weights_buffer), "portable QSA: weight allocation failed");
        ggml_backend_buffer_set_usage(weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto &view : aw) {
            auto *target = w.at(view.name.substr(prefix.size()));
            if (view.type != 30) {
                ggml_backend_tensor_set(target, view.data, 0, view.bytes);
                continue;
            }
            // Reuse the existing exact BF16 decoder. One row of staging is
            // bounded; the sole persistent copy is the admitted F32 backend buffer.
            std::vector<float> row(size_t(view.shape[0]));
            const auto *source = static_cast<const uint8_t *>(view.data);
            for (size_t r = 0; r < size_t(view.shape[1]); ++r) {
                dequantize_bf16(source + r * row.size() * 2, row.data(), int(row.size()));
                ggml_backend_tensor_set(target, row.data(), r * row.size() * sizeof(float), row.size() * sizeof(float));
            }
            fixed_placement.push_back({"load", "BF16_TO_F32_EXACT", view.name, "BF16->F32", "CPU", "", view.shape});
        }
        const size_t actual_weights = ggml_backend_buffer_get_size(weights_buffer.get());
        require(actual_weights <= options.execution.weight_budget - ffn->weight_bytes(),
                "portable QSA: actual aligned weight budget exceeded");
        attention_device_weights = weight_backend != cpu.get() ? actual_weights : 0;
        weights_size = ffn->weight_bytes() + actual_weights;
        project = std::make_unique<Phase>();
        auto *c = project->ctx.get();
        input = project->input({H, HC, 1});
        pos = project->input({4}, GGML_TYPE_I32);
        auto *norm =
            ggml_mul(c, ggml_reshape_2d(c, ggml_rms_norm(c, input, 1e-6f), RES, 1), w.at("hc_attn_norm.weight"));
        auto *low = ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, w.at("hc_attn_down.weight"), norm), 0.25f));
        auto *gates = ggml_sigmoid(c, ggml_mul_mat(c, w.at("hc_attn_up.weight"), low));
        auto *weighted = ggml_reshape_2d(c, ggml_mul(c, norm, gates), H, HC);
        auto *mixed = ggml_cont(c, ggml_view_1d(c, weighted, H, 0));
        for (int s = 1; s < HC; ++s)
            mixed = ggml_add(c, mixed, ggml_view_1d(c, weighted, H, size_t(s * H) * sizeof(float)));
        mixed = ggml_scale(c, mixed, 0.25f);
        inject = ggml_mul_mat(c, w.at("hc_attn_inject.weight"), norm);
        auto *full = ggml_mul_mat(c, w.at("attn_q.weight"), mixed);
        q = ggml_cont(c, ggml_view_3d(c, full, D, QH, 1, 2 * D * sizeof(float), 2 * D * QH * sizeof(float), 0));
        gate = ggml_cont(
            c, ggml_view_3d(c, full, D, QH, 1, 2 * D * sizeof(float), 2 * D * QH * sizeof(float), D * sizeof(float)));
        q = rope(c, ggml_mul(c, ggml_rms_norm(c, q, 1e-6f), w.at("attn_q_norm.weight")), pos);
        k = ggml_reshape_3d(c, ggml_mul_mat(c, w.at("attn_k.weight"), mixed), D, KH, 1);
        k = rope(c, ggml_mul(c, ggml_rms_norm(c, k, 1e-6f), w.at("attn_k_norm.weight")), pos);
        v = ggml_mul_mat(c, w.at("attn_v.weight"), mixed);
        auto *index_query = ggml_mul_mat(c, w.at("indexer.q_proj.weight"), mixed);
        ik = ggml_mul_mat(c, w.at("indexer.k_proj.weight"), mixed);
        // F32 storage must not select a lower-precision accumulation path.
        if (w.at("indexer.q_proj.weight")->type == GGML_TYPE_F32)
            ggml_mul_mat_set_prec(index_query, GGML_PREC_F32);
        if (w.at("indexer.k_proj.weight")->type == GGML_TYPE_F32)
            ggml_mul_mat_set_prec(ik, GGML_PREC_F32);
        iq = ggml_reshape_3d(c, index_query, ID, IH, 1);
        iq = rope(c, ggml_mul(c, ggml_rms_norm(c, iq, 1e-6f), w.at("indexer.q_norm.weight")), pos);
        for (auto pair : {std::pair<ggml_tensor *, const char *>{q, "qsa-query"},
                          {k, "qsa-key"},
                          {v, "qsa-value"},
                          {gate, "qsa-raw-gate"},
                          {iq, "qsa-index-query"},
                          {ik, "qsa-index-raw-key"},
                          {inject, "qsa-inject"}})
            project->output(pair.first, pair.second);
        // An input flag alone does not prevent gallocr's in-place reuse after
        // its last project consumer. The later graph still needs this residual.
        if (options.device_combine) project->output(input, "qsa-combine-residual-input");
        project->schedule("qsa-project", options.execution, cpu, fixed_placement, fixed_graphs);
        if (options.production)
            allocate_state();
        pool = std::make_unique<Phase>();
        c = pool->ctx.get();
        pool_input = options.production ? ggml_view_3d(c, device_tail, ID, 4, 1, device_tail->nb[1], device_tail->nb[2], 0)
                                        : pool->input({ID, 4, 1});
        if (options.production)
            require(ggml_backend_view_init(pool_input) == GGML_STATUS_SUCCESS, "portable QSA: pool tail binding failed");
        auto *mean = ggml_cont(c, ggml_view_3d(c, pool_input, ID, 1, 1, ID * sizeof(float), 4 * ID * sizeof(float), 0));
        for (int b = 1; b < 4; ++b)
            mean = ggml_add(c, mean,
                            ggml_view_3d(c, pool_input, ID, 1, 1, ID * sizeof(float), 4 * ID * sizeof(float),
                                         b * ID * sizeof(float)));
        mean = ggml_scale(c, mean, 0.25f);
        pool_pos = pool->input({4}, GGML_TYPE_I32);
        pool_output = rope(c, ggml_mul(c, ggml_rms_norm(c, mean, 1e-6f), w.at("indexer.k_norm.weight")), pool_pos);
        pool->output(pool_output, "qsa-pooled-key");
        pool->schedule("qsa-pool", options.execution, cpu, fixed_placement, fixed_graphs);
        if (!options.device_combine) {
            combine = std::make_unique<Phase>();
            c = combine->ctx.get();
            out_input = combine->input({D * QH});
            out_gate = combine->input({D * QH});
            out_residual = combine->input({H, HC, 1});
            out_inject = combine->input({HC});
            gated = ggml_mul(c, out_input, ggml_sigmoid(c, out_gate));
            auto *projected = ggml_mul_mat(c, w.at("attn_output.weight"), gated);
            auto *scatter =
                ggml_reshape_3d(c, ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, out_inject, 0.25f)), 2.0f), 1, HC, 1);
            auto *repeat = ggml_repeat_4d(c, ggml_reshape_3d(c, projected, H, 1, 1), H, HC, 1, 1);
            attn_residual = ggml_add(c, out_residual, ggml_mul(c, repeat, scatter));
            combine->output(gated, "qsa-gated-attention");
            combine->output(attn_residual, "qsa-attention-residual");
            combine->schedule("qsa-combine", options.execution, cpu, fixed_placement, fixed_graphs);
        }
        if (!options.production) {
            keys.reserve(options.capacity * KV);
            values.reserve(options.capacity * KV);
            pooled.reserve((options.capacity / 4 + 2) * ID);
            pooled.assign(ID, 0);
            tail.assign(4 * ID, 0);
        }
        placement = fixed_placement;
        graphs_size = fixed_graphs;
        if (options.reuse_attention) {
            attention = std::make_unique<Phase>();
            build_attention(*attention, options.capacity);
            auto reserve_placement = fixed_placement;
            attention->schedule("qsa-attend", options.execution, cpu, reserve_placement, graphs_size, true);
            retained_attention_bytes = graphs_size - fixed_graphs;
            // Capacity priming allocates scratch but performs no attention math.
            // Exact active graphs are rebuilt in this same arena, never padded.
        }
    }
};
size_t PortableQsaLayer::device_state_allocation_size(ggml_backend *backend, size_t capacity) {
    Context c(ggml_init({4096, nullptr, true}), ggml_free);
    require(bool(c), "portable QSA: state estimate metadata allocation failed");
    state_tensors(c.get(), capacity);
    return state_allocation_size(c.get(), backend);
}
PortableQsaLayer::PortableQsaLayer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PortableQsaLayer::~PortableQsaLayer() = default;
std::unique_ptr<PortableQsaLayer> PortableQsaLayer::create(int64_t layer, const std::vector<PortableTensorView> &views,
                                                           PortableQsaOptions opts, std::string &error) {
    error.clear();
    try {
        return std::unique_ptr<PortableQsaLayer>(new PortableQsaLayer(std::make_unique<Impl>(layer, views, opts)));
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
std::unique_ptr<PortableQsaLayer> PortableQsaLayer::from_gguf(const GgufModel &model, int64_t layer,
                                                              PortableQsaOptions opts, std::string &error) {
    error.clear();
    try {
        require(layer >= 0 && layer < 48 && layer % 4 == 3, "portable QSA: expected a QSA layer 3/7/.../47");
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto &meta = model.meta();
        const auto *arch = meta.get("general.architecture");
        require(arch && arch->s == "qwen4exp", "portable QSA: requires qwen4exp");
        require(check_architecture(meta).empty(), "portable QSA: unsupported model architecture");
        for (auto pair : {std::pair<const char *, uint64_t>{"embedding_length", H},
                          {"block_count", 48},
                          {"hyper_connection.count", HC},
                          {"hyper_connection.low_rank", R},
                          {"expert_count", 512},
                          {"expert_used_count", 10},
                          {"expert_feed_forward_length", 640},
                          {"attention.head_count", QH},
                          {"attention.head_count_kv", KH},
                          {"attention.key_length", D},
                          {"attention.value_length", D},
                          {"attention.indexer.head_count", IH},
                          {"attention.indexer.key_length", ID},
                          {"attention.indexer.top_k", 2048},
                          {"rope.dimension_count", 64},
                          {"full_attention_interval", 4}}) {
            const auto *v = meta.get("qwen4exp." + std::string(pair.first));
            require(v && v->is_num() && v->u == pair.second,
                    "portable QSA: missing/mismatched " + std::string(pair.first));
        }
        auto number = [&](const char *key, float expected, bool required) {
            const auto *v = meta.get("qwen4exp." + std::string(key));
            if (!v) {
                require(!required, "portable QSA: missing " + std::string(key));
                return;
            }
            require((v->type == MetaType::F32 || v->type == MetaType::F64) && std::isfinite(v->f) &&
                        float(v->f) == expected,
                    "portable QSA: unsupported " + std::string(key));
        };
        number("rope.freq_base", 1e7f, true);
        number("attention.layer_norm_rms_epsilon", 1e-6f, true);
        number("rope.scaling.factor", 1.0f, false);
        number("rope.scale_linear", 1.0f, false);
        number("rope.scaling.alpha", 1.0f, false);
        number("rope.scaling.attn_factor", 1.0f, false);
        number("attention.scale", 1.0f / 16.0f, false);
        const auto *scaling = meta.get("qwen4exp.rope.scaling.type");
        require(!scaling || (scaling->type == MetaType::STRING && (scaling->s == "none" || scaling->s == "linear")),
                "portable QSA: RoPE scaling unsupported");
        // Reject unhandled extension fields rather than silently running default
        // RoPE.
        for (const char *key :
             {"rope.scaling.finetuned", "rope.scaling.original_context_length", "rope.scaling.yarn_log_multiplier",
              "rope.scaling.yarn_ext_factor", "rope.scaling.yarn_attn_factor", "rope.scaling.yarn_beta_fast",
              "rope.scaling.yarn_beta_slow", "rope.freq_factors"})
            require(!meta.get("qwen4exp." + std::string(key)), "portable QSA: unsupported " + std::string(key));
        const auto *sections = meta.get("qwen4exp.rope.dimension_sections");
        require(sections && sections->type == MetaType::ARRAY && sections->items.size() == 4,
                "portable QSA: missing RoPE sections");
        const uint64_t expected[]{11, 11, 10, 0};
        for (size_t i = 0; i < 4; ++i)
            require(sections->items[i].is_num() && sections->items[i].u == expected[i],
                    "portable QSA: unsupported RoPE sections");
        const auto *ratios = meta.get("qwen4exp.attention.compress_ratios");
        require(ratios && ratios->type == MetaType::ARRAY && ratios->items.size() == 48,
                "portable QSA: explicit compression ratios required");
        for (size_t i = 0; i < 48; ++i)
            require(ratios->items[i].is_num() && ratios->items[i].u == (i % 4 == 3 ? 4u : 0u),
                    "portable QSA: unsupported compression ratios");
        const auto *ple = meta.get("qwen4exp.ple.layers");
        if (ple) {
            require(ple->type == MetaType::ARRAY, "portable QSA: malformed PLE layers");
            for (const auto &v : ple->items)
                require(v.is_num() && v.u != uint64_t(layer), "portable QSA: PLE at a QSA layer unsupported");
        }
        std::vector<PortableTensorView> views;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto &t : model.shard(s).tensors()) {
                if (t.name.rfind(prefix, 0) != 0)
                    continue;
                const auto suffix = t.name.substr(prefix.size());
                // Routed expert payloads are owned by the existing callback. Reject
                // any other unrecognized layer tensor instead of ignoring biases or
                // extension weights that would change the layer's semantics.
                if (suffix.rfind("ffn_", 0) == 0 && suffix.find("_exps") != std::string::npos)
                    continue;
                require(model.in_bounds(t, s), "portable QSA: tensor outside shard");
                PortableTensorView v{
                    t.name, {}, t.type, model.shard(s).tensor_data(t), size_t(tensor_payload_bytes(t))};
                for (auto d : t.shape) {
                    require(d <= INT64_MAX, "portable QSA: dimension overflow");
                    v.shape.push_back(int64_t(d));
                }
                views.push_back(std::move(v));
            }
        return create(layer, views, opts, error);
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
bool PortableQsaLayer::step(uint64_t position, const std::vector<float> &residual, const PortableExpertRows &dispatch,
                            PortableRoute &route, std::string &error, PortableQsaTrace *trace) {
    error.clear();
    route = {};
    if (trace)
        *trace = {};
    try {
        auto &s = *impl_;
        require(!s.chunk_active,"portable QSA: chunk execution active");
        require(s.healthy, "portable QSA: device state poisoned; reset required");
        require(!s.options.production || !trace, "portable QSA: trace requires eager diagnostic policy");
        require(position == s.count, "portable QSA: positions must be contiguous "
                                     "text positions starting zero");
        require(s.count < s.options.capacity, "portable QSA: cache capacity exhausted");
        require(residual.size() == RES && bool(dispatch), "portable QSA: residual/callback required");
        auto attended = s.attend(position,residual,{s.keys,s.values,s.pooled,s.tail},trace != nullptr);
        auto next_residual = std::move(attended.residual);
        auto& hk = attended.key; auto& hv = attended.value;
        auto& next_pool = attended.pooled; auto& next_tail = attended.tail;
        auto next_placement = std::move(attended.placement);
        const size_t next_graphs = attended.graphs;
        const size_t n = size_t(position)+1;
        auto next_trace = std::move(attended.trace);
        require(s.ffn->reset(next_residual, error), error);
        PortableRoute candidate;
        require(s.ffn->step(dispatch, candidate, error), error);
        next_residual = s.ffn->residual();
        finite(next_residual);
        // Publish only after all computations succeed. Diagnostics append to
        // reserved host history here; production already appended device state
        // and poisons the layer on any failure before this point.
        if (!s.options.production) {
            s.keys.resize(n * KV);
            s.values.resize(n * KV);
            std::copy(hk.begin(), hk.end(), s.keys.begin() + (n - 1) * KV);
            std::copy(hv.begin(), hv.end(), s.values.begin() + (n - 1) * KV);
            s.pooled.resize(next_pool.size());
            std::copy(next_pool.begin(), next_pool.end(), s.pooled.begin());
            s.tail.swap(next_tail);
        }
        s.last.swap(next_residual);
        s.placement.swap(next_placement);
        s.graphs_size = next_graphs;
        ++s.count;
        route = std::move(candidate);
        if (trace)
            *trace = std::move(next_trace);
        return true;
    } catch (const std::exception &e) {
        if (impl_->options.production)
            impl_->healthy = false;
        error = e.what();
        return false;
    } catch (...) {
        if (impl_->options.production)
            impl_->healthy = false;
        error = "portable QSA: unknown failure; reset required";
        return false;
    }
}
bool PortableQsaLayer::prepare_chunk(size_t tokens,std::string& error) {
    error.clear();
    try { require(!impl_->chunk_active,"portable layer: active chunk"); impl_->prepare(tokens,error); return true; }
    catch(const std::exception& e) { error=e.what(); return false; }
}
bool PortableQsaLayer::run_chunk(const std::vector<int32_t>& tokens,uint64_t position0,
    std::vector<float>& residual_rows,const PortableExpertRowsBatch& dispatch,
    std::vector<PortableRoute>& routes,std::string& error) {
    error.clear(); routes.clear(); auto& s = *impl_;
    try {
        require(s.healthy && !s.chunk_active,"portable QSA: poisoned or active chunk; reset required");
        const size_t T=tokens.size();
        require(T>=1 && T<=8 && residual_rows.size()==T*RES && bool(dispatch),"portable QSA: invalid chunk shape/callback");
        require(position0==s.count && T<=s.options.capacity-s.count && T<=UINT64_MAX-s.chunk_work.tokens,
                "portable QSA: noncontiguous/exhausted chunk");
        struct Active { bool& flag; explicit Active(bool& f):flag(f){flag=true;} ~Active(){flag=false;} } active(s.chunk_active);
        s.prepare(T,error);
        auto keys=s.keys, values=s.values; auto pooled=s.pooled, tail=s.tail;
        if(!s.options.production) {
            keys.reserve(s.options.capacity*KV); values.reserve(s.options.capacity*KV);
            pooled.reserve((s.options.capacity/4+2)*ID);
        }
        std::vector<float> attended_rows; attended_rows.reserve(T*RES);
        std::vector<PortablePlacement> placement;
        size_t graphs=s.graphs_size;
        for(size_t t=0;t<T;++t) {
            const size_t n=size_t(position0)+t+1;
            std::vector<float> input(residual_rows.begin()+t*RES,residual_rows.begin()+(t+1)*RES);
            auto next=s.attend(position0+t,input,{keys,values,pooled,tail},false);
            if(!s.options.production) {
                keys.resize(n*KV); values.resize(n*KV);
                std::copy(next.key.begin(),next.key.end(),keys.begin()+(n-1)*KV);
                std::copy(next.value.begin(),next.value.end(),values.begin()+(n-1)*KV);
                pooled.resize(next.pooled.size()); std::copy(next.pooled.begin(),next.pooled.end(),pooled.begin());
                tail.swap(next.tail);
            }
            attended_rows.insert(attended_rows.end(),next.residual.begin(),next.residual.end());
            placement.swap(next.placement); graphs=next.graphs;
        }
        auto& ffn=T==1?*s.ffn:*s.chunk_ffn;
        const auto before=ffn.work(); require(ffn.reset(attended_rows,error),error);
        std::vector<PortableRoute> candidate;
        if(T==1) {
            PortableRoute route;
            require(ffn.step([&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e){
                return dispatch(l,{r},rows,e);
            },route,error),error);
            candidate.push_back(std::move(route));
        } else require(ffn.step_batch(dispatch,candidate,error),error);
        auto completed=ffn.residual();
        std::vector<float> last(completed.end()-RES,completed.end());
        placement.erase(placement.begin(),placement.begin()+s.ffn->placements().size());
        placement.insert(placement.begin(),ffn.placements().begin(),ffn.placements().end());
        const auto after=ffn.work(); auto work=s.chunk_work;
        ++work.chunks; work.tokens+=T; work.attention_calls+=T;
        work.ffn_pre_calls+=after.pre_graph_calls-before.pre_graph_calls;
        work.provider_calls+=after.provider_calls-before.provider_calls;
        work.ffn_post_calls+=after.post_graph_calls-before.post_graph_calls;
        require(s.healthy,"portable QSA: chunk invalidated; reset required");
        if(!s.options.production) { s.keys.swap(keys); s.values.swap(values); s.pooled.swap(pooled); s.tail.swap(tail); }
        s.last.swap(last); s.placement.swap(placement); s.graphs_size=graphs; s.count+=T; s.chunk_work=work;
        residual_rows.swap(completed); routes.swap(candidate);
        return true;
    } catch(const std::exception& e) { s.healthy=false; error=e.what(); return false; }
      catch(...) { s.healthy=false; error="portable QSA: unknown chunk failure; reset required"; return false; }
}
PortableQsaGraphWork PortableQsaLayer::graph_work() const { return impl_->graph_work; }
PortableLayerChunkWork PortableQsaLayer::chunk_work() const { return impl_->chunk_work; }
bool PortableQsaLayer::reset(std::string &error) {
    error.clear();
    try {
        auto &s = *impl_;
        // Poison before any allocating precondition/error message, as the
        // original production reset did before its first allocation.
        if (s.options.production)
            s.healthy = false;
        require(!s.chunk_active,"portable QSA: chunk execution active");
        s.chunk_ffn.reset(); s.chunk_host_bound = 0;
        std::vector<float> zero(RES, 0);
        require(s.ffn->reset(zero, error), error);
        auto placement = s.fixed_placement;
        if (s.options.production)
            s.initialize_state();
        else {
            s.keys.clear();
            s.values.clear();
            s.pooled.resize(ID);
            std::fill(s.pooled.begin(), s.pooled.end(), 0);
            std::fill(s.tail.begin(), s.tail.end(), 0);
        }
        s.last.clear();
        s.count = 0;
        s.placement.swap(placement);
        s.graphs_size = s.fixed_graphs + s.retained_attention_bytes;
        s.healthy = true;
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    } catch (...) {
        error = "portable QSA: unknown reset failure; reset required";
        return false;
    }
}
uint64_t PortableQsaLayer::tokens() const { return impl_->count; }
const std::vector<uint16_t> &PortableQsaLayer::keys() const {
    require(!impl_->options.production, "portable QSA: host history unavailable; use inspect_state");
    return impl_->keys;
}
const std::vector<uint16_t> &PortableQsaLayer::values() const {
    require(!impl_->options.production, "portable QSA: host history unavailable; use inspect_state");
    return impl_->values;
}
const std::vector<float> &PortableQsaLayer::pooled_keys() const {
    require(!impl_->options.production, "portable QSA: host history unavailable; use inspect_state");
    return impl_->pooled;
}
const std::vector<float> &PortableQsaLayer::indexer_tail() const {
    require(!impl_->options.production, "portable QSA: host history unavailable; use inspect_state");
    return impl_->tail;
}
const std::vector<float> &PortableQsaLayer::last_residual() const { return impl_->last; }
const std::vector<PortablePlacement> &PortableQsaLayer::placements() const { return impl_->placement; }
size_t PortableQsaLayer::weight_bytes() const { return impl_->weights_size; }
size_t PortableQsaLayer::graph_bytes() const {
    return (impl_->attention ? impl_->fixed_graphs+impl_->retained_attention_bytes : impl_->graphs_size)+impl_->chunk_graphs();
}
size_t PortableQsaLayer::state_bytes() const {
    return impl_->states_size+impl_->chunk_host_bound + (impl_->state_buffer ? ggml_backend_buffer_get_size(impl_->state_buffer.get()) : 0);
}
bool PortableQsaLayer::poisoned() const { return !impl_->healthy; }
bool PortableQsaLayer::inspect_state(PortableQsaState &state, std::string &error) const {
    error.clear();
    state = {};
    try {
        const auto &s = *impl_;
        require(s.options.production, "portable QSA: inspect_state requires device production");
        auto download = [](ggml_tensor *t, auto &data) {
            data.resize(size_t(ggml_nelements(t)));
            ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        };
        PortableQsaState candidate;
        download(s.device_k, candidate.keys);
        download(s.device_v, candidate.values);
        download(s.device_pool, candidate.pooled);
        download(s.device_spare, candidate.spare);
        download(s.device_tail, candidate.tail);
        download(s.device_mapping, candidate.mapping);
        download(s.device_bias, candidate.bias);
        state = std::move(candidate);
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}
PortableMemoryUsage PortableQsaLayer::memory_usage() const {
    const auto &s = *impl_;
    auto m = s.ffn->memory_usage();
    if(s.chunk_ffn) { std::string error; require(s.ffn->combined_memory_usage(*s.chunk_ffn,m,error),error); }
    m.device_weights += s.attention_device_weights;
    m.host_weights = s.weights_size - m.device_weights;
    for (auto *phase : {s.project.get(), s.pool.get(), s.combine.get(), s.attention.get()}) {
        if (!phase || !phase->scheduler)
            continue;
        m.host_graph += ggml_backend_sched_get_buffer_size(phase->scheduler.get(), s.cpu.get());
        if (s.options.execution.accelerator && s.options.execution.request_accelerator)
            m.device_graph +=
                ggml_backend_sched_get_buffer_size(phase->scheduler.get(), s.options.execution.accelerator);
    }
    // Legacy attention has a temporary scheduler; reuse retains its actual buffers.
    // Reserve the unspent aggregate layer graph cap on each backend, including workspaces; fixed
    // scheduler buffers above are actual per-backend allocations.
    const size_t buffers = m.host_graph + m.device_graph;
    const size_t allowance =
        s.options.execution.graph_budget > buffers ? s.options.execution.graph_budget - buffers : 0;
    m.device_state += s.state_buffer ? ggml_backend_buffer_get_size(s.state_buffer.get()) : 0;
    m.host_extra_bound += s.states_size+s.chunk_host_bound + (16u << 20) + allowance + s.conversion_scratch;
    if (s.options.execution.request_accelerator && s.options.execution.accelerator) {
        // fixed_graphs includes fixed scheduler buffers AND private workspace.
        // Legacy attention is token-local through expert dispatch; reuse owns
        // its buffer high water across steps, failures and reset.
        size_t fixed_buffers = buffers;
        if (s.attention && s.attention->scheduler) {
            fixed_buffers -= ggml_backend_sched_get_buffer_size(s.attention->scheduler.get(), s.cpu.get());
            fixed_buffers -= ggml_backend_sched_get_buffer_size(s.attention->scheduler.get(), s.options.execution.accelerator);
        }
        if(s.chunk_ffn) {
            const auto chunk=s.chunk_ffn->memory_usage();
            fixed_buffers-=chunk.host_graph+chunk.device_graph;
        }
        const size_t fixed_private = s.fixed_graphs > fixed_buffers ? s.fixed_graphs - fixed_buffers : 0;
        const size_t matrix_conversion=size_t(H*2*D*QH)*sizeof(float);
        require(fixed_private>=matrix_conversion,"portable QSA: fixed workspace accounting mismatch");
        m.device_extra_bound += fixed_private-matrix_conversion;
        portable_detail::account_matrix_conversion(m,s.options.execution.accelerator,matrix_conversion);
        if (s.attention) {
            // Future exact-shape replans may use the rest of this retained cap.
            m.device_extra_bound += s.options.execution.graph_budget - s.fixed_graphs-s.chunk_graphs() - (buffers-fixed_buffers-s.chunk_graphs());
        } else
            m.device_token_temporary_bound = s.options.execution.graph_budget > s.fixed_graphs+s.chunk_graphs()
                                            ? s.options.execution.graph_budget-s.fixed_graphs-s.chunk_graphs() : 0;
    } else if(s.options.execution.request_accelerator) {
        m.device_extra_bound += allowance; // preserve missing-backend fallback accounting
    }
    return m;
}
} // namespace strata::core
