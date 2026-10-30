#include "strata/core/portable_gdn.hpp"
#include "portable_expert_timer.hpp"
#include "portable_gdn_finite.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "portable_graph.hpp"
#include "portable_gdn_projection.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <cmath>
#include <map>

namespace strata::core {
namespace {
using portable_detail::require;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr int64_t H = 2560, HC = 4, R = 320, D = 128, HK = 16, HV = 48, INNER = D * HV, CHANNELS = 2 * D * HK + INNER;
constexpr size_t STATES = D * D * HV, HISTORY = 3 * CHANNELS, RESIDUAL = H * HC;
// Two committed/candidate copies plus graph-read residuals, reset scratch and
// optional trace vectors. Graph input/output buffers are priced separately.
constexpr size_t PLE_HISTORY = 9 * RESIDUAL;
constexpr size_t PLE_STATE_BYTES = (2 * PLE_HISTORY + 12 * RESIDUAL) * sizeof(float);
constexpr size_t HOST_STATE_BYTES =
    (2 * STATES + 2 * HISTORY + 8 * RESIDUAL + 8 * CHANNELS + 4 * INNER) * sizeof(float);
const std::map<std::string, std::vector<int64_t>> &shapes() {
    static const std::map<std::string, std::vector<int64_t>> s{{"hc_attn_norm.weight", {H * HC}},
                                                               {"hc_attn_down.weight", {H * HC, R}},
                                                               {"hc_attn_up.weight", {R, H * HC}},
                                                               {"hc_attn_inject.weight", {H * HC, HC}},
                                                               {"attn_qkv.weight", {H, CHANNELS}},
                                                               {"attn_gate.weight", {H, INNER}},
                                                               {"ssm_alpha.weight", {H, HV}},
                                                               {"ssm_beta.weight", {H, HV}},
                                                               {"ssm_a", {HV}},
                                                               {"ssm_dt.bias", {HV}},
                                                               {"ssm_conv1d.weight", {4, CHANNELS}},
                                                               {"ssm_norm.weight", {D}},
                                                               {"ssm_out.weight", {INNER, H}}};
    return s;
}
const std::map<std::string, std::vector<int64_t>> &ple_shapes() {
    static const std::map<std::string, std::vector<int64_t>> s{
        {"ple_key.weight", {H, H * HC}},    {"ple_value.weight", {H, H}},      {"ple_norm_key.weight", {H, HC}},
        {"ple_norm_query.weight", {H, HC}}, {"ple_norm_conv.weight", {H, HC}}, {"ple_conv1d.weight", {4, H * HC}}};
    return s;
}
Context context() {
    Context c(ggml_init({4u << 20, nullptr, true}), ggml_free);
    require(bool(c), "portable GDN: metadata allocation failed");
    return c;
}
void finite(const std::vector<float> &x) {
    if (portable_detail::gdn_all_finite(x))
        return;
    for (float f : x)
        if (!std::isfinite(f))
            require(false, "portable GDN: nonfinite input/result");
}
void read_into(ggml_tensor *t, std::vector<float> &out, PortableGdnWork *work = nullptr) {
    if (size_t(ggml_nelements(t)) != out.size())
        require(false, "portable GDN: candidate readback size mismatch");
    if (!work) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
        finite(out);
        return;
    }
    {
        portable_detail::ScopedExpertTimer<> timer(true, work->candidate_readback_ms, work->candidate_readback_calls);
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    }
    {
        portable_detail::ScopedExpertTimer<> timer(true, work->candidate_finite_ms, work->candidate_finite_calls);
        finite(out);
    }
}
std::vector<float> read(ggml_tensor *t, PortableGdnWork *work = nullptr) {
    std::vector<float> out(size_t(ggml_nelements(t)));
    read_into(t, out, work);
    return out;
}
void write(ggml_tensor *t, const std::vector<float> &x) {
    require(size_t(ggml_nelements(t)) == x.size(), "portable GDN: input size mismatch");
    ggml_backend_tensor_set(t, x.data(), 0, x.size() * sizeof(float));
}
} // namespace

struct PortableGdnLayer::Impl {
    PortableGdnOptions options;
    portable_detail::Backend cpu{nullptr, ggml_backend_free};
    Context weights_ctx{nullptr, ggml_free}, ctx{nullptr, ggml_free};
    Context state_ctx{nullptr, ggml_free};
    Buffer state_buffer{nullptr, ggml_backend_buffer_free};
    size_t device_state_size = 0;
    bool state_healthy = true;
    Buffer weights_buffer{nullptr, ggml_backend_buffer_free};
    portable_detail::Scheduler scheduler{nullptr, ggml_backend_sched_free};
    std::unique_ptr<PortableFfn> ffn, chunk_ffn;
    bool chunk_active = false;
    size_t chunk_host_bound = 0;
    PortableLayerChunkWork chunk_work;
    // Graphs borrow the original immutable weight tensors; no second weight bank.
    std::map<std::string, ggml_tensor*> weights;
    Context projection_ctx{nullptr, ggml_free};
    portable_detail::Scheduler projection_scheduler{nullptr, ggml_backend_sched_free};
    portable_detail::Scheduler tail_scheduler{nullptr, ggml_backend_sched_free};
    ggml_cgraph *projection_graph = nullptr, *tail_graph = nullptr;
    ggml_tensor *projection_input = nullptr;
    std::array<ggml_tensor*,5> projection_outputs{}, projection_inputs{};
    using AttentionOutputs = std::array<ggml_tensor*,11>;
    AttentionOutputs tail_outputs{};
    size_t projection_tokens = 0, projection_bytes = 0;
    uint64_t projection_calls = 0, projection_columns = 0;
    uint64_t causal_calls = 0, residual_handoffs = 0, causal_commits = 0;
    ggml_tensor* causal_residuals=nullptr;
    AttentionOutputs causal_outputs{};
    bool causal() const { return options.causal_chunk && !options.ple_table; }
    bool batch_projections() const { return options.batch_input_projections && !options.ple_table; }
    AttentionOutputs outputs() const {
        return {attn_residual,state_output,history_output,qkv,conv,q,k,v,g,beta,attention};
    }
    void outputs(const AttentionOutputs& a) {
        attn_residual=a[0];state_output=a[1];history_output=a[2];qkv=a[3];conv=a[4];
        q=a[5];k=a[6];v=a[7];g=a[8];beta=a[9];attention=a[10];
    }
    AttentionOutputs build_tail(ggml_context* c, ggml_tensor* hidden, ggml_tensor* inject,
        ggml_tensor* qkv_input, ggml_tensor* z, ggml_tensor* alpha, ggml_tensor* beta_input,
        ggml_tensor* state, ggml_tensor* history) {
        const auto& w=weights;
        qkv=qkv_input;
        beta=ggml_reshape_4d(c,ggml_sigmoid(c,beta_input),1,HV,1,1);
        g=ggml_reshape_4d(c,ggml_mul(c,ggml_softplus(c,ggml_add(c,alpha,w.at("ssm_dt.bias"))),w.at("ssm_a")),1,HV,1,1);
        auto *window = ggml_concat(c, history, ggml_reshape_3d(c, qkv, 1, CHANNELS, 1), 0);
        history_output = ggml_cont(c, ggml_view_2d(c, window, 3, CHANNELS, 4 * sizeof(float), sizeof(float)));
        conv = ggml_silu(c, ggml_ssm_conv(c, window, w.at("ssm_conv1d.weight")));
        auto *qraw = ggml_view_4d(c, conv, D, HK, 1, 1, D * sizeof(float), CHANNELS * sizeof(float),
                                  CHANNELS * sizeof(float), 0);
        auto *kraw = ggml_view_4d(c, conv, D, HK, 1, 1, D * sizeof(float), CHANNELS * sizeof(float),
                                  CHANNELS * sizeof(float), D * HK * sizeof(float));
        v = ggml_view_4d(c, conv, D, HV, 1, 1, D * sizeof(float), CHANNELS * sizeof(float), CHANNELS * sizeof(float),
                         2 * D * HK * sizeof(float));
        q = ggml_scale(c, ggml_rms_norm(c, qraw, 1e-6f / float(D)), 1.0f / std::sqrt(float(D)));
        k = ggml_scale(c, ggml_rms_norm(c, kraw, 1e-6f / float(D)), 1.0f / std::sqrt(float(D)));
        auto *result = ggml_gated_delta_net(c, q, k, v, g, beta, state, 1);
        attention = ggml_view_2d(c, result, D, HV, D * sizeof(float), 0);
        state_output = ggml_view_1d(c, result, STATES, INNER * sizeof(float));
        auto *normalized = ggml_mul(c, ggml_rms_norm(c, attention, 1e-6f), w.at("ssm_norm.weight"));
        auto *outgate = ggml_reshape_2d(c, ggml_sigmoid(c, z), D, HV);
        auto *projected =
            ggml_mul_mat(c, w.at("ssm_out.weight"), ggml_reshape_1d(c, ggml_mul(c, normalized, outgate), INNER));
        auto *scatter =
            ggml_reshape_3d(c, ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, inject, 0.25f)), 2.0f), 1, HC, 1);
        auto *repeat = ggml_repeat_4d(c, ggml_reshape_3d(c, projected, H, 1, 1), H, HC, 1, 1);
        attn_residual = ggml_add(c, hidden, ggml_mul(c, repeat, scatter));
        return outputs();
    }
    size_t chunk_graphs() const { return (chunk_ffn ? chunk_ffn->graph_bytes() : 0) + projection_bytes; }
    void prepare_projections(size_t T) {
        if (!batch_projections() || projection_tokens == T) return;
        // Drop the old shape before allocating its replacement. The tail and T1
        // schedulers keep their own buffers and are included in graphs_size.
        projection_scheduler.reset(); projection_ctx.reset(); projection_bytes=0; projection_tokens=0;
        projection_graph=nullptr; projection_input=nullptr; projection_outputs={};
        causal_residuals=nullptr; causal_outputs={};
        placement.erase(std::remove_if(placement.begin(),placement.end(),[](const auto& p) {
            return p.phase=="gdn-input-projections" || p.phase=="gdn-causal-chunk";
        }),placement.end());
        if(T==1) return;
        projection_ctx=context(); auto* c=projection_ctx.get();
        projection_input=ggml_new_tensor_3d(c,GGML_TYPE_F32,H,HC,T); ggml_set_input(projection_input);
        auto* norm=ggml_mul(c,ggml_reshape_2d(c,ggml_rms_norm(c,projection_input,1e-6f),H*HC,T),weights.at("hc_attn_norm.weight"));
        // Share three padded F32 banks across all seven quantized input
        // projections. Views exclude the pinned backend's MMVQ conversion;
        // padding and backend packing do not change the input values.
        auto matrix=[&](const char* name,ggml_tensor* columns){
            auto* t=ggml_mul_mat(c,weights.at(name),columns);ggml_set_name(t,name);return t;
        };
        auto* norm_columns=portable_detail::gdn_f32_projection_columns(c,norm);
        auto* low=ggml_silu(c,ggml_scale(c,matrix("hc_attn_down.weight",norm_columns),0.25f));
        auto* low_columns=portable_detail::gdn_f32_projection_columns(c,low);
        auto* gates=ggml_sigmoid(c,matrix("hc_attn_up.weight",low_columns));
        auto* gated=ggml_reshape_3d(c,ggml_mul(c,norm,gates),H,HC,T);
        auto* mixed=ggml_cont(c,ggml_view_2d(c,gated,H,T,H*HC*sizeof(float),0));
        for(int64_t stream=1;stream<HC;++stream)
            mixed=ggml_add(c,mixed,ggml_view_2d(c,gated,H,T,H*HC*sizeof(float),stream*H*sizeof(float)));
        mixed=ggml_scale(c,mixed,0.25f);
        auto* mixed_columns=portable_detail::gdn_f32_projection_columns(c,mixed);
        projection_outputs[0]=matrix("hc_attn_inject.weight",norm_columns);
        const char* names[]={"attn_qkv.weight","attn_gate.weight","ssm_alpha.weight","ssm_beta.weight"};
        projection_graph=ggml_new_graph(c);
        for(size_t i=0;i<4;++i) {
            projection_outputs[i+1]=matrix(names[i],mixed_columns);
        }
        for(auto* t:projection_outputs) {ggml_set_output(t);ggml_build_forward_expand(projection_graph,t);}
        if(causal()) {
            // Each next token consumes its predecessor's candidate tensors.
            // No committed state is overwritten while this graph or FFN runs.
            const auto saved=outputs();
            auto* state=state_input; auto* history=history_input;
            causal_residuals=nullptr;
            for(size_t t=0;t<T;++t) {
                std::array<ggml_tensor*,5> column{};
                for(size_t i=0;i<5;++i) {
                    const int64_t n=projection_outputs[i]->ne[0];
                    column[i]=ggml_view_1d(c,projection_outputs[i],n,t*n*sizeof(float));
                }
                auto* hidden=ggml_view_3d(c,projection_input,H,HC,1,H*sizeof(float),RESIDUAL*sizeof(float),t*RESIDUAL*sizeof(float));
                causal_outputs=build_tail(c,hidden,column[0],column[1],column[2],column[3],column[4],state,history);
                state=ggml_reshape_4d(c,causal_outputs[1],D,D,HV,1);
                history=ggml_reshape_3d(c,causal_outputs[2],3,CHANNELS,1);
                causal_residuals=causal_residuals?ggml_concat(c,causal_residuals,causal_outputs[0],2):causal_outputs[0];
            }
            causal_outputs[1]=state; causal_outputs[2]=history;
            for(auto* t:{causal_residuals,state,history}) {ggml_set_output(t);ggml_build_forward_expand(projection_graph,t);}
            outputs(saved);
        }
        auto opts=options.execution;
        require(graphs_size+chunk_graphs()<opts.graph_budget,"portable GDN: no projection graph budget");
        opts.graph_budget-=graphs_size+(chunk_ffn?chunk_ffn->graph_bytes():0);
        projection_scheduler=portable_detail::schedule(projection_graph,causal()?"gdn-causal-chunk":"gdn-input-projections",opts,cpu,placement,projection_bytes,causal());
        projection_tokens=T;
    }

    void prepare(size_t T,std::string& error) {
        auto& s=*this;
        require(T>=1&&T<=8,"portable layer: chunk size outside [1,8]");
        const size_t extra = (s.production() ? 0 : (STATES+HISTORY+(s.options.ple_table?PLE_HISTORY:0))*sizeof(float)) +
                             (3*T*RESIDUAL+RESIDUAL)*sizeof(float);
        require(s.host_state_bytes() <= s.options.state_budget && extra <= s.options.state_budget-s.host_state_bytes(),
                "portable GDN: chunk working state budget exceeded");
        s.chunk_host_bound = extra + (s.batch_projections() && T>1 ? T*(HC+CHANNELS+INNER+2*HV)*sizeof(float) : 0);
        require(s.chunk_host_bound <= s.options.state_budget-s.host_state_bytes(),
                "portable GDN: projection host scratch budget exceeded");
        if (T == 1 || (s.chunk_ffn && s.chunk_ffn->tokens() != int64_t(T))) s.chunk_ffn.reset();
        if (T > 1 && !s.chunk_ffn) {
            require(s.graphs_size+s.projection_bytes < s.options.execution.graph_budget,"portable GDN: no chunk graph budget");
            s.chunk_ffn = s.ffn->create_chunk_execution(int64_t(T),error,s.options.execution.graph_budget-s.graphs_size-s.projection_bytes);
            require(bool(s.chunk_ffn),error);
        }
        s.prepare_projections(T);
    }
    struct CursorView {
        const std::vector<float>& state;
        const std::vector<float>& history;
        const std::vector<float>& ple_history;
        const std::array<int32_t,2>& previous;
    };
    struct AttentionResult {
        std::vector<float> residual;
        std::array<int32_t,2> previous;
        PortableGdnTrace trace;
    };
    AttentionResult attend(int32_t token, const std::vector<float>& residual,
                           CursorView cursor, bool want_trace, PortableGdnWork* work, bool projected = false) {
        auto& s = *this;
        const auto saved_outputs=projected?s.outputs():AttentionOutputs{};
        struct Restore { Impl& s; AttentionOutputs saved; bool enabled; ~Restore(){if(enabled)s.outputs(saved);} } restore{s,saved_outputs,projected};
        if(projected) s.outputs(s.tail_outputs);
        std::string error;
        finite(residual);
        const bool trace = want_trace;
        PortableGdnTrace next_trace;
        auto next_previous = cursor.previous;
        PortableGdnWork disabled;
        auto &timing = work ? *work : disabled;
        std::vector<float> next_residual;
        {
            portable_detail::ScopedExpertTimer<> attention_timer(work != nullptr,
                timing.attention_exchange_ms, timing.attention_exchange_calls);
            if (s.options.ple_table) {
                std::vector<float> embedding;
                std::array<uint32_t, 16> rows{};
                require(s.options.ple_table->gather(token, cursor.previous, rows, embedding, error), error);
                write(s.ple_emb, embedding);
                if (!s.state_buffer)
                    write(s.ple_hist, cursor.ple_history);
                next_previous = {{cursor.previous[1], token}};
                if (trace) {
                    next_trace.ple_rows = rows;
                    next_trace.ple_embedding = std::move(embedding);
                }
            }
            write(s.input, residual);
            if (!s.state_buffer) {
                write(s.state_input, cursor.state);
                write(s.history_input, cursor.history);
            }
            ggml_status status;
            {
                portable_detail::ScopedExpertTimer<> compute_timer(work != nullptr,
                    timing.attention_compute_ms, timing.attention_compute_calls);
                status = ggml_backend_sched_graph_compute(projected?s.tail_scheduler.get():s.scheduler.get(), projected?s.tail_graph:s.graph);
            }
            require(status == GGML_STATUS_SUCCESS, "portable GDN: attention graph failed");
            if (!s.production()) {
                read_into(s.state_output, s.candidate_state, work);
                read_into(s.history_output, s.candidate_history, work);
            }
            next_residual = read(s.attn_residual, work);
            if (s.options.ple_table && !s.production())
                read_into(s.ple_hist_out, s.candidate_ple_history, work);
            if (trace) {
                next_trace.qkv = read(s.qkv);
                next_trace.convolution = read(s.conv);
                next_trace.query = read(s.q);
                next_trace.key = read(s.k);
                next_trace.value = read(s.v);
                next_trace.decay = read(s.g);
                next_trace.beta = read(s.beta);
                next_trace.attention = read(s.attention);
                if (s.options.ple_table) {
                    next_trace.ple_key = read(s.ple_key);
                    next_trace.ple_value = read(s.ple_value);
                    next_trace.ple_gate = read(s.ple_gate);
                    next_trace.ple_normalized = read(s.ple_norm);
                    next_trace.ple_convolution = read(s.ple_conv);
                    next_trace.ple_residual = read(s.ple_result);
                }
            }
        }
        return {std::move(next_residual),next_previous,std::move(next_trace)};
    }
    void commit_device(PortableGdnWork* work, bool projected = false, bool causal = false) {
        auto& s = *this;
        PortableGdnWork disabled;
        auto& timing = work ? *work : disabled;
        if (s.state_buffer) {
            // Retain separate synchronous copies on this backend API. Diagnostics
            // validate full candidates; production only reads the FFN residual.
            // T1 and the causal graph invoke this after FFN. The legacy chunk
            // advances its working cursor after each serial attention step.
            // A recoverable copy exception poisons device state until reset;
            // public mirrors still describe the last successful token.
            portable_detail::ScopedExpertTimer<> commit_timer(work != nullptr,
                timing.state_commit_ms, timing.state_commit_calls);
            s.state_healthy = false;
            ggml_backend_tensor_copy(causal?s.causal_outputs[1]:(projected?s.tail_outputs[1]:s.state_output), s.state_input);
            ggml_backend_tensor_copy(causal?s.causal_outputs[2]:(projected?s.tail_outputs[2]:s.history_output), s.history_input);
            if (s.options.ple_table)
                ggml_backend_tensor_copy(s.ple_hist_out, s.ple_hist);
            ggml_backend_synchronize(s.options.execution.accelerator);
            s.state_healthy = true;
            if(causal) ++s.causal_commits;
        }
    }

    ggml_cgraph *graph = nullptr;
    ggml_tensor *input = nullptr, *state_input = nullptr, *history_input = nullptr, *attn_residual = nullptr;
    ggml_tensor *state_output = nullptr, *history_output = nullptr, *qkv = nullptr, *conv = nullptr, *q = nullptr,
                *k = nullptr, *v = nullptr, *g = nullptr, *beta = nullptr, *attention = nullptr;
    ggml_tensor *ple_emb = nullptr, *ple_hist = nullptr, *ple_hist_out = nullptr, *ple_key = nullptr,
                *ple_value = nullptr, *ple_gate = nullptr, *ple_norm = nullptr, *ple_conv = nullptr,
                *ple_result = nullptr;
    std::array<int32_t, 2> previous{{-1, -1}};
    std::vector<float> ple_history;
    std::vector<float> state, history, last;
    // Separate host banks, already priced by HOST_STATE_BYTES/PLE_STATE_BYTES.
    // Only a complete successful step swaps these into the public mirrors.
    std::vector<float> candidate_state, candidate_history, candidate_ple_history;
    std::vector<PortablePlacement> placement;
    size_t weights_size = 0, graphs_size = 0, attention_device_weights = 0;
    uint64_t count = 0;

    bool production() const { return options.policy == PortableGdnPolicy::device_production; }
    size_t host_state_bytes() const {
        return HOST_STATE_BYTES - (production() ? (2 * STATES + 2 * HISTORY) * sizeof(float) : 0) +
            (options.ple_table ? PLE_STATE_BYTES - (production() ? 2 * PLE_HISTORY * sizeof(float) : 0) : 0);
    }

    Impl(int64_t layer, const std::vector<PortableTensorView> &tensors, PortableGdnOptions opts) : options(opts) {
        require(layer >= 0 && layer < 48 && layer % 4 != 3,
                "portable GDN: expected a non-QSA layer in [0,47]");
        require(!options.ple_table || layer == 1,
                "portable GDN: this model supports PLE only at layer 1");
        require(!production() || (options.resident_state && options.execution.request_accelerator &&
                    options.execution.accelerator && !options.execution.allow_cpu_fallback),
                "portable GDN: production requires resident state and strict selected backend");
        require(!options.causal_chunk || (options.batch_input_projections && production()),
                "portable GDN: causal chunk requires production and batched input projections");
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        auto expected_shapes = shapes();
        if (options.ple_table)
            expected_shapes.insert(ple_shapes().begin(), ple_shapes().end());
        require(options.state_budget >= host_state_bytes(),
                "portable GDN: host state budget exceeded");
        std::vector<PortableTensorView> fw, aw;
        for (const auto &t : tensors) {
            require(t.name.rfind(prefix, 0) == 0, "portable GDN: tensor belongs to another layer");
            (expected_shapes.count(t.name.substr(prefix.size())) ? aw : fw).push_back(t);
        }
        require(aw.size() == expected_shapes.size() && fw.size() == 9,
                "portable GDN: missing/extra attention, PLE or FFN tensors");
        std::string error;
        ffn = PortableFfn::create({H, HC, R, 640, 512, 10, 1e-6f}, layer, fw, options.execution, error);
        require(bool(ffn), error);
        weights_size = ffn->weight_bytes();
        graphs_size = ffn->graph_bytes();
        placement = ffn->placements();
        cpu.reset(ggml_backend_cpu_init());
        require(bool(cpu), "portable GDN: CPU unavailable");
        ggml_backend_cpu_set_n_threads(cpu.get(), options.execution.threads);
        weights_ctx = context();
        ctx = context();
        std::map<std::string, ggml_tensor *> w;
        for (const auto &view : aw) {
            const auto key = view.name.substr(prefix.size());
            const auto &expected = expected_shapes.at(key);
            const bool flat_norm = key.rfind("ple_norm_", 0) == 0 && view.shape == std::vector<int64_t>{H * HC};
            require(!w.count(key) && (view.shape == expected || flat_norm),
                    "portable GDN: bad/duplicate tensor shape: " + view.name);
            const bool quantized = expected.size() == 2 && key != "ssm_conv1d.weight" &&
                                   (key.rfind("ple_", 0) != 0 || key == "ple_key.weight" || key == "ple_value.weight");
            const bool half_conv = key == "ple_conv1d.weight" && view.type == 1;
            require((view.type == (quantized ? 8u : 0u) || half_conv) && view.data,
                    "portable GDN: expected fleet Q8_0/F32 encoding: " + view.name);
            auto *t = ggml_new_tensor(weights_ctx.get(), ggml_type(view.type), int(expected.size()), expected.data());
            ggml_set_name(t, view.name.c_str());
            require(ggml_nbytes(t) == view.bytes && view.bytes <= options.execution.weight_budget - weights_size,
                    "portable GDN: weight size/budget mismatch");
            weights_size += view.bytes;
            w[key] = t;
        }
        const bool resident = (options.resident_attention_weights || options.execution.resident_dense_weights) &&
                              options.execution.request_accelerator &&
                              options.execution.accelerator;
        auto *weight_backend = resident ? options.execution.accelerator : cpu.get();
        const size_t allocation = ggml_backend_alloc_ctx_tensors_from_buft_size(
            weights_ctx.get(), ggml_backend_get_default_buffer_type(weight_backend));
        require(allocation <= options.execution.weight_budget - ffn->weight_bytes(),
                "portable GDN: aligned weight budget exceeded");
        weights_buffer.reset(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), weight_backend));
        require(bool(weights_buffer), "portable GDN: weight allocation failed");
        ggml_backend_buffer_set_usage(weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto &view : aw)
            ggml_backend_tensor_set(w.at(view.name.substr(prefix.size())), view.data, 0, view.bytes);
        const size_t actual_weights = ggml_backend_buffer_get_size(weights_buffer.get());
        require(actual_weights <= options.execution.weight_budget - ffn->weight_bytes(),
                "portable GDN: actual aligned weight budget exceeded");
        attention_device_weights = resident ? actual_weights : 0;
        weights_size = ffn->weight_bytes() + actual_weights;
        weights=w;
        auto *c = ctx.get();
        input = ggml_new_tensor_3d(c, GGML_TYPE_F32, H, HC, 1);
        const bool keep_state = options.resident_state && options.execution.request_accelerator &&
                                options.execution.accelerator;
        if (keep_state)
            state_ctx = context();
        auto *sc = keep_state ? state_ctx.get() : c;
        state_input = ggml_new_tensor_4d(sc, GGML_TYPE_F32, D, D, HV, 1);
        history_input = ggml_new_tensor_3d(sc, GGML_TYPE_F32, 3, CHANNELS, 1);
        for (auto *t : {input, state_input, history_input})
            ggml_set_input(t);
        auto *hidden = input;
        if (options.ple_table) {
            ple_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, H, 1);
            ple_hist = ggml_new_tensor_2d(sc, GGML_TYPE_F32, 9, RESIDUAL);
            ggml_set_input(ple_emb);
            ggml_set_input(ple_hist);
            auto grouped = [&](ggml_tensor *x, const char *name) {
                return ggml_mul(c, ggml_rms_norm(c, ggml_reshape_3d(c, x, H, HC, 1), 1e-6f), w.at(name));
            };
            ple_key = grouped(ggml_mul_mat(c, w.at("ple_key.weight"), ple_emb), "ple_norm_key.weight");
            ple_value = ggml_mul_mat(c, w.at("ple_value.weight"), ple_emb);
            auto *query = grouped(input, "ple_norm_query.weight");
            auto *score = ggml_scale(c, ggml_sum_rows(c, ggml_mul(c, ple_key, query)), 1.0f / std::sqrt(float(H)));
            auto *mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, score), 1e-6f, INFINITY));
            ple_gate = ggml_sigmoid(c, ggml_mul(c, ggml_sgn(c, score), mag));
            auto *value = ggml_repeat_4d(c, ggml_reshape_3d(c, ple_value, H, 1, 1), H, HC, 1, 1);
            auto *gated_value = ggml_mul(c, value, ple_gate);
            ple_norm = grouped(gated_value, "ple_norm_conv.weight");
            auto *padded = ggml_concat(c, ple_hist, ggml_reshape_2d(c, ple_norm, 1, RESIDUAL), 0);
            ple_hist_out = ggml_cont(c, ggml_view_2d(c, padded, 9, RESIDUAL, 10 * sizeof(float), sizeof(float)));
            ggml_tensor *sum = nullptr;
            for (int tap = 0; tap < 4; ++tap) {
                auto *shifted = ggml_reshape_1d(c,
                                                ggml_cont(c, ggml_view_2d(c, padded, 1, RESIDUAL, 10 * sizeof(float),
                                                                          size_t(tap * 3) * sizeof(float))),
                                                RESIDUAL);
                auto *weight =
                    ggml_reshape_1d(c,
                                    ggml_cont(c, ggml_view_2d(c, w.at("ple_conv1d.weight"), 1, RESIDUAL,
                                                              w.at("ple_conv1d.weight")->nb[1],
                                                              size_t(tap) * w.at("ple_conv1d.weight")->nb[0])),
                                    RESIDUAL);
                if (weight->type != GGML_TYPE_F32)
                    weight = ggml_cast(c, weight, GGML_TYPE_F32);
                auto *term = ggml_mul(c, shifted, weight);
                sum = sum ? ggml_add(c, sum, term) : term;
            }
            ple_conv = ggml_reshape_3d(c, ggml_silu(c, sum), H, HC, 1);
            ple_result = ggml_add(c, input, ggml_add(c, gated_value, ple_conv));
            ggml_set_name(ple_result, "ple_residual");
            hidden = ple_result;
        }
        auto build = [&](bool external) {
            auto *norm =
                ggml_mul(c, ggml_reshape_2d(c, ggml_rms_norm(c, hidden, 1e-6f), H * HC, 1), w.at("hc_attn_norm.weight"));
            auto *low = ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, w.at("hc_attn_down.weight"), norm), 0.25f));
            auto *gates = ggml_sigmoid(c, ggml_mul_mat(c, w.at("hc_attn_up.weight"), low));
            auto *gated = ggml_reshape_2d(c, ggml_mul(c, norm, gates), H, HC);
            auto *mixed = ggml_cont(c, ggml_view_1d(c, gated, H, 0));
            for (int64_t s = 1; s < HC; ++s)
                mixed = ggml_add(c, mixed, ggml_view_1d(c, gated, H, size_t(s * H) * sizeof(float)));
            mixed = ggml_scale(c, mixed, 0.25f);
            auto *inject = ggml_mul_mat(c, w.at("hc_attn_inject.weight"), norm);
            auto leaf=[&](int64_t n) {auto* t=ggml_new_tensor_1d(c,GGML_TYPE_F32,n);ggml_set_input(t);return t;};
            if(external) for(size_t i=0;i<5;++i) projection_inputs[i]=leaf(std::array<int64_t,5>{HC,CHANNELS,INNER,HV,HV}[i]);
            if(external) inject=projection_inputs[0];
            qkv = external ? projection_inputs[1] : ggml_mul_mat(c, w.at("attn_qkv.weight"), mixed);
            auto *z = external ? projection_inputs[2] : ggml_mul_mat(c, w.at("attn_gate.weight"), mixed);
            auto* beta_input=external ? projection_inputs[4] : ggml_mul_mat(c,w.at("ssm_beta.weight"),mixed);
            auto* alpha=external ? projection_inputs[3] : ggml_mul_mat(c,w.at("ssm_alpha.weight"),mixed);
            outputs(build_tail(c,hidden,inject,qkv,z,alpha,beta_input,state_input,history_input));
            graph = ggml_new_graph(c);
            for (auto *t : {attn_residual, state_output, history_output, qkv, conv, q, k, v, g, beta, attention}) {
                ggml_set_output(t);
                ggml_build_forward_expand(graph, t);
            }
            if (options.ple_table) {
                for (auto *t : {ple_hist_out, ple_key, ple_value, ple_gate, ple_norm, ple_conv, ple_result}) {
                    ggml_set_output(t);
                    ggml_build_forward_expand(graph, t);
                }
                if (!production()) {
                    ple_history.assign(PLE_HISTORY, 0);
                    candidate_ple_history.resize(PLE_HISTORY);
                }
            }
            if (keep_state && !external) {
                // Native session ownership: one committed state allocation per layer.
                // Candidate graph outputs remain separate until attention AND FFN pass.
                auto *backend = options.execution.accelerator;
                const size_t bound = ggml_backend_alloc_ctx_tensors_from_buft_size(
                    state_ctx.get(), ggml_backend_get_default_buffer_type(backend));
                require(bound <= options.resident_state_budget, "portable GDN: resident state budget exceeded");
                state_buffer.reset(ggml_backend_alloc_ctx_tensors(state_ctx.get(), backend));
                require(bool(state_buffer), "portable GDN: resident state allocation failed");
                device_state_size = ggml_backend_buffer_get_size(state_buffer.get());
                require(device_state_size <= options.resident_state_budget,
                        "portable GDN: actual resident state budget exceeded");
                ggml_backend_buffer_clear(state_buffer.get(), 0);
                ggml_backend_synchronize(backend);
                // The copy API requires identical layouts. These are metadata-only
                // views of the unchanged candidate bytes, allocated before scheduling.
                state_output = ggml_reshape_4d(c, state_output, D, D, HV, 1);
                history_output = ggml_reshape_3d(c, history_output, 3, CHANNELS, 1);
                for (auto *t : {state_output, history_output}) {
                    ggml_set_output(t);
                    ggml_build_forward_expand(graph, t);
                }
            }
            if(external && keep_state) {
                state_output=ggml_reshape_4d(c,state_output,D,D,HV,1);
                history_output=ggml_reshape_3d(c,history_output,3,CHANNELS,1);
                for(auto* t:{state_output,history_output}) {ggml_set_output(t);ggml_build_forward_expand(graph,t);}
            }
            auto sched=portable_detail::schedule(graph,external?"gdn-chunk-recurrence":"gdn-attention",options.execution,cpu,placement,graphs_size);
            if(external) { tail_scheduler=std::move(sched); tail_graph=graph; tail_outputs=outputs(); }
            else scheduler=std::move(sched);
        };
        build(false);
        if(batch_projections() && !causal()) {
            const auto saved=outputs(); auto* saved_graph=graph;
            build(true); outputs(saved); graph=saved_graph;
        }
        if (!production()) {
            state.assign(STATES, 0);
            history.assign(HISTORY, 0);
            candidate_state.resize(STATES);
            candidate_history.resize(HISTORY);
        }
    }
};

PortableGdnLayer::PortableGdnLayer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PortableGdnLayer::~PortableGdnLayer() = default;
std::unique_ptr<PortableGdnLayer> PortableGdnLayer::create(int64_t layer,
                                                           const std::vector<PortableTensorView> &tensors,
                                                           PortableGdnOptions options, std::string &error) {
    error.clear();
    try {
        return std::unique_ptr<PortableGdnLayer>(new PortableGdnLayer(std::make_unique<Impl>(layer, tensors, options)));
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
std::unique_ptr<PortableGdnLayer> PortableGdnLayer::from_gguf(const GgufModel &model, int64_t layer,
                                                              PortableGdnOptions options, std::string &error) {
    error.clear();
    try {
        require(layer >= 0 && layer < 48 && layer % 4 != 3,
                "portable GDN: expected a non-QSA layer in [0,47]");
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto *arch = model.meta().get("general.architecture");
        require(arch && arch->s == "qwen4exp", "portable GDN: requires general.architecture=qwen4exp");
        const auto geometry_error = check_architecture(model.meta());
        require(geometry_error.empty(), "portable GDN: " + geometry_error);
        const auto *ple = model.meta().get("qwen4exp.ple.layers");
        bool configured_ple = false;
        if (ple) {
            require(ple->type == MetaType::ARRAY && ple->items.size() <= 1,
                    "portable GDN: malformed/unsupported PLE layer list");
            for (const auto &v : ple->items) {
                require(v.is_num() && v.u == 1, "portable GDN: supports PLE only at layer 1");
                configured_ple = layer == 1;
            }
        }
        require(!options.ple_table || configured_ple,
                "portable GDN: PLE table supplied for a layer without configured PLE");
        if (configured_ple) {
            options.ple_table = PortablePleTable::from_gguf(model, error);
            require(bool(options.ple_table), error);
        }
        for (auto pair : {std::pair<const char *, uint64_t>{"embedding_length", H},
                          {"block_count", 48},
                          {"ssm.state_size", D},
                          {"ssm.group_count", HK},
                          {"ssm.time_step_rank", HV},
                          {"ssm.inner_size", INNER},
                          {"ssm.conv_kernel", 4},
                          {"hyper_connection.count", HC},
                          {"hyper_connection.low_rank", R},
                          {"expert_feed_forward_length", 640}}) {
            const auto *value = model.meta().get("qwen4exp." + std::string(pair.first));
            require(value && value->is_num() && value->u == pair.second,
                    "portable GDN: missing/mismatched " + std::string(pair.first));
        }
        const auto *experts = model.meta().get("qwen4exp.expert_count");
        const auto *used = model.meta().get("qwen4exp.expert_used_count");
        require(experts && experts->u == 512 && used && used->u == 10, "portable GDN: expected 512 experts/top10");
        std::vector<PortableTensorView> views;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto &t : model.shard(s).tensors()) {
                if (t.name.rfind(prefix, 0) != 0)
                    continue;
                const auto suffix = t.name.substr(prefix.size());
                if (!shapes().count(suffix) && suffix.rfind("ple_", 0) != 0 &&
                    suffix.rfind("hc_ffn_", 0) != 0 &&
                    (suffix.rfind("ffn_", 0) != 0 || suffix.find("_exps") != std::string::npos))
                    continue;
                require(model.in_bounds(t, s), "portable GDN: tensor outside shard");
                PortableTensorView v{
                    t.name, {}, t.type, model.shard(s).tensor_data(t), size_t(tensor_payload_bytes(t))};
                for (auto d : t.shape) {
                    require(d <= INT64_MAX, "portable GDN: dimension overflow");
                    v.shape.push_back(int64_t(d));
                }
                views.push_back(std::move(v));
            }
        return create(layer, views, options, error);
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
bool PortableGdnLayer::reset(std::string &error) {
    error.clear();
    try {
        impl_->state_healthy = false;
        require(!impl_->chunk_active, "portable GDN: chunk execution active");
        std::vector<PortablePlacement> next_placement;
        if(impl_->chunk_host_bound) {
            next_placement=impl_->placement;
            next_placement.erase(std::remove_if(next_placement.begin(),next_placement.end(),[](const auto& p){
                return p.phase != "gdn-attention" && p.phase != "gdn-chunk-recurrence" && p.phase != "gdn-input-projections" && p.phase != "gdn-causal-chunk";
            }),next_placement.end());
            next_placement.insert(next_placement.begin(),impl_->ffn->placements().begin(),impl_->ffn->placements().end());
        }
        impl_->chunk_ffn.reset(); impl_->chunk_host_bound = 0;
        std::vector<float> zero(RESIDUAL, 0);
        require(impl_->ffn->reset(zero, error), error);
        if (impl_->state_buffer) {
            impl_->state_healthy = false;
            ggml_backend_buffer_clear(impl_->state_buffer.get(), 0);
            ggml_backend_synchronize(impl_->options.execution.accelerator);
            impl_->state_healthy = true;
        }
        // Private candidates may contain a rejected step. Keep their capacity:
        // every successful step fully reads and validates them before swapping.
        std::fill(impl_->state.begin(), impl_->state.end(), 0);
        std::fill(impl_->history.begin(), impl_->history.end(), 0);
        std::fill(impl_->ple_history.begin(), impl_->ple_history.end(), 0);
        impl_->previous = {{-1, -1}};
        impl_->last.clear();
        impl_->count = 0;
        if(!next_placement.empty()) impl_->placement.swap(next_placement);
        impl_->state_healthy = true;
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}
bool PortableGdnLayer::step(const std::vector<float> &residual, const PortableExpertRows &dispatch,
                            PortableRoute &route, std::string &error, PortableGdnTrace *trace, PortableGdnWork *work) {
    return step(-1, residual, dispatch, route, error, trace, work);
}
bool PortableGdnLayer::step(int32_t token, const std::vector<float> &residual, const PortableExpertRows &dispatch,
                            PortableRoute &route, std::string &error, PortableGdnTrace *trace, PortableGdnWork *work) {
    error.clear();
    route = {};
    if (trace)
        *trace = {};
    try {
        auto &s = *impl_;
        require(!s.chunk_active, "portable GDN: chunk execution active");
        require(s.state_healthy, "portable GDN: resident state poisoned; reset required");
        require(!s.production() || !trace, "portable GDN: trace requires eager diagnostic policy");
        require(residual.size() == RESIDUAL && bool(dispatch), "portable GDN: residual/callback required");
        require(s.count < UINT64_MAX, "portable GDN: token counter exhausted");
        auto attended = s.attend(token, residual, {s.state,s.history,s.ple_history,s.previous}, trace != nullptr, work);
        auto next_residual = std::move(attended.residual);
        auto next_previous = attended.previous;
        auto next_trace = std::move(attended.trace);
        PortableGdnWork disabled;
        auto &timing = work ? *work : disabled;
        PortableRoute candidate;
        {
            portable_detail::ScopedExpertTimer<> ffn_timer(work != nullptr, timing.ffn_ms, timing.ffn_calls);
            require(s.ffn->reset(next_residual, error), error);
            const bool ok = s.ffn->step(dispatch, candidate, error);
            require(ok, error);
            next_residual = s.ffn->residual(); // allocate before the no-throw public commit
        }
        std::vector<PortablePlacement> next_placement;
        if(s.chunk_host_bound) {
            next_placement=s.placement;
            next_placement.erase(std::remove_if(next_placement.begin(),next_placement.end(),[](const auto& p){
                return p.phase != "gdn-attention" && p.phase != "gdn-chunk-recurrence" && p.phase != "gdn-input-projections" && p.phase != "gdn-causal-chunk";
            }),next_placement.end());
            next_placement.insert(next_placement.begin(),s.ffn->placements().begin(),s.ffn->placements().end());
        }
        s.commit_device(work);
        if (!s.production()) {
            s.ple_history.swap(s.candidate_ple_history);
            s.state.swap(s.candidate_state);
            s.history.swap(s.candidate_history);
        }
        s.previous = next_previous;
        s.last.swap(next_residual);
        ++s.count;
        if(!next_placement.empty()) s.placement.swap(next_placement);
        route = std::move(candidate);
        if (trace)
            *trace = std::move(next_trace);
        return true;
    } catch (const std::exception &e) {
        if (impl_->production())
            impl_->state_healthy = false;
        error = e.what();
        return false;
    } catch (...) {
        if (impl_->production())
            impl_->state_healthy = false;
        error = "portable GDN: unknown execution failure; reset required";
        return false;
    }
}
bool PortableGdnLayer::prepare_chunk(size_t tokens,std::string& error) {
    error.clear();
    try { require(!impl_->chunk_active,"portable layer: active chunk"); impl_->prepare(tokens,error); return true; }
    catch(const std::exception& e) { error=e.what(); return false; }
}
bool PortableGdnLayer::run_chunk(const std::vector<int32_t>& tokens, uint64_t position0,
    std::vector<float>& residual_rows, const PortableExpertRowsBatch& dispatch,
    std::vector<PortableRoute>& routes, std::string& error) {
    routes.clear(); error.clear();
    auto& s = *impl_;
    try {
        require(s.state_healthy && !s.chunk_active, "portable GDN: poisoned or active chunk; reset required");
        const size_t T = tokens.size();
        require(T >= 1 && T <= 8 && residual_rows.size() == T*RESIDUAL && bool(dispatch),
                "portable GDN: invalid chunk shape/callback");
        require(position0 == s.count && T <= UINT64_MAX-s.count && T <= UINT64_MAX-s.chunk_work.tokens,
                "portable GDN: noncontiguous/exhausted chunk");
        struct Active { bool& flag; explicit Active(bool& f):flag(f){flag=true;} ~Active(){flag=false;} } active(s.chunk_active);
        s.prepare(T,error);
        auto state = s.state, history = s.history, ple_history = s.ple_history;
        auto previous = s.previous;
        std::vector<float> attended_rows; attended_rows.reserve(T*RESIDUAL);
        const bool projected=s.batch_projections() && T>1;
        const bool causal=s.causal() && T>1;
        require(!causal || (s.causal_calls<UINT64_MAX && s.residual_handoffs<UINT64_MAX && s.causal_commits<UINT64_MAX),
                "portable GDN: exhausted causal chunk counters");
        std::array<std::vector<float>,5> projections;
        if(projected) {
            require(s.projection_calls<UINT64_MAX && T<=UINT64_MAX-s.projection_columns,
                    "portable GDN: exhausted input projection counters");
            finite(residual_rows); write(s.projection_input,residual_rows);
            require(ggml_backend_sched_graph_compute(s.projection_scheduler.get(),s.projection_graph)==GGML_STATUS_SUCCESS,
                    "portable GDN: input projection graph failed");
            ++s.projection_calls; s.projection_columns+=T;
            if(causal) ++s.causal_calls;
            for(size_t i=0;i<5;++i) projections[i]=read(s.projection_outputs[i]);
        }
        if(causal) {
            // Keep the five bulk reads for the existing finite checks. Their
            // device tensors feed recurrence directly; never upload host slices.
            attended_rows=read(s.causal_residuals);
            ++s.residual_handoffs;
        } else for(size_t t=0;t<T;++t) {
            if(projected) for(size_t i=0;i<5;++i) {
                const size_t n=size_t(ggml_nelements(s.projection_inputs[i]));
                ggml_backend_tensor_set(s.projection_inputs[i],projections[i].data()+t*n,0,n*sizeof(float));
            }
            std::vector<float> input(residual_rows.begin()+t*RESIDUAL,residual_rows.begin()+(t+1)*RESIDUAL);
            auto next = s.attend(tokens[t],input,{state,history,ple_history,previous},false,nullptr,projected);
            // The recurrence/conv/PLE cursor advances now, before the next
            // attention. Public diagnostic banks remain unchanged until FFN.
            s.commit_device(nullptr,projected);
            if(!s.production()) {
                state.swap(s.candidate_state); history.swap(s.candidate_history);
                ple_history.swap(s.candidate_ple_history);
            }
            previous = next.previous;
            attended_rows.insert(attended_rows.end(),next.residual.begin(),next.residual.end());
        }
        auto& ffn = T == 1 ? *s.ffn : *s.chunk_ffn;
        const auto before = ffn.work();
        require(ffn.reset(attended_rows,error),error);
        std::vector<PortableRoute> candidate;
        if(T == 1) {
            PortableRoute route;
            require(ffn.step([&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e) {
                return dispatch(l,{r},rows,e);
            },route,error),error);
            candidate.push_back(std::move(route));
        } else require(ffn.step_batch(dispatch,candidate,error),error);
        auto completed = ffn.residual();
        std::vector<float> last(completed.end()-RESIDUAL,completed.end());
        auto placement = s.placement;
        placement.erase(std::remove_if(placement.begin(),placement.end(),[](const auto& p){
            return p.phase != "gdn-attention" && p.phase != "gdn-chunk-recurrence" && p.phase != "gdn-input-projections" && p.phase != "gdn-causal-chunk";
        }),placement.end());
        placement.insert(placement.begin(),ffn.placements().begin(),ffn.placements().end());
        const auto after = ffn.work();
        auto work = s.chunk_work;
        ++work.chunks; work.tokens += T; work.attention_calls += T;
        work.ffn_pre_calls += after.pre_graph_calls-before.pre_graph_calls;
        work.provider_calls += after.provider_calls-before.provider_calls;
        work.ffn_post_calls += after.post_graph_calls-before.post_graph_calls;
        require(s.state_healthy,"portable GDN: chunk invalidated; reset required");
        if(causal) s.commit_device(nullptr,false,true);
        if(!s.production()) { s.state.swap(state); s.history.swap(history); s.ple_history.swap(ple_history); }
        s.previous = previous; s.last.swap(last); s.count += T;
        s.placement.swap(placement); s.chunk_work = work;
        residual_rows.swap(completed); routes.swap(candidate);
        return true;
    } catch(const std::exception& e) { s.state_healthy=false; error=e.what(); return false; }
      catch(...) { s.state_healthy=false; error="portable GDN: unknown chunk failure; reset required"; return false; }
}
uint64_t PortableGdnLayer::causal_chunk_calls() const { return impl_->causal_calls; }
uint64_t PortableGdnLayer::causal_residual_handoffs() const { return impl_->residual_handoffs; }
uint64_t PortableGdnLayer::causal_state_commits() const { return impl_->causal_commits; }
uint64_t PortableGdnLayer::input_projection_calls() const { return impl_->projection_calls; }
uint64_t PortableGdnLayer::input_projection_columns() const { return impl_->projection_columns; }
PortableLayerChunkWork PortableGdnLayer::chunk_work() const { return impl_->chunk_work; }
bool PortableGdnLayer::poisoned() const { return !impl_->state_healthy; }
PortableGdnPolicy PortableGdnLayer::policy() const { return impl_->options.policy; }
const std::vector<float> &PortableGdnLayer::ple_convolution_history() const {
    require(!impl_->production(), "portable GDN: state inspection requires eager diagnostic policy");
    return impl_->ple_history;
}
const std::array<int32_t, 2> &PortableGdnLayer::token_history() const { return impl_->previous; }
const std::vector<float> &PortableGdnLayer::recurrent_state() const {
    require(!impl_->production(), "portable GDN: state inspection requires eager diagnostic policy");
    return impl_->state;
}
const std::vector<float> &PortableGdnLayer::convolution_history() const {
    require(!impl_->production(), "portable GDN: state inspection requires eager diagnostic policy");
    return impl_->history;
}
const std::vector<float> &PortableGdnLayer::last_residual() const { return impl_->last; }
uint64_t PortableGdnLayer::tokens() const { return impl_->count; }
const std::vector<PortablePlacement> &PortableGdnLayer::placements() const { return impl_->placement; }
size_t PortableGdnLayer::weight_bytes() const { return impl_->weights_size; }
size_t PortableGdnLayer::graph_bytes() const { return impl_->graphs_size+impl_->chunk_graphs(); }
size_t PortableGdnLayer::state_bytes() const { return impl_->host_state_bytes()+impl_->chunk_host_bound; }
PortableMemoryUsage PortableGdnLayer::memory_usage() const {
    const auto &s = *impl_;
    auto m = s.ffn->memory_usage();
    if(s.chunk_ffn) { std::string error; require(s.ffn->combined_memory_usage(*s.chunk_ffn,m,error),error); }
    m.device_state += s.device_state_size;
    // Extra metadata context plus the copy API's bounded host staging fallback.
    if (s.state_buffer)
        m.host_extra_bound += (4u << 20) + STATES * sizeof(float);
    m.device_weights += s.attention_device_weights;
    m.host_weights = s.weights_size - m.device_weights;
    m.host_graph += ggml_backend_sched_get_buffer_size(s.scheduler.get(), s.cpu.get());
    if (s.options.execution.accelerator && s.options.execution.request_accelerator)
        m.device_graph += ggml_backend_sched_get_buffer_size(s.scheduler.get(), s.options.execution.accelerator);
    for(auto* sched:{s.tail_scheduler.get(),s.projection_scheduler.get()}) if(sched) {
        m.host_graph+=ggml_backend_sched_get_buffer_size(sched,s.cpu.get());
        if(s.options.execution.accelerator && s.options.execution.request_accelerator)
            m.device_graph+=ggml_backend_sched_get_buffer_size(sched,s.options.execution.accelerator);
    }
    if(s.tail_graph) m.host_extra_bound+=ggml_graph_plan(s.tail_graph,s.options.execution.threads,nullptr).work_size;
    if(s.projection_ctx) m.host_extra_bound+=(4u<<20)+ggml_graph_plan(s.projection_graph,s.options.execution.threads,nullptr).work_size;
    const size_t buffers = m.host_graph + m.device_graph;
    const size_t allowance = graph_bytes() > buffers ? graph_bytes() - buffers : 0;
    m.host_extra_bound += state_bytes() + (8u << 20) + allowance +
                          ggml_graph_plan(s.graph, s.options.execution.threads, nullptr).work_size;
    if (s.options.execution.request_accelerator) {
        m.device_extra_bound += allowance + (8u << 20);
        portable_detail::account_matrix_conversion(m,s.options.execution.accelerator,size_t(H*CHANNELS)*sizeof(float));
        // Strided F32 inputs use the backend's distinct persistent prealloc_y
        // packing bank. Price the largest permitted shape even after replacing
        // it with a smaller graph; backend scratch retains its high-water size.
        // Padded graph tensors are already included in scheduler graph bytes.
        if(s.batch_projections()) m.device_extra_bound+=8*RESIDUAL*sizeof(float);
    }
    return m;
}
} // namespace strata::core
