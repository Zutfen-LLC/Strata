#include "strata/core/portable_gdn.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "portable_graph.hpp"
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
    for (float f : x)
        require(std::isfinite(f), "portable GDN: nonfinite input/result");
}
std::vector<float> read(ggml_tensor *t) {
    std::vector<float> out(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    finite(out);
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
    Buffer weights_buffer{nullptr, ggml_backend_buffer_free};
    portable_detail::Scheduler scheduler{nullptr, ggml_backend_sched_free};
    std::unique_ptr<PortableFfn> ffn;
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
    std::vector<PortablePlacement> placement;
    size_t weights_size = 0, graphs_size = 0;
    uint64_t count = 0;

    Impl(int64_t layer, const std::vector<PortableTensorView> &tensors, PortableGdnOptions opts) : options(opts) {
        require(layer >= 0 && layer < 48 && layer % 4 != 3,
                "portable GDN: expected a non-QSA layer in [0,47]");
        require(!options.ple_table || layer == 1,
                "portable GDN: this model supports PLE only at layer 1");
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        auto expected_shapes = shapes();
        if (options.ple_table)
            expected_shapes.insert(ple_shapes().begin(), ple_shapes().end());
        require(options.state_budget >= HOST_STATE_BYTES + (options.ple_table ? PLE_STATE_BYTES : 0),
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
        const size_t allocation = ggml_backend_alloc_ctx_tensors_from_buft_size(
            weights_ctx.get(), ggml_backend_get_default_buffer_type(cpu.get()));
        require(allocation <= options.execution.weight_budget - ffn->weight_bytes(),
                "portable GDN: aligned weight budget exceeded");
        weights_buffer.reset(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), cpu.get()));
        require(bool(weights_buffer), "portable GDN: weight allocation failed");
        ggml_backend_buffer_set_usage(weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto &view : aw)
            ggml_backend_tensor_set(w.at(view.name.substr(prefix.size())), view.data, 0, view.bytes);
        weights_size = ffn->weight_bytes() + ggml_backend_buffer_get_size(weights_buffer.get());
        auto *c = ctx.get();
        input = ggml_new_tensor_3d(c, GGML_TYPE_F32, H, HC, 1);
        state_input = ggml_new_tensor_4d(c, GGML_TYPE_F32, D, D, HV, 1);
        history_input = ggml_new_tensor_3d(c, GGML_TYPE_F32, 3, CHANNELS, 1);
        for (auto *t : {input, state_input, history_input})
            ggml_set_input(t);
        auto *hidden = input;
        if (options.ple_table) {
            ple_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, H, 1);
            ple_hist = ggml_new_tensor_2d(c, GGML_TYPE_F32, 9, RESIDUAL);
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
            hidden = ple_result;
        }
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
        qkv = ggml_mul_mat(c, w.at("attn_qkv.weight"), mixed);
        auto *z = ggml_mul_mat(c, w.at("attn_gate.weight"), mixed);
        beta = ggml_reshape_4d(c, ggml_sigmoid(c, ggml_mul_mat(c, w.at("ssm_beta.weight"), mixed)), 1, HV, 1, 1);
        auto *alpha = ggml_mul_mat(c, w.at("ssm_alpha.weight"), mixed);
        g = ggml_reshape_4d(c, ggml_mul(c, ggml_softplus(c, ggml_add(c, alpha, w.at("ssm_dt.bias"))), w.at("ssm_a")), 1,
                            HV, 1, 1);
        auto *window = ggml_concat(c, history_input, ggml_reshape_3d(c, qkv, 1, CHANNELS, 1), 0);
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
        auto *result = ggml_gated_delta_net(c, q, k, v, g, beta, state_input, 1);
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
            ple_history.assign(PLE_HISTORY, 0);
        }
        scheduler = portable_detail::schedule(graph, "gdn-attention", options.execution, cpu, placement, graphs_size);
        state.assign(STATES, 0);
        history.assign(HISTORY, 0);
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
        std::vector<float> zero(RESIDUAL, 0);
        require(impl_->ffn->reset(zero, error), error);
        std::fill(impl_->state.begin(), impl_->state.end(), 0);
        std::fill(impl_->history.begin(), impl_->history.end(), 0);
        std::fill(impl_->ple_history.begin(), impl_->ple_history.end(), 0);
        impl_->previous = {{-1, -1}};
        impl_->last.clear();
        impl_->count = 0;
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}
bool PortableGdnLayer::step(const std::vector<float> &residual, const PortableExpertRows &dispatch,
                            PortableRoute &route, std::string &error, PortableGdnTrace *trace) {
    return step(-1, residual, dispatch, route, error, trace);
}
bool PortableGdnLayer::step(int32_t token, const std::vector<float> &residual, const PortableExpertRows &dispatch,
                            PortableRoute &route, std::string &error, PortableGdnTrace *trace) {
    error.clear();
    route = {};
    if (trace)
        *trace = {};
    try {
        auto &s = *impl_;
        require(residual.size() == RESIDUAL && bool(dispatch), "portable GDN: residual/callback required");
        require(s.count < UINT64_MAX, "portable GDN: token counter exhausted");
        finite(residual);
        PortableGdnTrace next_trace;
        auto next_previous = s.previous;
        if (s.options.ple_table) {
            std::vector<float> embedding;
            std::array<uint32_t, 16> rows{};
            require(s.options.ple_table->gather(token, s.previous, rows, embedding, error), error);
            write(s.ple_emb, embedding);
            write(s.ple_hist, s.ple_history);
            next_previous = {{s.previous[1], token}};
            if (trace) {
                next_trace.ple_rows = rows;
                next_trace.ple_embedding = std::move(embedding);
            }
        }
        write(s.input, residual);
        write(s.state_input, s.state);
        write(s.history_input, s.history);
        require(ggml_backend_sched_graph_compute(s.scheduler.get(), s.graph) == GGML_STATUS_SUCCESS,
                "portable GDN: attention graph failed");
        auto next_state = read(s.state_output), next_history = read(s.history_output),
             next_residual = read(s.attn_residual);
        auto next_ple_history = s.options.ple_table ? read(s.ple_hist_out) : std::vector<float>{};
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
        require(s.ffn->reset(next_residual, error), error);
        PortableRoute candidate;
        const bool ok = s.ffn->step(dispatch, candidate, error);
        require(ok, error);
        next_residual = s.ffn->residual(); // allocate before the no-throw public commit
        s.ple_history.swap(next_ple_history);
        s.previous = next_previous;
        s.state.swap(next_state);
        s.history.swap(next_history);
        s.last.swap(next_residual);
        ++s.count;
        route = std::move(candidate);
        if (trace)
            *trace = std::move(next_trace);
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}
const std::vector<float> &PortableGdnLayer::ple_convolution_history() const { return impl_->ple_history; }
const std::array<int32_t, 2> &PortableGdnLayer::token_history() const { return impl_->previous; }
const std::vector<float> &PortableGdnLayer::recurrent_state() const { return impl_->state; }
const std::vector<float> &PortableGdnLayer::convolution_history() const { return impl_->history; }
const std::vector<float> &PortableGdnLayer::last_residual() const { return impl_->last; }
uint64_t PortableGdnLayer::tokens() const { return impl_->count; }
const std::vector<PortablePlacement> &PortableGdnLayer::placements() const { return impl_->placement; }
size_t PortableGdnLayer::weight_bytes() const { return impl_->weights_size; }
size_t PortableGdnLayer::graph_bytes() const { return impl_->graphs_size; }
size_t PortableGdnLayer::state_bytes() const { return HOST_STATE_BYTES + (impl_->options.ple_table ? PLE_STATE_BYTES : 0); }
} // namespace strata::core
