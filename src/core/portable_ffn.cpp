#include "strata/core/portable_ffn.hpp"
#include "portable_graph.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

namespace strata::core {
namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void finite(const std::vector<float>& x) {
    for (float v : x) require(std::isfinite(v), "portable FFN: nonfinite activation/result");
}
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
Context context() {
    ggml_init_params params{2u << 20, nullptr, true};
    Context result(ggml_init(params), ggml_free);
    require(bool(result), "portable FFN: metadata allocation failed");
    return result;
}
ggml_tensor* named(ggml_tensor* t, const char* name) { ggml_set_name(t, name); return t; }
std::vector<float> read(ggml_tensor* t) {
    std::vector<float> out(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    finite(out);
    return out;
}
void write(ggml_tensor* t, const std::vector<float>& data) {
    require(size_t(ggml_nelements(t)) == data.size(), "portable FFN: input size mismatch");
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

// Same softmax -> stable top-k -> selected-weight renormalization contract as
// router_top10.cu. Keep full softmax and its 2^-14 clamp, not a subset softmax.
void route_logits(const std::vector<float>& logits, int64_t k, PortableRoute& route) {
    const float max = *std::max_element(logits.begin(), logits.end());
    std::vector<double> exp(logits.size());
    double sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) sum += exp[i] = std::exp(double(logits[i]) - double(max));
    const float inv = float(1.0 / sum);
    std::vector<float> probabilities(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) probabilities[i] = float(exp[i] * inv);
    std::vector<int32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) {
        return probabilities[a] > probabilities[b];
    });
    route.ids.assign(ids.begin(), ids.begin() + k);
    route.weights.resize(size_t(k));
    float selected = 0;
    for (int64_t i = 0; i < k; ++i) selected += probabilities[ids[size_t(i)]];
    selected = std::max(selected, 0x1p-14f);
    for (int64_t i = 0; i < k; ++i) route.weights[size_t(i)] = probabilities[ids[size_t(i)]] / selected;
}
} // namespace

struct PortableFfn::Impl {
    PortableFfnGeometry g;
    int64_t layer;
    PortableFfnOptions options;
    Backend cpu{nullptr, ggml_backend_free};
    Context weights_ctx{nullptr, ggml_free}, pre_ctx{nullptr, ggml_free}, post_ctx{nullptr, ggml_free};
    Buffer weights_buffer{nullptr, ggml_backend_buffer_free};
    Scheduler pre_sched{nullptr, ggml_backend_sched_free}, post_sched{nullptr, ggml_backend_sched_free};
    ggml_cgraph *pre = nullptr, *post = nullptr;
    ggml_tensor *input = nullptr, *mixed = nullptr, *inject = nullptr, *logits = nullptr, *shared = nullptr;
    ggml_tensor *post_input = nullptr, *post_inject = nullptr, *post_block = nullptr, *output = nullptr;
    std::vector<float> state;
    std::vector<PortablePlacement> placement;
    size_t weights_size = 0, graphs_size = 0;
    bool initialized = false;


    Impl(PortableFfnGeometry geometry, int64_t index, const std::vector<PortableTensorView>& weights,
         PortableFfnOptions opts) : g(geometry), layer(index), options(opts) {
        require(g.hidden > 0 && g.hidden <= 16384 && g.streams == 4 && g.rank > 0 && g.rank <= 4096 &&
                g.shared_ff > 0 && g.shared_ff <= 65536 && g.experts > 0 && g.experts <= 4096 &&
                g.selected > 0 && g.selected <= g.experts && g.selected <= 64 && layer >= 0 &&
                std::isfinite(g.epsilon) && g.epsilon > 0 && options.threads > 0,
                "portable FFN: invalid geometry/options");
        cpu.reset(ggml_backend_cpu_init());
        require(bool(cpu), "portable FFN: CPU backend unavailable");
        require(!options.accelerator || options.accelerator != cpu.get(), "portable FFN: duplicate backend");
        ggml_backend_cpu_set_n_threads(cpu.get(), options.threads);
        weights_ctx = context(); pre_ctx = context(); post_ctx = context();
        const int64_t width = g.hidden * g.streams;
        std::map<std::string, std::vector<int64_t>> shapes{
            {"hc_ffn_norm.weight", {width}}, {"hc_ffn_down.weight", {width, g.rank}},
            {"hc_ffn_up.weight", {g.rank, width}}, {"hc_ffn_inject.weight", {width, g.streams}},
            {"ffn_gate_inp.weight", {g.hidden, g.experts}},
            {"ffn_gate_inp_shexp.weight", {g.hidden}},
            {"ffn_gate_shexp.weight", {g.hidden, g.shared_ff}},
            {"ffn_up_shexp.weight", {g.hidden, g.shared_ff}},
            {"ffn_down_shexp.weight", {g.shared_ff, g.hidden}}};
        std::map<std::string, ggml_tensor*> w;
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        require(weights.size() == shapes.size(), "portable FFN: expected nine sublayer tensors");
        for (const auto& view : weights) {
            require(view.name.rfind(prefix, 0) == 0, "portable FFN: tensor belongs to another layer");
            const std::string key = view.name.substr(prefix.size());
            auto expected = shapes.find(key);
            require(expected != shapes.end() && view.shape == expected->second && !w.count(key),
                    "portable FFN: unexpected/duplicate tensor or shape: " + view.name);
            require(view.type == 0 || view.type == 8, "portable FFN: only F32/Q8_0 weights supported");
            require(view.data && (view.type != 8 || view.shape[0] % 32 == 0), "portable FFN: invalid tensor payload");
            // Norm/router/scalar gate are F32 in the confirmed fleet GGUF.
            if (key == "hc_ffn_norm.weight" || key == "ffn_gate_inp.weight" || key == "ffn_gate_inp_shexp.weight")
                require(view.type == 0, "portable FFN: norm/router/shared scalar gate require F32");
            auto* t = ggml_new_tensor(weights_ctx.get(), ggml_type(view.type), int(view.shape.size()), view.shape.data());
            ggml_set_name(t, view.name.c_str());
            require(ggml_nbytes(t) == view.bytes && view.bytes <= options.weight_budget - weights_size,
                    "portable FFN: weight size/budget mismatch: " + view.name);
            weights_size += view.bytes;
            w[key] = t;
        }
        const size_t allocation = ggml_backend_alloc_ctx_tensors_from_buft_size(weights_ctx.get(), ggml_backend_get_default_buffer_type(cpu.get()));
        require(allocation <= options.weight_budget, "portable FFN: aligned weight budget exceeded");
        weights_buffer.reset(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), cpu.get()));
        require(bool(weights_buffer), "portable FFN: weight allocation failed");
        ggml_backend_buffer_set_usage(weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto& view : weights) ggml_backend_tensor_set(w.at(view.name.substr(prefix.size())), view.data, 0, view.bytes);
        weights_size = ggml_backend_buffer_get_size(weights_buffer.get());

        auto* c = pre_ctx.get();
        input = named(ggml_new_tensor_3d(c, GGML_TYPE_F32, g.hidden, g.streams, 1), "residual");
        ggml_set_input(input);
        auto* norm = ggml_mul(c, ggml_reshape_2d(c, ggml_rms_norm(c, input, g.epsilon), width, 1), w.at("hc_ffn_norm.weight"));
        auto* low = ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, w.at("hc_ffn_down.weight"), norm), 1.0f / float(g.streams)));
        auto* gate = ggml_sigmoid(c, ggml_mul_mat(c, w.at("hc_ffn_up.weight"), low));
        auto* gated = ggml_reshape_2d(c, ggml_mul(c, norm, gate), g.hidden, g.streams);
        mixed = ggml_cont(c, ggml_view_1d(c, gated, g.hidden, 0));
        for (int64_t s = 1; s < g.streams; ++s)
            mixed = ggml_add(c, mixed, ggml_view_1d(c, gated, g.hidden, size_t(s * g.hidden) * sizeof(float)));
        mixed = named(ggml_scale(c, mixed, 1.0f / float(g.streams)), "mixed");
        inject = named(ggml_mul_mat(c, w.at("hc_ffn_inject.weight"), norm), "inject");
        logits = named(ggml_mul_mat(c, w.at("ffn_gate_inp.weight"), mixed), "router_logits");
        auto* sg = ggml_silu(c, ggml_mul_mat(c, w.at("ffn_gate_shexp.weight"), mixed));
        auto* su = ggml_mul_mat(c, w.at("ffn_up_shexp.weight"), mixed);
        shared = ggml_mul_mat(c, w.at("ffn_down_shexp.weight"), ggml_mul(c, sg, su));
        shared = named(ggml_mul(c, shared, ggml_sigmoid(c, ggml_mul_mat(c, w.at("ffn_gate_inp_shexp.weight"), mixed))), "shared");
        pre = ggml_new_graph(c);
        for (auto* t : {mixed, inject, logits, shared}) { ggml_set_output(t); ggml_build_forward_expand(pre, t); }
        pre_sched = portable_detail::schedule(pre, "pre-experts", options, cpu, placement, graphs_size);

        c = post_ctx.get();
        post_input = ggml_new_tensor_3d(c, GGML_TYPE_F32, g.hidden, g.streams, 1);
        post_inject = ggml_new_tensor_1d(c, GGML_TYPE_F32, g.streams);
        post_block = ggml_new_tensor_1d(c, GGML_TYPE_F32, g.hidden);
        for (auto* t : {post_input, post_inject, post_block}) ggml_set_input(t);
        auto* scatter = ggml_reshape_3d(c, ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, post_inject, 1.0f / float(g.streams))), 2.0f), 1, g.streams, 1);
        auto* block = ggml_repeat_4d(c, ggml_reshape_3d(c, post_block, g.hidden, 1, 1), g.hidden, g.streams, 1, 1);
        output = named(ggml_add(c, post_input, ggml_mul(c, block, scatter)), "residual_out");
        ggml_set_output(output);
        post = ggml_new_graph(c); ggml_build_forward_expand(post, output);
        post_sched = portable_detail::schedule(post, "post-experts", options, cpu, placement, graphs_size);
        placement.push_back({"host", "ROUTE_AND_MERGE", "indexed expert boundary", "F32", "CPU", "", {g.hidden, g.selected}});
    }
};

PortableFfn::PortableFfn(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PortableFfn::~PortableFfn() = default;
std::unique_ptr<PortableFfn> PortableFfn::create(PortableFfnGeometry g, int64_t layer,
    const std::vector<PortableTensorView>& weights, PortableFfnOptions options, std::string& error) {
    error.clear();
    try { return std::unique_ptr<PortableFfn>(new PortableFfn(std::make_unique<Impl>(g, layer, weights, options))); }
    catch (const std::exception& e) { error = e.what(); return nullptr; }
}

std::unique_ptr<PortableFfn> PortableFfn::from_gguf(const GgufModel& model, int64_t layer,
    PortableFfnOptions options, std::string& error) {
    error.clear();
    try {
        ModelGeometry g;
        const auto guard = read_geometry(model.meta(), g);
        require(guard.empty(), guard);
        require(layer >= 0 && layer < g.n_layers, "portable FFN: layer out of range");
        std::vector<PortableTensorView> views;
        for (const char* suffix : {"hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
                                  "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
                                  "ffn_up_shexp.weight", "ffn_down_shexp.weight"}) {
            const auto name = "blk." + std::to_string(layer) + "." + suffix;
            size_t shard = 0;
            const auto* t = model.find(name, &shard);
            require(t && model.in_bounds(*t, shard), "portable FFN: missing/out-of-bounds " + name);
            PortableTensorView v{name, {}, t->type, model.shard(shard).tensor_data(*t), size_t(tensor_payload_bytes(*t))};
            for (auto d : t->shape) { require(d <= INT64_MAX, "portable FFN: tensor dimension overflow"); v.shape.push_back(int64_t(d)); }
            views.push_back(std::move(v));
        }
        // read_geometry checks that routed and shared widths agree; its legacy
        // qwen4exp result stores that width in n_ff.
        return create({g.n_embd, g.hc, g.hc_lr, g.n_ff, g.n_expert, g.n_expert_used, float(g.rms_eps)}, layer, views, options, error);
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
}

bool PortableFfn::reset(const std::vector<float>& residual, std::string& error) {
    error.clear();
    try {
        require(residual.size() == size_t(impl_->g.hidden * impl_->g.streams), "portable FFN: residual size mismatch");
        finite(residual); impl_->state = residual; impl_->initialized = true; return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
bool PortableFfn::step(const PortableExpertRows& dispatch, PortableRoute& route, std::string& error) {
    error.clear(); route = {};
    try {
        auto& s = *impl_;
        require(s.initialized && bool(dispatch), "portable FFN: reset and expert callback required");
        write(s.input, s.state);
        require(ggml_backend_sched_graph_compute(s.pre_sched.get(), s.pre) == GGML_STATUS_SUCCESS, "portable FFN: pre graph failed");
        PortableRoute candidate;
        candidate.activation = read(s.mixed);
        route_logits(read(s.logits), s.g.selected, candidate);
        std::vector<float> rows;
        const bool dispatched = dispatch(s.layer, candidate, rows, error);
        require(dispatched, error.empty() ? "portable FFN: expert dispatch failed" : error);
        require(rows.size() == size_t(s.g.hidden * s.g.selected), "portable FFN: indexed expert row count mismatch");
        finite(rows);
        auto block = read(s.shared);
        for (int64_t h = 0; h < s.g.hidden; ++h) {
            float routed = 0;
            for (int64_t k = 0; k < s.g.selected; ++k) routed += rows[size_t(k * s.g.hidden + h)] * candidate.weights[size_t(k)];
            block[size_t(h)] += routed;
        }
        finite(block);
        write(s.post_input, s.state); write(s.post_inject, read(s.inject)); write(s.post_block, block);
        require(ggml_backend_sched_graph_compute(s.post_sched.get(), s.post) == GGML_STATUS_SUCCESS, "portable FFN: post graph failed");
        auto next = read(s.output);
        s.state = std::move(next); route = std::move(candidate); error.clear(); return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
const std::vector<float>& PortableFfn::residual() const { return impl_->state; }
const std::vector<PortablePlacement>& PortableFfn::placements() const { return impl_->placement; }
size_t PortableFfn::weight_bytes() const { return impl_->weights_size; }
size_t PortableFfn::graph_bytes() const { return impl_->graphs_size; }

PortableMemoryUsage PortableFfn::memory_usage() const {
    const auto &s = *impl_;
    PortableMemoryUsage m;
    m.host_weights = s.weights_size;
    for (auto *sched : {s.pre_sched.get(), s.post_sched.get()}) {
        m.host_graph += ggml_backend_sched_get_buffer_size(sched, s.cpu.get());
        if (s.options.accelerator && s.options.request_accelerator)
            m.device_graph += ggml_backend_sched_get_buffer_size(sched, s.options.accelerator);
    }
    m.host_extra_bound = 6u << 20; // three owning metadata contexts, plus token staging below
    m.host_extra_bound += size_t(s.g.hidden * s.g.streams * 8 + s.g.selected * s.g.hidden) * sizeof(float);
    m.host_extra_bound += ggml_graph_plan(s.pre, s.options.threads, nullptr).work_size +
                          ggml_graph_plan(s.post, s.options.threads, nullptr).work_size;
    if (s.options.request_accelerator && s.options.accelerator) {
        const size_t largest = std::max({s.g.hidden * s.g.streams * s.g.rank,
            s.g.hidden * s.g.shared_ff, s.g.hidden * s.g.experts});
        m.device_extra_bound = 8u << 20;
        portable_detail::account_matrix_conversion(m,s.options.accelerator,largest*sizeof(float));
    }
    return m;
}
} // namespace strata::core
