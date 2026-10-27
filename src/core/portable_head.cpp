#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "portable_graph.hpp"
#include "strata/core/portable_model.hpp"
#include <algorithm>
#include <cmath>
#include <map>

namespace strata::core {
using portable_detail::require;
namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
Context context() {
    Context c(ggml_init({2u << 20, nullptr, true}), ggml_free);
    require(bool(c), "portable head: metadata allocation failed");
    return c;
}
} // namespace
struct PortableHead::Impl {
    PortableHeadGeometry g;
    portable_detail::Backend cpu{nullptr, ggml_backend_free};
    Context weights{context()}, ctx{context()};
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    portable_detail::Scheduler scheduler{nullptr, ggml_backend_sched_free};
    ggml_cgraph *graph = nullptr;
    ggml_tensor *input = nullptr, *output = nullptr;
    std::vector<PortablePlacement> placement;
    size_t weight_size = 0, graph_size = 0, device_bytes = 0, device_workspace = 0;
    Impl(PortableHeadGeometry geometry, const std::vector<PortableTensorView> &views, PortableFfnOptions opts)
        : g(geometry) {
        require(g.hidden > 0 && g.hidden <= 16384 && g.streams == 4 && g.rank > 0 && g.rank <= 4096 &&
                    g.vocabulary > 0 && g.vocabulary <= 1048576 && opts.threads > 0 && std::isfinite(g.epsilon) &&
                    g.epsilon > 0,
                "portable head: invalid geometry/options");
        cpu.reset(ggml_backend_cpu_init());
        require(bool(cpu), "portable head: CPU unavailable");
        ggml_backend_cpu_set_n_threads(cpu.get(), opts.threads);
        const int64_t width = g.hidden * g.streams;
        const std::map<std::string, std::vector<int64_t>> shapes{{"output_hc_norm.weight", {width}},
                                                                 {"output_hc_down.weight", {width, g.rank}},
                                                                 {"output_hc_up.weight", {g.rank, width}},
                                                                 {"output.weight", {g.hidden, g.vocabulary}}};
        require(views.size() == shapes.size(), "portable head: expected four weights");
        std::map<std::string, ggml_tensor *> w;
        for (const auto &v : views) {
            const auto expected = shapes.find(v.name);
            require(expected != shapes.end() && v.shape == expected->second && !w.count(v.name) && v.data &&
                        (v.type == 0 || v.type == 8) && (v.type != 8 || v.shape[0] % 32 == 0),
                    "portable head: unexpected tensor/type/shape: " + v.name);
            require(v.name != "output_hc_norm.weight" || v.type == 0, "portable head: norm requires F32");
            auto *t = ggml_new_tensor(weights.get(), ggml_type(v.type), int(v.shape.size()), v.shape.data());
            ggml_set_name(t, v.name.c_str());
            require(ggml_nbytes(t) == v.bytes && weight_size <= opts.weight_budget &&
                        v.bytes <= opts.weight_budget - weight_size,
                    "portable head: weight budget/size");
            weight_size += v.bytes;
            w[v.name] = t;
        }
        require(ggml_backend_alloc_ctx_tensors_from_buft_size(
                    weights.get(), ggml_backend_get_default_buffer_type(cpu.get())) <= opts.weight_budget,
                "portable head: aligned weight budget");
        buffer.reset(ggml_backend_alloc_ctx_tensors(weights.get(), cpu.get()));
        require(bool(buffer), "portable head: weight allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto &v : views)
            ggml_backend_tensor_set(w.at(v.name), v.data, 0, v.bytes);
        weight_size = ggml_backend_buffer_get_size(buffer.get());
        auto *c = ctx.get();
        input = ggml_new_tensor_3d(c, GGML_TYPE_F32, g.hidden, g.streams, 1);
        ggml_set_input(input);
        auto *norm = ggml_mul(c, ggml_reshape_2d(c, ggml_rms_norm(c, input, g.epsilon), width, 1),
                              w.at("output_hc_norm.weight"));
        auto *low = ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, w.at("output_hc_down.weight"), norm), 0.25f));
        auto *gate = ggml_sigmoid(c, ggml_mul_mat(c, w.at("output_hc_up.weight"), low));
        auto *weighted = ggml_reshape_2d(c, ggml_mul(c, norm, gate), g.hidden, g.streams);
        auto *mixed = ggml_cont(c, ggml_view_1d(c, weighted, g.hidden, 0));
        for (int64_t s = 1; s < g.streams; ++s)
            mixed = ggml_add(c, mixed, ggml_view_1d(c, weighted, g.hidden, size_t(s * g.hidden) * sizeof(float)));
        output = ggml_mul_mat(c, w.at("output.weight"), ggml_scale(c, mixed, 0.25f));
        ggml_set_name(output, "portable_logits");
        ggml_set_output(output);
        graph = ggml_new_graph(c);
        ggml_build_forward_expand(graph, output);
        const size_t workspace = ggml_graph_plan(graph, opts.threads, nullptr).work_size;
        require(workspace <= opts.graph_budget, "portable head: CPU workspace budget");
        graph_size = workspace;
        if (opts.request_accelerator && opts.accelerator) {
            // Pinned Vulkan tensor workspace: a full F32 image of the largest head matrix plus
            // input/output conversion and split-K temporary rows (<=8 splits).
            device_workspace =
                std::max(size_t(g.hidden * g.vocabulary), size_t(width * g.rank)) * sizeof(float) + size_t(g.vocabulary * 16 + width * 4) * sizeof(float);
            require(device_workspace <= opts.graph_budget - graph_size, "portable head: device workspace budget");
            graph_size += device_workspace;
        }
        scheduler = portable_detail::schedule(graph, "global-head", opts, cpu, placement, graph_size);
        if (opts.request_accelerator && opts.accelerator)
            device_bytes = ggml_backend_sched_get_buffer_size(scheduler.get(), opts.accelerator);
    }
};
PortableHead::PortableHead(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PortableHead::~PortableHead() = default;
std::unique_ptr<PortableHead> PortableHead::create(PortableHeadGeometry g, const std::vector<PortableTensorView> &v,
                                                   PortableFfnOptions o, std::string &error) {
    error.clear();
    try {
        return std::unique_ptr<PortableHead>(new PortableHead(std::make_unique<Impl>(g, v, o)));
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
bool PortableHead::run(const std::vector<float> &streams, std::vector<float> &logits, std::string &error) {
    error.clear();
    logits.clear();
    try {
        auto &s = *impl_;
        require(streams.size() == size_t(s.g.hidden * s.g.streams), "portable head: residual size");
        for (float v : streams)
            require(std::isfinite(v), "portable head: nonfinite residual");
        ggml_backend_tensor_set(s.input, streams.data(), 0, streams.size() * sizeof(float));
        require(ggml_backend_sched_graph_compute(s.scheduler.get(), s.graph) == GGML_STATUS_SUCCESS,
                "portable head: compute failed");
        std::vector<float> next(size_t(s.g.vocabulary));
        ggml_backend_tensor_get(s.output, next.data(), 0, next.size() * sizeof(float));
        for (float v : next)
            require(std::isfinite(v), "portable head: nonfinite logits");
        logits.swap(next);
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}
size_t PortableHead::weight_bytes() const { return impl_->weight_size; }
size_t PortableHead::graph_bytes() const { return impl_->graph_size; }
const std::vector<PortablePlacement> &PortableHead::placements() const { return impl_->placement; }
PortableMemoryUsage PortableHead::memory_usage() const {
    const auto &s = *impl_;
    PortableMemoryUsage m;
    m.host_weights = s.weight_size;
    m.host_graph = ggml_backend_sched_get_buffer_size(s.scheduler.get(), s.cpu.get());
    // A head borrows at most one accelerator; its actual reservation is retained.
    m.device_graph = s.device_bytes;
    const size_t buffers = m.host_graph + m.device_graph;
    m.host_extra_bound = (4u << 20) + size_t(s.g.vocabulary * 2 + s.g.hidden * 16) * sizeof(float);
    if (s.graph_size > buffers)
        m.host_extra_bound += s.graph_size - buffers;
    m.device_extra_bound = s.device_workspace;
    return m;
}
} // namespace strata::core
