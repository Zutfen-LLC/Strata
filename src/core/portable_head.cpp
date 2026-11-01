#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "portable_graph.hpp"
#include "portable_head_admission.hpp"
#include "portable_logits_guard.hpp"
#include "strata/core/portable_model.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <iostream>

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
void validate_geometry(const PortableHeadGeometry& g) {
    require(g.hidden > 0 && g.hidden <= 16384 && g.streams == 4 && g.rank > 0 && g.rank <= 4096 &&
                g.vocabulary > 0 && g.vocabulary <= 1048576 && std::isfinite(g.epsilon) && g.epsilon > 0,
            "portable head: invalid geometry");
}
std::map<std::string, ggml_tensor*> weight_metadata(ggml_context* c, const PortableHeadGeometry& g,
                                                 const std::vector<PortableTensorView>& views, size_t budget) {
    validate_geometry(g);
    const int64_t width = g.hidden * g.streams;
    size_t payload_bytes = 0;
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
        auto *t = ggml_new_tensor(c, ggml_type(v.type), int(v.shape.size()), v.shape.data());
        ggml_set_name(t, v.name.c_str());
        require(ggml_nbytes(t) == v.bytes && payload_bytes <= budget &&
                    v.bytes <= budget - payload_bytes,
                "portable head: weight budget/size");
        payload_bytes += v.bytes;
        w[v.name] = t;
    }
    return w;
}
size_t aligned_weight_size(ggml_context* c, ggml_backend_t backend) {
    require(backend != nullptr, "portable head: weight estimate requires selected backend");
    auto* buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    require(alignment > 0 && (alignment & (alignment-1)) == 0, "portable head: invalid weight alignment");
    size_t total = 0;
    for (auto* t=ggml_get_first_tensor(c); t; t=ggml_get_next_tensor(c,t)) {
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft,t);
        require(bytes <= SIZE_MAX-(alignment-1), "portable head: weight alignment overflow");
        const size_t padded = (bytes+alignment-1) & ~(alignment-1);
        require(padded <= SIZE_MAX-total, "portable head: weight allocation overflow");
        total += padded;
    }
    // Same checked padding/sum method as QSA state admission. GGML can split
    // this weight context across buffers; the sum prices every chunk. No backend allocation.
    require(total == ggml_backend_alloc_ctx_tensors_from_buft_size(c,buft),
            "portable head: weight estimator mismatch");
    return total;
}
struct HeadGraph { ggml_tensor *input, *output; ggml_cgraph* graph; };
HeadGraph build_head_graph(ggml_context* c, const PortableHeadGeometry& g,
                          const std::map<std::string, ggml_tensor*>& w) {
    const int64_t width=g.hidden*g.streams;
    ggml_tensor *input=nullptr,*output=nullptr;
    ggml_cgraph* graph=nullptr;
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
    return {input,output,graph};
}
size_t add_bound(size_t a,size_t b) {
    require(b<=SIZE_MAX-a,"portable head: admission overflow");return a+b;
}
// Deliberately mirrors portable_graph's no-reuse node/source-copy envelope.
// Max of both default buffer types bounds either scheduler placement; every
// unallocated graph input gets original storage on BOTH backends as well.
// Immutable weights already have separately admitted original storage.
size_t graph_storage_bound(ggml_cgraph* graph, ggml_backend_t cpu, ggml_backend_t accelerator,
                          const std::map<std::string,ggml_tensor*>& weights) {
    auto bytes=[&](ggml_backend_t backend,ggml_tensor* t) {
        while(t->view_src)t=t->view_src;
        auto* buft=ggml_backend_get_default_buffer_type(backend);
        const size_t alignment=ggml_backend_buft_get_alignment(buft);
        require(alignment && !(alignment&(alignment-1)),"portable head: graph alignment");
        return add_bound(ggml_backend_buft_get_alloc_size(buft,t),alignment-1);
    };
    auto external=[&](ggml_tensor* root) {
        for(const auto& item:weights)if(item.second==root)return true;
        return false;
    };
    auto max_bytes=[&](ggml_tensor* t) {
        const size_t host=bytes(cpu,t);
        return accelerator?std::max(host,bytes(accelerator,t)):host;
    };
    size_t bound=0;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto* node=ggml_graph_node(graph,i);
        if(!node->view_src)bound=add_bound(bound,max_bytes(node));
        for(auto* src:node->src)if(src) {
            auto* root=src;while(root->view_src)root=root->view_src;
            bound=add_bound(bound,max_bytes(root));
            if(root->op==GGML_OP_NONE && !external(root)) {
                bound=add_bound(bound,bytes(cpu,root));
                if(accelerator)bound=add_bound(bound,bytes(accelerator,root));
            }
        }
    }
    return bound;
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
    std::unique_ptr<portable_detail::PortableLogitsGuard> logits_guard;
    std::vector<PortablePlacement> placement;
    size_t device_weights = 0;
    size_t weight_size = 0, graph_size = 0, device_bytes = 0, device_workspace = 0, device_conversion = 0;
    ggml_backend_t accelerator = nullptr;
    Impl(PortableHeadGeometry geometry, const std::vector<PortableTensorView> &views, PortableFfnOptions opts, bool guard_logits)
        : g(geometry) {
        validate_geometry(g);
        require(opts.threads > 0, "portable head: invalid thread count");
        cpu.reset(ggml_backend_cpu_init());
        require(bool(cpu), "portable head: CPU unavailable");
        ggml_backend_cpu_set_n_threads(cpu.get(), opts.threads);
        const int64_t width = g.hidden * g.streams;
        auto w = weight_metadata(weights.get(), g, views, opts.weight_budget);
        auto *weight_backend = portable_detail::weight_backend(opts, cpu.get());
        const size_t weight_bound = aligned_weight_size(weights.get(),weight_backend);
        require(weight_bound <= opts.weight_budget, "portable head: aligned weight budget");
        buffer.reset(ggml_backend_alloc_ctx_tensors(weights.get(), weight_backend));
        require(bool(buffer), "portable head: weight allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto &v : views)
            ggml_backend_tensor_set(w.at(v.name), v.data, 0, v.bytes);
        weight_size = ggml_backend_buffer_get_size(buffer.get());
        require(weight_size <= weight_bound && weight_size <= opts.weight_budget,
                "portable head: actual aligned weight budget/estimate exceeded");
        device_weights = weight_backend != cpu.get() ? weight_size : 0;
        auto *c = ctx.get();
        const auto built=build_head_graph(c,g,w);
        input=built.input;output=built.output;graph=built.graph;
        const size_t workspace = ggml_graph_plan(graph, opts.threads, nullptr).work_size;
        require(workspace <= opts.graph_budget, "portable head: CPU workspace budget");
        graph_size = workspace;
        if (opts.request_accelerator && opts.accelerator) {
            // Only discount the full output-matrix conversion when the exact
            // pinned T=1 matvec path and physical storage range are proven.
            accelerator=opts.accelerator;
            const auto limit=portable_detail::head_storage_limit(accelerator);
            const bool direct=portable_detail::direct_head_q8_matvec(output,limit);
            const size_t output_f32=size_t(g.hidden*g.vocabulary)*sizeof(float);
            const size_t hc_f32=size_t(width*g.rank)*sizeof(float);
            const size_t conservative=portable_detail::head_conversion_bound(output_f32,hc_f32,false);
            device_conversion=portable_detail::head_conversion_bound(output_f32,hc_f32,direct);
            const size_t other=size_t(g.vocabulary*16+width*4)*sizeof(float);
            device_workspace=device_conversion+other;
            std::clog << "head_admission pci=" << (limit.pci.empty()?"unknown":limit.pci)
                      << " max_storage_buffer_range=" << limit.max_storage_buffer_range
                      << " weight_type=" << ggml_type_name(output->src[0]->type)
                      << " weight_ne0=" << output->src[0]->ne[0] << " weight_ne1=" << output->src[0]->ne[1]
                      << " weight_bytes=" << ggml_nbytes(output->src[0])
                      << " weight_contiguous=" << ggml_is_contiguous(output->src[0])
                      << " rhs_type=" << ggml_type_name(output->src[1]->type) << " tokens=" << output->ne[1]
                      << " direct_q8_matvec=" << direct << " conservative_conversion_bytes=" << conservative
                      << " retained_conversion_bytes=" << device_conversion << " other_workspace_bytes=" << other << '\n';
            require(device_workspace <= opts.graph_budget - graph_size, "portable head: device workspace budget");
            graph_size += device_workspace;
        }
        scheduler = portable_detail::schedule(graph, "global-head", opts, cpu, placement, graph_size);
        if (opts.request_accelerator && opts.accelerator)
            device_bytes = ggml_backend_sched_get_buffer_size(scheduler.get(), opts.accelerator);
        if (guard_logits) {
            auto *backend = opts.request_accelerator && opts.accelerator ? opts.accelerator : cpu.get();
            require(graph_size <= opts.graph_budget, "portable head: guard graph budget");
            logits_guard = std::make_unique<portable_detail::PortableLogitsGuard>(backend, opts.graph_budget - graph_size);
            const size_t guard_bytes = logits_guard->device_bytes() + logits_guard->device_resource_bound();
            require(guard_bytes <= opts.graph_budget - graph_size, "portable head: integer guard budget");
            graph_size += guard_bytes;
            placement.push_back({"global-head-guard", "BIT_NONFINITE", "portable_logits_status", "I32",
                                 ggml_backend_name(backend), "", {16, 1, 1, 1}, false});
        }
    }
};
size_t PortableHead::weight_allocation_size(ggml_backend* backend, PortableHeadGeometry geometry,
                                          const std::vector<PortableTensorView>& views) {
    auto metadata = context();
    weight_metadata(metadata.get(),geometry,views,SIZE_MAX);
    return aligned_weight_size(metadata.get(),backend);
}
PortableHeadAdmission PortableHead::admission(PortableHeadGeometry g,
    const std::vector<PortableTensorView>& views, PortableFfnOptions opts, bool guard_logits) {
    require(opts.threads>0,"portable head: invalid thread count");
    auto weights=context(),ctx=context();
    auto w=weight_metadata(weights.get(),g,views,opts.weight_budget);
    portable_detail::Backend cpu(ggml_backend_cpu_init(),ggml_backend_free);
    require(bool(cpu),"portable head: CPU unavailable for metadata sizing");
    auto built=build_head_graph(ctx.get(),g,w);
    auto* accelerator=opts.request_accelerator?opts.accelerator:nullptr;
    PortableHeadAdmission a;
    a.scheduler_bound=graph_storage_bound(built.graph,cpu.get(),accelerator,w);
    a.cpu_workspace=ggml_graph_plan(built.graph,opts.threads,nullptr).work_size;
    if(accelerator) {
        const auto limit=portable_detail::head_storage_limit(accelerator);
        a.direct_q8=portable_detail::direct_head_q8_matvec(built.output,limit);
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
        a.verified_vulkan=ggml_backend_is_vk(accelerator) && a.direct_q8 && !opts.allow_cpu_fallback;
#endif
        const size_t output_f32=size_t(g.hidden*g.vocabulary)*sizeof(float);
        const size_t hc_f32=size_t(g.hidden*g.streams*g.rank)*sizeof(float);
        a.device_workspace=add_bound(portable_detail::head_conversion_bound(output_f32,hc_f32,a.direct_q8),
            size_t(g.vocabulary*16+g.hidden*g.streams*4)*sizeof(float));
        if(guard_logits) {
            auto status=context();ggml_new_tensor_1d(status.get(),GGML_TYPE_I32,16);
            a.guard_bound=add_bound(aligned_weight_size(status.get(),accelerator),1u<<20);
        }
    }
    a.graph_bound=add_bound(add_bound(a.scheduler_bound,a.cpu_workspace),
                            add_bound(a.device_workspace,a.guard_bound));
    return a;
}
PortableHead::PortableHead(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PortableHead::~PortableHead() = default;
std::unique_ptr<PortableHead> PortableHead::create(PortableHeadGeometry g, const std::vector<PortableTensorView> &v,
                                                   PortableFfnOptions o, std::string &error, bool guard_logits) {
    error.clear();
    try {
        return std::unique_ptr<PortableHead>(new PortableHead(std::make_unique<Impl>(g, v, o, guard_logits)));
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
            if (!std::isfinite(v))
                require(false, "portable head: nonfinite residual");
        ggml_backend_tensor_set(s.input, streams.data(), 0, streams.size() * sizeof(float));
        require(ggml_backend_sched_graph_compute(s.scheduler.get(), s.graph) == GGML_STATUS_SUCCESS,
                "portable head: compute failed");
        if (s.logits_guard)
            require(s.logits_guard->check(s.output), "portable head: nonfinite logits (integer status guard)");
        std::vector<float> next(size_t(s.g.vocabulary));
        ggml_backend_tensor_get(s.output, next.data(), 0, next.size() * sizeof(float));
        for (float v : next)
            if (!std::isfinite(v))
                require(false, "portable head: nonfinite logits");
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
    m.host_weights = s.weight_size - s.device_weights;
    m.device_weights = s.device_weights;
    m.host_graph = ggml_backend_sched_get_buffer_size(s.scheduler.get(), s.cpu.get());
    // A head borrows at most one accelerator; its actual reservation is retained.
    m.device_graph = s.device_bytes;
    const size_t guard_device = s.logits_guard ? s.logits_guard->device_bytes() +
                                               s.logits_guard->device_resource_bound() : 0;
    const size_t buffers = m.host_graph + m.device_graph + guard_device;
    m.device_state += s.logits_guard ? s.logits_guard->device_bytes() : 0;
    m.device_extra_bound += s.logits_guard ? s.logits_guard->device_resource_bound() : 0;
    m.host_extra_bound = (4u << 20) + size_t(s.g.vocabulary * 2 + s.g.hidden * 16) * sizeof(float);
    if (s.graph_size > buffers)
        m.host_extra_bound += s.graph_size - buffers;
    if (s.logits_guard) m.host_extra_bound += s.logits_guard->host_bound();
    if(s.device_workspace) {
        const size_t conversion=s.device_conversion;
        m.device_extra_bound+=s.device_workspace-conversion;
        portable_detail::account_matrix_conversion(m,s.accelerator,conversion);
    }
    return m;
}
} // namespace strata::core
