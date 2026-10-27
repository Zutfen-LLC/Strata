#pragma once
// Internal execution policy shared by the opt-in FFN and GDN coordinator paths.
#include "ggml-backend.h"
#include "ggml.h"
#include "strata/core/portable_ffn.hpp"
#include <memory>
#include <stdexcept>
namespace strata::core::portable_detail {
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
inline void require(bool ok, const std::string &error) {
    if (!ok)
        throw std::runtime_error(error);
}
inline bool metadata_only(ggml_tensor *t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}
inline Scheduler schedule(ggml_cgraph *graph, const char *phase, const PortableFfnOptions &options, Backend &cpu,
                          std::vector<PortablePlacement> &placement, size_t &graphs_size) {
    ggml_backend_t backends[] = {options.accelerator, cpu.get()};
    auto *list = options.accelerator && options.request_accelerator ? backends : backends + 1;
    const int count = list == backends ? 2 : 1;
    Scheduler sched(ggml_backend_sched_new(list, nullptr, count, GGML_DEFAULT_GRAPH_SIZE, false, false),
                    ggml_backend_sched_free);
    require(bool(sched), "portable graph: scheduler creation failed");
    const size_t begin = placement.size();
    std::vector<ggml_backend_t> assignments;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto *node = ggml_graph_node(graph, i);
        const bool alias = metadata_only(node);
        ggml_backend_t chosen = cpu.get();
        std::string reason;
        if (options.request_accelerator && !alias) {
            if (!options.accelerator)
                reason = "requested accelerator unavailable";
            else if (!ggml_backend_supports_op(options.accelerator, node))
                reason = "requested backend declines operation/type/shape";
            else
                chosen = options.accelerator;
            require(reason.empty() || options.allow_cpu_fallback,
                    "portable graph: " + reason + " for " + ggml_op_name(node->op));
        }
        require(ggml_backend_supports_op(chosen, node), "portable graph: CPU fallback declines operation");
        // Views must follow their backing storage. Pinning a host view as
        // device storage would suppress the scheduler's required copy into
        // its device consumer and send a CPU buffer into Vulkan execution.
        assignments.push_back(alias ? nullptr : chosen);
        placement.push_back({phase,
                             ggml_op_name(node->op),
                             ggml_get_name(node),
                             ggml_type_name(node->type),
                             ggml_backend_name(chosen),
                             reason,
                             {node->ne[0], node->ne[1], node->ne[2], node->ne[3]},
                             alias});
    }
    // At this pin reserve_size discards manual backend assignments. Do not
    // use its automatic CPU plan to price an explicitly placed device graph.
    // Instead bound all node storage plus a full source copy per consumer,
    // with backend-specific padding/alignment and no liveness reuse credit.
    // This overcounts same-device/reused inputs, but bounds scheduler copies
    // without private ggml APIs or allocating a trial device graph first.
    size_t bound = 0;
    const size_t remaining = options.graph_budget - graphs_size;
    auto account = [&](ggml_backend_t backend, ggml_tensor *t) {
        while (t->view_src)
            t = t->view_src;
        auto *buft = ggml_backend_get_default_buffer_type(backend);
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, t);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        require(alignment > 0 && bytes <= remaining - bound, "portable graph: conservative graph budget exceeded");
        bound += bytes;
        require(alignment - 1 <= remaining - bound, "portable graph: graph alignment budget exceeded");
        bound += alignment - 1;
    };
    auto assign = [&] {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            if (assignments[size_t(i)])
                ggml_backend_sched_set_tensor_backend(sched.get(), ggml_graph_node(graph, i), assignments[size_t(i)]);
    };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto *node = ggml_graph_node(graph, i);
        auto *backend = assignments[size_t(i)] ? assignments[size_t(i)] : cpu.get();
        if (!node->buffer && !node->view_src)
            account(backend, node);
        for (auto *src : node->src)
            if (src) {
                account(backend, src);
                auto *root = src;
                while (root->view_src)
                    root = root->view_src;
                // Unallocated graph inputs can get original storage on either
                // backend as well as the consumer copy accounted above.
                if (root->op == GGML_OP_NONE && !root->buffer)
                    for (int b = 0; b < count; ++b)
                        account(list[b], root);
            }
    }
    assign();
    // Fresh-scheduler alloc_graph reserves backing buffers internally. This
    // is a single split/allocation pass: no reserve_size reset can discard
    // pins, and no earlier split can leave temporary copy sources in this
    // owning graph before the scheduler rebuilds its context.
    require(ggml_backend_sched_alloc_graph(sched.get(), graph), "portable graph: graph allocation failed");
    // Report the scheduler's actual assignment as well as any support rejection.
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto *actual = ggml_backend_sched_get_tensor_backend(sched.get(), ggml_graph_node(graph, i));
        auto &record = placement[begin + size_t(i)];
        require(actual || record.metadata_only, "portable graph: computation has no actual backend assignment");
        if (actual) {
            record.backend = ggml_backend_name(actual);
            if (options.request_accelerator && !record.metadata_only && actual != options.accelerator &&
                record.fallback_reason.empty()) {
                record.fallback_reason = "scheduler placed operation on CPU";
                require(options.allow_cpu_fallback, "portable graph: scheduler fallback forbidden");
            }
        }
    }
    size_t allocated = 0;
    for (int i = 0; i < count; ++i)
        allocated += ggml_backend_sched_get_buffer_size(sched.get(), list[i]);
    require(allocated <= bound && allocated <= remaining, "portable graph: allocated graph exceeds admitted bound");
    graphs_size += allocated;
    return sched;
}

} // namespace strata::core::portable_detail
