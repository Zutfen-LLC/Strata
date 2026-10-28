#pragma once
#include "../../src/core/portable_logits_guard.hpp"
#include "../../src/core/portable_gdn_finite.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <bit>
#include <memory>
#include <stdexcept>
#include <vector>

inline void logits_guard_test(ggml_backend_t backend) {
    using namespace strata::core;
    auto check = [](bool ok, const char *message) { if (!ok) throw std::runtime_error(message); };
    for (int64_t n : {1, 17, 257, 4097, 248320}) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2u << 20, nullptr, true}), ggml_free);
        check(bool(ctx), "guard context");
        auto *input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
        ggml_set_input(input);
        auto *status = portable_detail::logits_status(ctx.get(), input);
        auto *graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, status);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            check(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)), "guard backend support required");
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
        check(ggml_gallocr_alloc_graph(alloc.get(), graph), "guard allocation");
        const uint32_t valid[] = {0, 0x80000000u, 1, 0x80000001u, 0x00800000u, 0x7f7fffffu, 0xff7fffffu};
        std::vector<float> values(size_t(n), 0);
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::bit_cast<float>(valid[i % std::size(valid)]);
        auto run = [&] {
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
            check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "guard compute");
            std::vector<float> result(1);
            ggml_backend_tensor_get(status, result.data(), 0, sizeof(float));
            return portable_detail::gdn_all_finite(result) && result[0] == 0;
        };
        check(run(), "guard rejects finite edges or overflows");
        for (size_t at : {size_t(0), size_t(n / 2), size_t(n - 1)}) {
            const float saved = values[at];
            for (uint32_t bad : {0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800001u}) {
                values[at] = std::bit_cast<float>(bad);
                check(!run(), "guard missed infinity/quiet/signaling NaN");
            }
            values[at] = saved;
        }
        check(run(), "guard status must refresh after failure");
    }
}
