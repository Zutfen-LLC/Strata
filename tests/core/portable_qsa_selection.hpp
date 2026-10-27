#pragma once
#include "../../src/core/portable_graph.hpp"
#include "../../src/core/portable_qsa_graph.hpp"
#include "ggml-cpu.h"
#include "portable_qsa_fixture.hpp"
#include <numeric>
namespace qsa_fixture {
inline void selection_tests(PortableFfnOptions options) {
    // The shared scheduler prices a full backing root per view consumer.
    // Explicitly admit this conservative bound for the 24-head boundary fixture;
    // actual buffers remain small and are checked against this same limit.
    options.graph_budget = 1ull << 30;
    for (int n : {2, 2050, 2051, 2052, 2055, 2056})
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<float> scores(size_t(n), 0);
            for (int i = 0; i < n; ++i)
                scores[size_t(i)] = i >= n / 4 * 4 ? 1e9f
                                    : mode == 0    ? 0.0f
                                                   : float(mode == 1 ? (i / 4) % 7 : (n - i) / 4);
            portable_detail::Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2u << 20, nullptr, true}),
                                                                    ggml_free);
            auto *c = ctx.get();
            auto *input = ggml_new_tensor_1d(c, GGML_TYPE_F32, n);
            ggml_set_input(input);
            auto *output = portable_detail::qsa_selection(c, input);
            ggml_set_output(output);
            auto *ki = ggml_new_tensor_3d(c, GGML_TYPE_F16, 256, 2, n);
            auto *vi = ggml_new_tensor_3d(c, GGML_TYPE_F16, 256, 2, n);
            auto *qi = ggml_new_tensor_3d(c, GGML_TYPE_F32, 256, 12, 2);
            for (auto *t : {ki, vi, qi})
                ggml_set_input(t);
            auto *attention = portable_detail::qsa_dense_attention(c, ki, vi, qi, output);
            ggml_set_output(attention);
            auto *graph = ggml_new_graph(c);
            ggml_build_forward_expand(graph, output);
            ggml_build_forward_expand(graph, attention);
            std::vector<PortablePlacement> placement;
            size_t bytes = 0;
            auto sched = portable_detail::schedule(graph, "qsa-selection", options, cpu, placement, bytes);
            if (n > 2051)
                ggml_backend_tensor_set(input, scores.data(), 0, scores.size() * sizeof(float));
            std::vector<uint16_t> keys(size_t(n) * 512, 0), values(size_t(n) * 512);
            std::vector<float> query(6144);
            for (size_t h = 0; h < 24; ++h)
                for (size_t d = 0; d < 256; ++d)
                    query[h * 256 + d] = float(int(h % 7) - 3) * float(int(d % 5) - 2) * 0.03f;
            for (int cell = 0; cell < n; ++cell)
                for (int h = 0; h < 2; ++h)
                    for (int d = 0; d < 256; ++d)
                        values[size_t(cell) * 512 + size_t(h) * 256 + size_t(d)] = ggml_fp32_to_fp16(
                            float(cell % 23 - 11) * 0.003f + float(h) * 0.1f + float(d % 5) * 0.001f);
            for (int cell = 0; cell < n; ++cell)
                for (int h = 0; h < 2; ++h)
                    for (int d = 0; d < 256; ++d)
                        keys[size_t(cell) * 512 + size_t(h) * 256 + size_t(d)] =
                            ggml_fp32_to_fp16(float(cell % 7 - 3) * float(d % 5 - 2) * 0.02f + float(h) * 0.003f);
            if (n == 2) {
                for (size_t h = 0; h < 24; ++h)
                    for (size_t d = 0; d < 256; ++d)
                        query[h * 256 + d] = 0.3f + float(h) * 0.01f;
                for (size_t cell = 0; cell < 2; ++cell)
                    for (size_t h = 0; h < 2; ++h)
                        for (size_t d = 0; d < 256; ++d) {
                            keys[cell * 512 + h * 256 + d] = ggml_fp32_to_fp16(cell == 0 ? -0.2f : 0.2f);
                            values[cell * 512 + h * 256 + d] =
                                ggml_fp32_to_fp16(float(cell) * 0.4f + float(h) * 0.1f);
                        }
            }
            ggml_backend_tensor_set(ki, keys.data(), 0, keys.size() * 2);
            ggml_backend_tensor_set(vi, values.data(), 0, values.size() * 2);
            ggml_backend_tensor_set(qi, query.data(), 0, query.size() * 4);
            check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS,
                  "selection compute");
            std::vector<float> actual(size_t(n), 0), expected(size_t(n), 0);
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            std::vector<int> ids(size_t(n), 0);
            std::iota(ids.begin(), ids.end(), 0);
            std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                return scores[size_t(a)] != scores[size_t(b)] ? scores[size_t(a)] > scores[size_t(b)] : a < b;
            });
            for (int i = 0; i < std::min(n, 2051); ++i)
                expected[size_t(ids[size_t(i)])] = 1;
            close(actual, expected, 0, "deterministic cell selection and tail boundary");
            std::vector<float> actual_attention(6144), expected_attention(6144);
            ggml_backend_tensor_get(attention, actual_attention.data(), 0, 6144 * sizeof(float));
            for (size_t h = 0; h < 24; ++h) {
                std::vector<double> logits(size_t(n), 0);
                double max = -INFINITY, total = 0;
                for (int cell = 0; cell < n; ++cell)
                    if (expected[size_t(cell)] != 0) {
                        double dot = 0;
                        for (size_t d = 0; d < 256; ++d)
                            dot += double(query[h * 256 + d]) *
                                   ggml_fp16_to_fp32(keys[size_t(cell) * 512 + (h / 12) * 256 + d]);
                        logits[size_t(cell)] = dot / 16;
                        max = std::max(max, dot / 16);
                    }
                for (int cell = 0; cell < n; ++cell) {
                    logits[size_t(cell)] = expected[size_t(cell)] != 0 ? std::exp(logits[size_t(cell)] - max) : 0;
                    total += logits[size_t(cell)];
                }
                for (size_t d = 0; d < 256; ++d) {
                    double sum = 0;
                    for (int cell = 0; cell < n; ++cell)
                        sum += logits[size_t(cell)] / total *
                               ggml_fp16_to_fp32(values[size_t(cell) * 512 + (h / 12) * 256 + d]);
                    expected_attention[h * 256 + d] = float(sum);
                }
            }
            close(actual_attention, expected_attention, 0.0002f,
                  ("selected full-cache attention n=" + std::to_string(n)).c_str());
            if (mode == 2) {
                size_t topk = 0, mats = 0;
                for (const auto &p : placement) {
                    if (p.operation == "TOP_K")
                        ++topk;
                    if (p.operation == "MUL_MAT")
                        ++mats;
                }
                std::printf("PASS cached_cells=%d width=%d TOP_K=%zu MUL_MAT=%zu admitted_graph_limit=%zu "
                            "actual_scheduler_bytes=%zu\n",
                            n, std::min(n, 2051), topk, mats, options.graph_budget, bytes);
            }
            for (const auto &p : placement)
                if (options.request_accelerator && !options.allow_cpu_fallback && !p.metadata_only)
                    check(p.backend == ggml_backend_name(options.accelerator) && p.fallback_reason.empty(),
                          "strict selection placement");
        }
}
} // namespace qsa_fixture
