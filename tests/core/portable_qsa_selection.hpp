#pragma once
#include "../../src/core/portable_graph.hpp"
#include "../../src/core/portable_qsa_graph.hpp"
#include "ggml-cpu.h"
#include "portable_qsa_fixture.hpp"
#include <numeric>
namespace qsa_fixture {
// Uses the real coordinator's retained spare, then the production score and
// selection builders at a long-cache boundary without long-cache projections.
inline void spare_high_score_test(PortableFfnOptions options) {
    Fixture f;
    std::fill(f.data[11].begin(), f.data[11].end(), 0);
    const auto scale = ggml_fp32_to_fp16(0.25f);
    std::memcpy(f.data[11].data() + 64 * 2720, &scale, 2);
    f.data[11][64 * 2720 + 2] = 1; // first raw key is strictly positive e64
    for (size_t j = 0; j < 128; ++j) {
        const float norm = 1e5f / std::sqrt(128.0f);
        std::memcpy(f.data[13].data() + j * 4, &norm, 4);
    }
    PortableQsaOptions opts;
    opts.capacity = 1;
    opts.execution = options;
    std::string error;
    auto layer = PortableQsaLayer::create(3, f.views, opts, error);
    check(bool(layer), error);
    PortableRoute route;
    PortableQsaTrace trace;
    check(layer->step(0, std::vector<float>(10240, 1), experts(), route, error, &trace), error);
    auto expected_spare = trace.index_raw_key;
    normalize(expected_spare, 128, f, 13);
    rotate(expected_spare, 128, 0);
    check(expected_spare[64] > 9e4f, "high-score first-key magnitude");
    constexpr int64_t cells = 2055, complete = 513, blocks = 514;
    std::vector<float> pooled(size_t(blocks) * 128, 0), query(512, 0), bias(size_t(cells), 0);
    std::vector<int32_t> mapping(size_t(cells), 0);
    for (size_t b = 0; b < size_t(complete); ++b) {
        pooled[b * 128 + 64] = 2e4f;
        pooled[b * 128 + 65] = std::sqrt(1e10f - 4e8f);
    }
    std::copy(layer->pooled_keys().begin(), layer->pooled_keys().end(), pooled.begin() + complete * 128);
    query[64] = 1e5f;
    for (size_t cell = 0; cell < size_t(cells); ++cell) {
        mapping[cell] = int32_t(cell / 4);
        if (cell >= size_t(complete) * 4)
            bias[cell] = 1e9f;
    }
    portable_detail::Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2u << 20, nullptr, true}), ggml_free);
    auto *c = ctx.get();
    auto *pk = ggml_new_tensor_2d(c, GGML_TYPE_F32, 128, blocks);
    auto *q = ggml_new_tensor_2d(c, GGML_TYPE_F32, 128, 4);
    auto *map = ggml_new_tensor_1d(c, GGML_TYPE_I32, cells);
    auto *tail_bias = ggml_new_tensor_1d(c, GGML_TYPE_F32, cells);
    for (auto *t : {pk, q, map, tail_bias})
        ggml_set_input(t);
    auto *scores = portable_detail::qsa_cell_scores(c, pk, q, map, tail_bias);
    auto *selected = portable_detail::qsa_selection(c, scores);
    ggml_set_output(scores);
    ggml_set_output(selected);
    auto *graph = ggml_new_graph(c);
    ggml_build_forward_expand(graph, scores);
    ggml_build_forward_expand(graph, selected);
    size_t bytes = 0;
    std::vector<PortablePlacement> placement;
    auto sched = portable_detail::schedule(graph, "qsa-spare-regression", options, cpu, placement, bytes);
    ggml_backend_tensor_set(pk, pooled.data(), 0, pooled.size() * 4);
    ggml_backend_tensor_set(q, query.data(), 0, query.size() * 4);
    ggml_backend_tensor_set(map, mapping.data(), 0, mapping.size() * 4);
    ggml_backend_tensor_set(tail_bias, bias.data(), 0, bias.size() * 4);
    check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "high-score graph");
    std::vector<float> actual(size_t(cells), 0), actual_scores(size_t(cells), 0), expected(size_t(cells), 0);
    ggml_backend_tensor_get(selected, actual.data(), 0, actual.size() * 4);
    ggml_backend_tensor_get(scores, actual_scores.data(), 0, actual_scores.size() * 4);
    for (size_t cell = 0; cell < 2048; ++cell)
        expected[cell] = 1;
    for (size_t cell = 2052; cell < 2055; ++cell)
        expected[cell] = 1;
    close(actual, expected, 0, "high-score spare retains tails2052-2054 and complete0-2047");
    close(layer->pooled_keys(), expected_spare, 3e-5f, "first normalized raw key is spare");
    const float expected_tail = 1e9f + 1e5f * expected_spare[64];
    check(expected_tail > 1e10f, "true tail exceeds completed block scores");
    for (size_t cell = 2052; cell < 2055; ++cell)
        check(std::fabs(actual_scores[cell] - expected_tail) < 1e-5f * expected_tail,
              "spare dot plus finite tail bias");
    for (const auto &p : placement)
        if (options.request_accelerator && !options.allow_cpu_fallback && !p.metadata_only)
            check(p.backend == ggml_backend_name(options.accelerator) && p.fallback_reason.empty(),
                  "strict high-score placement");
    std::printf("PASS high-score spare cached_cells=2055 width=2051 tails2052-2054 complete0-2047\n");
}
inline void selection_tests(PortableFfnOptions options, bool reuse = false, bool external_state = false) {
    // The shared scheduler prices a full backing root per view consumer.
    // Explicitly admit this conservative bound for the 24-head boundary fixture;
    // actual buffers remain small and are checked against this same limit.
    options.graph_budget = 1ull << 30;
    portable_detail::Backend retained_cpu(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> retained_ctx(ggml_init({2u << 20, nullptr, true}), ggml_free);
    portable_detail::Scheduler retained_sched{nullptr, ggml_backend_sched_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_ctx(nullptr, ggml_free);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> state_buffer(nullptr, ggml_backend_buffer_free);
    ggml_tensor *state_k = nullptr, *state_v = nullptr, *state_scores = nullptr;
    if (external_state) {
        check(reuse, "external state selection requires reuse");
        state_ctx.reset(ggml_init({4096, nullptr, true}));
        state_k = ggml_new_tensor_3d(state_ctx.get(), GGML_TYPE_F16, 256, 2, 2056);
        state_v = ggml_new_tensor_3d(state_ctx.get(), GGML_TYPE_F16, 256, 2, 2056);
        state_scores = ggml_new_tensor_1d(state_ctx.get(), GGML_TYPE_F32, 2056);
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(state_ctx.get(), options.request_accelerator ? options.accelerator : retained_cpu.get()));
        check(bool(state_buffer), "external selection state allocation");
    }
    size_t retained_buffers = 0;
    ggml_backend_buffer_t key_buffer = nullptr, value_buffer = nullptr;
    for (int sweep = 0; sweep < (reuse ? 2 : 1); ++sweep)
    for (int n : {2056, 2, 2050, 2051, 2052, 2055, 2056})
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<float> scores(size_t(n), 0);
            for (int i = 0; i < n; ++i)
                scores[size_t(i)] = i >= n / 4 * 4 ? 1e9f
                                    : mode == 0    ? 0.0f
                                                   : float(mode == 1 ? (i / 4) % 7 : (n - i) / 4);
            portable_detail::Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2u << 20, nullptr, true}),
                                                                    ggml_free);
            if (reuse && retained_sched)
                ggml_backend_sched_reset(retained_sched.get());
            if (reuse)
                ggml_reset(retained_ctx.get());
            auto *c = reuse ? retained_ctx.get() : ctx.get();
            auto *input = external_state ? ggml_view_1d(c, state_scores, n, 0) : ggml_new_tensor_1d(c, GGML_TYPE_F32, n);
            if (external_state) check(ggml_backend_view_init(input) == GGML_STATUS_SUCCESS,"external score view");
            else ggml_set_input(input);
            auto *output = portable_detail::qsa_selection(c, input);
            ggml_set_output(output);
            auto *ki = external_state ? ggml_view_3d(c, state_k, 256, 2, n, state_k->nb[1], state_k->nb[2], 0)
                                      : ggml_new_tensor_3d(c, GGML_TYPE_F16, 256, 2, n);
            auto *vi = external_state ? ggml_view_3d(c, state_v, 256, 2, n, state_v->nb[1], state_v->nb[2], 0)
                                      : ggml_new_tensor_3d(c, GGML_TYPE_F16, 256, 2, n);
            if (external_state)
                for (auto *t : {ki, vi}) check(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS,"external KV active view");
            auto *qi = ggml_new_tensor_3d(c, GGML_TYPE_F32, 256, 12, 2);
            ggml_set_input(qi);
            if (!external_state)
                for (auto *t : {ki, vi}) ggml_set_input(t);
            auto *attention = portable_detail::qsa_dense_attention(c, ki, vi, qi, output);
            ggml_set_output(attention);
            auto *graph = ggml_new_graph(c);
            ggml_build_forward_expand(graph, output);
            ggml_build_forward_expand(graph, attention);
            std::vector<PortablePlacement> placement;
            size_t bytes = 0;
            portable_detail::Scheduler temporary{nullptr, ggml_backend_sched_free};
            auto &sched = reuse ? retained_sched : temporary;
            auto &execution_cpu = reuse ? retained_cpu : cpu;
            portable_detail::schedule_reusing(graph, "qsa-selection", options, execution_cpu, placement, bytes, sched);
            if (reuse) {
                if (!key_buffer) {
                    key_buffer = ki->buffer;
                    value_buffer = vi->buffer;
                }
                check(ki->buffer == key_buffer && vi->buffer == value_buffer,
                      "selection exact shapes retain backing buffer identities");
                if (sweep == 1)
                    check(bytes == retained_buffers, "selection sweep retains bounded scheduler buffers");
                else
                    retained_buffers = bytes;
            }
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
