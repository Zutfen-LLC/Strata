#pragma once
#include "../../src/core/portable_graph.hpp"
#include "../../src/core/portable_qsa_graph.hpp"
#include "ggml-cpu.h"
#include "portable_qsa_fixture.hpp"
#include <numeric>
namespace qsa_fixture {
inline void selection_tests(PortableFfnOptions options) {
  for (int n : {2050, 2051, 2052, 2055, 2056})
    for (int mode = 0; mode < 3; ++mode) {
      std::vector<float> scores(size_t(n), 0);
      for (int i = 0; i < n; ++i)
        scores[size_t(i)] = i >= n / 4 * 4 ? 1e9f
                            : mode == 0
                                ? 0.0f
                                : float(mode == 1 ? (i / 4) % 7 : (n - i) / 4);
      portable_detail::Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
      std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
          ggml_init({2u << 20, nullptr, true}), ggml_free);
      auto *c = ctx.get();
      auto *input = ggml_new_tensor_1d(c, GGML_TYPE_F32, n);
      ggml_set_input(input);
      auto *output = portable_detail::qsa_selection(c, input);
      ggml_set_output(output);
      auto *graph = ggml_new_graph(c);
      ggml_build_forward_expand(graph, output);
      std::vector<PortablePlacement> placement;
      size_t bytes = 0;
      auto sched = portable_detail::schedule(graph, "qsa-selection", options,
                                             cpu, placement, bytes);
      if (n > 2051)
        ggml_backend_tensor_set(input, scores.data(), 0,
                                scores.size() * sizeof(float));
      check(ggml_backend_sched_graph_compute(sched.get(), graph) ==
                GGML_STATUS_SUCCESS,
            "selection compute");
      std::vector<float> actual(size_t(n), 0), expected(size_t(n), 0);
      ggml_backend_tensor_get(output, actual.data(), 0,
                              actual.size() * sizeof(float));
      std::vector<int> ids(size_t(n), 0);
      std::iota(ids.begin(), ids.end(), 0);
      std::sort(ids.begin(), ids.end(), [&](int a, int b) {
        return scores[size_t(a)] != scores[size_t(b)]
                   ? scores[size_t(a)] > scores[size_t(b)]
                   : a < b;
      });
      for (int i = 0; i < std::min(n, 2051); ++i)
        expected[size_t(ids[size_t(i)])] = 1;
      close(actual, expected, 0,
            "deterministic cell selection and tail boundary");
      for (const auto &p : placement)
        if (options.request_accelerator && !options.allow_cpu_fallback &&
            !p.metadata_only)
          check(p.backend == ggml_backend_name(options.accelerator) &&
                    p.fallback_reason.empty(),
                "strict selection placement");
    }
}
} // namespace qsa_fixture
