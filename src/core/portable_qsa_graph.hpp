#pragma once
#include "ggml.h"
#include <algorithm>

namespace strata::core::portable_detail {
// TOP_K supplies only the cutoff VALUE. Its arbitrary tie membership/order is
// discarded. CUMSUM resolves equal-cutoff cells by ascending cell ID exactly,
// without perturbing floating scores or assuming a backend sort is stable.
inline ggml_tensor *qsa_selection(ggml_context *c, ggml_tensor *scores) {
  const int64_t n = scores->ne[0], width = std::min<int64_t>(n, 2051);
  auto *ones = ggml_fill(c, ggml_new_tensor_1d(c, GGML_TYPE_F32, n), 1.0f);
  if (n <= width)
    return ones;
  auto *ids = ggml_top_k(c, scores, width);
  auto *chosen = ggml_reshape_1d(
      c, ggml_get_rows(c, ggml_reshape_2d(c, scores, 1, n), ids), width);
  auto *minimum_id = ggml_argmax(c, ggml_neg(c, chosen));
  auto *cutoff = ggml_reshape_1d(
      c, ggml_get_rows(c, ggml_reshape_2d(c, chosen, 1, width), minimum_id), 1);
  auto *diff = ggml_sub(c, scores, cutoff);
  auto *greater = ggml_step(c, diff);
  auto *equal = ggml_sub(c, ones, ggml_abs(c, ggml_sgn(c, diff)));
  auto *room = ggml_add1(c, ggml_neg(c, ggml_sum(c, greater)),
                         ggml_fill(c, ggml_new_tensor_1d(c, GGML_TYPE_F32, 1),
                                   float(width) + 0.5f));
  auto *keep_equal = ggml_step(
      c, ggml_sub(c, ggml_repeat(c, room, equal), ggml_cumsum(c, equal)));
  return ggml_add(c, greater, ggml_mul(c, equal, keep_equal));
}
} // namespace strata::core::portable_detail
