#pragma once
#include "ggml.h"
#include <algorithm>

namespace strata::core::portable_detail {
// Retain FP16 K/V, but perform attention arithmetic in F32. Explicit
// per-query-head products avoid the pinned Vulkan batched 12-column path,
// which produces wrong logits after column 7 on the two-cell regression.
inline ggml_tensor *qsa_dense_attention(ggml_context *c, ggml_tensor *ki, ggml_tensor *vi, ggml_tensor *qi,
                                        ggml_tensor *selected) {
    const int64_t cells = ki->ne[2];
    auto *mask = ggml_log(c, selected); // exact 0 / -infinity
    auto *key = ggml_cast(c, ggml_cont(c, ggml_permute(c, ki, 0, 2, 1, 3)), GGML_TYPE_F32);
    auto *value = ggml_cast(c, ggml_cont(c, ggml_permute(c, vi, 1, 2, 0, 3)), GGML_TYPE_F32);
    ggml_tensor *out = nullptr;
    for (int64_t h = 0; h < 24; ++h) {
        const size_t kv = size_t(h / 12);
        auto *kh = ggml_view_2d(c, key, 256, cells, key->nb[1], kv * key->nb[2]);
        auto *qh = ggml_view_1d(c, qi, 256, size_t(h) * 256 * sizeof(float));
        auto *logits = ggml_mul_mat(c, kh, qh);
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        auto *prob = ggml_soft_max_ext(c, logits, mask, 1.0f / 16.0f, 0.0f);
        auto *vh = ggml_view_2d(c, value, cells, 256, value->nb[1], kv * value->nb[2]);
        auto *head = ggml_mul_mat(c, vh, prob);
        ggml_mul_mat_set_prec(head, GGML_PREC_F32);
        out = out ? ggml_concat(c, out, head, 1) : head;
    }
    return ggml_reshape_1d(c, out, 6144);
}
// Expand complete-block and moving spare scores to cells, then add the
// finite incomplete-tail bias. The spare participates in the dot product.
inline ggml_tensor *qsa_cell_scores(ggml_context *c, ggml_tensor *pooled, ggml_tensor *query, ggml_tensor *mapping,
                                    ggml_tensor *bias) {
    const int64_t blocks = pooled->ne[1], cells = mapping->ne[0];
    auto *perhead = ggml_relu(c, ggml_mul_mat(c, pooled, query));
    auto *sum = ggml_cont(c, ggml_view_1d(c, perhead, blocks, 0));
    for (int64_t h = 1; h < query->ne[1]; ++h)
        sum = ggml_add(c, sum, ggml_view_1d(c, perhead, blocks, size_t(h * blocks) * sizeof(float)));
    return ggml_add(c, ggml_reshape_1d(c, ggml_get_rows(c, ggml_reshape_2d(c, sum, 1, blocks), mapping), cells),
                    bias);
}
// TOP_K supplies only the cutoff VALUE. Its arbitrary tie membership/order is
// discarded. CUMSUM resolves equal-cutoff cells by ascending cell ID exactly,
// without perturbing floating scores or assuming a backend sort is stable.
inline ggml_tensor *qsa_selection(ggml_context *c, ggml_tensor *scores) {
    const int64_t n = scores->ne[0], width = std::min<int64_t>(n, 2051);
    auto *ones = ggml_fill(c, ggml_new_tensor_1d(c, GGML_TYPE_F32, n), 1.0f);
    if (n <= width)
        return ones;
    auto *ids = ggml_top_k(c, scores, width);
    auto *chosen = ggml_reshape_1d(c, ggml_get_rows(c, ggml_reshape_2d(c, scores, 1, n), ids), width);
    auto *minimum_id = ggml_argmax(c, ggml_neg(c, chosen));
    auto *cutoff = ggml_reshape_1d(c, ggml_get_rows(c, ggml_reshape_2d(c, chosen, 1, width), minimum_id), 1);
    auto *diff = ggml_sub(c, scores, cutoff);
    auto *greater = ggml_step(c, diff);
    auto *equal = ggml_sub(c, ones, ggml_abs(c, ggml_sgn(c, diff)));
    auto *room = ggml_add1(c, ggml_neg(c, ggml_sum(c, greater)),
                           ggml_fill(c, ggml_new_tensor_1d(c, GGML_TYPE_F32, 1), float(width) + 0.5f));
    auto *keep_equal = ggml_step(c, ggml_sub(c, ggml_repeat(c, room, equal), ggml_cumsum(c, equal)));
    return ggml_add(c, greater, ggml_mul(c, equal, keep_equal));
}
} // namespace strata::core::portable_detail
