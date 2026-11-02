#pragma once
#include <cstddef>
#include <stdexcept>
namespace strata::core::portable_detail {
constexpr size_t qsa_qkv_blocks = 80;
constexpr size_t qsa_qkv_packed_bytes = qsa_qkv_blocks * 36;
constexpr size_t qsa_qkv_row_bytes = qsa_qkv_blocks * 34;
inline void qsa_qkv_require_layout(size_t rows, size_t weights, size_t input, size_t output) {
    if ((rows != 12288 && rows != 512) || weights != rows * qsa_qkv_row_bytes ||
        input != qsa_qkv_packed_bytes || output != rows * sizeof(float))
        throw std::runtime_error("QSA QKV: unsupported shape or buffer size");
}
inline size_t qsa_qkv_query_row(size_t head, size_t component) {
    if (head >= 24 || component >= 256) throw std::runtime_error("QSA QKV: query row out of range");
    return head * 512 + component;
}
inline size_t qsa_qkv_gate_row(size_t head, size_t component) {
    return qsa_qkv_query_row(head, component) + 256;
}
}
