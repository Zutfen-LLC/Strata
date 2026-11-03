#pragma once
#include <cstddef>
#include <stdexcept>
namespace strata::core::portable_detail {
// 16,384 rows fit the Vulkan minimum storage-buffer range and dispatch limits.
constexpr size_t head_vocab_slice_rows = 16384;
constexpr size_t head_vocab_row_bytes = 80 * 34;
constexpr size_t head_vocab_packed_bytes = 80 * 36;
inline void head_vocab_require_layout(size_t rows, size_t weights, size_t input, size_t output) {
    if (!rows || rows > head_vocab_slice_rows || weights != rows * head_vocab_row_bytes ||
        input != head_vocab_packed_bytes || output != rows * sizeof(float))
        throw std::runtime_error("native head vocabulary: unsupported Q8 span");
}
} // namespace strata::core::portable_detail
