#pragma once
#include <cstddef>
#include <stdexcept>
namespace strata::core::portable_detail {
constexpr size_t gdn_input_blocks = 80;
constexpr size_t gdn_input_packed_bytes = gdn_input_blocks * 36;
constexpr size_t gdn_input_row_bytes = gdn_input_blocks * 34;
inline void gdn_input_require_layout(size_t rows, size_t weights, size_t input, size_t output) {
    if ((rows != 10240 && rows != 6144) || weights != rows * gdn_input_row_bytes ||
        input != gdn_input_packed_bytes || output != rows * sizeof(float))
        throw std::runtime_error("GDN input projection: unsupported shape or buffer size");
}
}
