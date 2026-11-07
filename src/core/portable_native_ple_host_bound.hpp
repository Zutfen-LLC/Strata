#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace strata::core::portable_detail {
// Payload accounting only: excludes allocator bookkeeping and borrowed buffers.
constexpr size_t ple_host_add(size_t a, size_t b) {
    return b > std::numeric_limits<size_t>::max() - a
        ? throw std::overflow_error("PLE host bound addition") : a + b;
}
constexpr size_t ple_host_mul(size_t a, size_t b) {
    return a && b > std::numeric_limits<size_t>::max() / a
        ? throw std::overflow_error("PLE host bound multiplication") : a * b;
}
}
