#pragma once
#include "native_component_host_bound.hpp"
#include <vector>
#include <utility>
#include <limits>
namespace strata::core::portable_detail {
inline std::vector<std::pair<NativeComponentHostKind, size_t>> bound_calls;
inline bool bound_overflow = false;
size_t native_component_factory_host_bound(NativeComponentHostKind kind, size_t bytes) {
    bound_calls.emplace_back(kind, bytes);
    return bound_overflow ? std::numeric_limits<size_t>::max() : 100 * (1 + static_cast<size_t>(kind)) + bytes;
}
}
