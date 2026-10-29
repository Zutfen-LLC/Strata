#pragma once
#include "strata/core/portable_model.hpp"
#include <stdexcept>

namespace strata::core::portable_detail {
inline void validate_dense_topology(const PortableModelOptions& options) {
    if (options.dense_topology == PortableDenseTopology::primary) {
        if (options.accelerators.size() != 2)
            throw std::runtime_error("portable model: primary dense topology requires two accelerators");
    } else if (options.dense_topology != PortableDenseTopology::split) {
        throw std::runtime_error("portable model: unknown dense topology");
    } else if (options.accelerators.size() == 2 && (options.split < 1 || options.split >= 48)) {
        throw std::runtime_error("portable model: invalid contiguous split");
    }
}
// Shared by execution, state/weight preflight, final ledgers and output head.
inline size_t dense_device(const PortableModelOptions& options, size_t layer) {
    return options.dense_topology == PortableDenseTopology::split &&
           options.accelerators.size() == 2 && layer >= options.split ? 1 : 0;
}
} // namespace strata::core::portable_detail
