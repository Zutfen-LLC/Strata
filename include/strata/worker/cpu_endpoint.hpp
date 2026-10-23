#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::worker {

inline constexpr size_t kMaxCpuWorkers = 8;

/// Placement input for one CPU worker. `capacity` is a hard maximum number of resident expert blobs (zero
/// means unbounded); `speed` is a relative execution weight used to send hotter profile entries to faster
/// workers. Endpoint order is significant and is also the deterministic reduction order.
struct CpuEndpoint {
    std::string host;
    uint16_t port = 0;
    uint64_t capacity = 0;
    double speed = 1.0;
};

} // namespace strata::worker
