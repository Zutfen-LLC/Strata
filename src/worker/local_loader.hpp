#pragma once
// Internal, synchronous batch seam. The caller validates placement, opens source descriptors and
// supplies disjoint destinations in its ONE final arena. No task owns another expert's storage.
#include "strata/worker/local_source.hpp"

namespace strata::worker {
struct LocalLoadTask {
    uint16_t layer, expert;
    uint8_t* dst;
    std::array<bool, 3> present;
};
struct LocalLoadResult {
    uint16_t status = ST_OK;
    std::string error;
    uint64_t bytes = 0, verified_bytes = 0, read_ns = 0, verify_ns = 0;
    size_t completed = 0;
};
// At most 32 tasks and 4 lanes, including the caller. No payload staging or outstanding work on return.
// cancelled may be called concurrently and must be thread-safe; no arena/file teardown until return.
LocalLoadResult load_local_parallel(const LocalSource& source,
    const strata::kernels::cpu::NativeFmt& fmt, const std::vector<LocalLoadTask>& tasks,
    int threads, const std::function<bool()>& cancelled);
} // namespace strata::worker
