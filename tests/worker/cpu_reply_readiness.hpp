#pragma once
// Offline zero-cost serializer over the existing pool's completion trace. No I/O in the pool.
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::worker::diagnostic {
struct ReplyReadiness {
    uint64_t bytes = 36, first_range = UINT64_MAX, first_prefix = 0, first_row = UINT64_MAX, first_wire_row = 0;
    uint64_t prefix_finish = 0, row_finish = 0, batch_finish = 0;
};
inline ReplyReadiness reply_readiness(const strata::kernels::cpu::ExpertPool::NativeReadiness& trace,
                                      int jobs, int H, const std::vector<int>& entry_job, uint64_t begin) {
    const auto check = [](bool ok) { if (!ok) throw std::runtime_error("invalid reply readiness trace"); };
    check(jobs > 0 && jobs <= strata::kernels::cpu::ExpertPool::kMaxSplitMulti && H > 0 &&
          trace.count > 0 && trace.count <= strata::kernels::cpu::ExpertPool::kMaxTasks &&
          begin <= trace.down_start_ns && trace.down_start_ns <= trace.down_done_ns &&
          trace.down_done_ns <= trace.pool_done_ns);
    const int64_t dimensions = int64_t(jobs) * H;
    ReplyReadiness out;
    // Exact existing frame bytes; header is optimistically ready at pool start. 125 MB/s = 8 ns/B.
    out.prefix_finish = out.row_finish = begin + 36 * 8;
    for (int task = 0; task < trace.count; ++task) {
        const auto done = trace.tasks[task].done_ns;
        check(done >= trace.down_start_ns && done <= trace.down_done_ns);
        if (dimensions * task / trace.count < dimensions * (task + 1) / trace.count)
            out.first_range = std::min(out.first_range, done);
    }
    for (int job : entry_job) {
        if (job < 0) continue; // non-CPU entry: absent on this peer's wire
        check(job < jobs);
        uint64_t ready = 0;
        int covered = 0;
        out.prefix_finish += 2 * 8; // next ascending entry index
        for (int task = 0; task < trace.count; ++task) {
            const int64_t lo = std::max(dimensions * task / trace.count, int64_t(job) * H);
            const int64_t hi = std::min(dimensions * (task + 1) / trace.count, int64_t(job + 1) * H);
            if (lo >= hi) continue;
            check(lo == int64_t(job) * H + covered);
            const auto done = trace.tasks[task].done_ns;
            if (!out.first_prefix) out.first_prefix = done;
            ready = std::max(ready, done);
            out.prefix_finish = std::max(out.prefix_finish, done) + uint64_t(hi - lo) * 4 * 8;
            covered += int(hi - lo);
        }
        check(covered == H);
        if (!out.first_wire_row) out.first_wire_row = ready;
        out.first_row = std::min(out.first_row, ready);
        const uint64_t bytes = 2 + uint64_t(H) * 4;
        out.bytes += bytes;
        out.row_finish = std::max(out.row_finish, ready) + bytes * 8;
    }
    check(out.first_prefix && out.first_wire_row);
    out.batch_finish = trace.pool_done_ns + out.bytes * 8;
    check(out.prefix_finish <= out.batch_finish && out.row_finish <= out.batch_finish);
    return out;
}
} // namespace strata::worker::diagnostic
