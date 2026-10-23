// include/strata/worker/rpc_timing.hpp - opt-in, bounded CPU-worker RPC timing.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::worker {

struct RpcRequestShape { uint32_t entries = 0, distinct_experts = 0; };
RpcRequestShape rpc_request_shape(const std::vector<uint8_t>& mask, const std::vector<uint16_t>& ids);

// Log2 microsecond buckets keep the diagnostic bounded. Durations are monotonic and local to one host;
// client and worker aggregates must never be subtracted from each other.
struct RpcLatencyDistribution {
    static constexpr size_t kBuckets = 22; // <=1 us through <=1,048,576 us, then overflow
    uint64_t count = 0, total_ns = 0, max_ns = 0;
    std::array<uint64_t, kBuckets> buckets{};

    void add(uint64_t ns);
    void reset();
    uint64_t percentile_us(unsigned percentile) const;
};

enum class RpcTimingPhase : uint8_t {
    ClientPrepare,
    ClientQueueWake,
    ClientEncodeSend,
    ClientReceiveDecode,
    ClientFinishWait,
    ClientResultMerge,
    WorkerDecodeValidate,
    WorkerQuantPrepare,
    WorkerPool,
    WorkerWeightedReduce,
    WorkerEncodeSend,
    WorkerRequest,
    Count,
};

class RpcTimingCounters {
public:
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }
    void reset(); // preserves enabled/disabled state

    void add_phase(RpcTimingPhase phase, uint64_t ns);
    void finish_call(uint8_t tokens, uint32_t entries, uint32_t experts, uint64_t request_bytes,
                     uint64_t reply_bytes, bool success);

    uint64_t calls() const { return calls_; }
    uint64_t successes() const { return successes_; }
    uint64_t failures() const { return failures_; }
    uint64_t request_bytes() const { return request_bytes_; }
    uint64_t reply_bytes() const { return reply_bytes_; }
    uint64_t tokens() const { return tokens_; }
    uint64_t entries() const { return entries_; }
    uint64_t experts() const { return experts_; }
    const std::array<uint64_t, 9>& token_shapes() const { return token_shapes_; }
    const RpcLatencyDistribution& phase(RpcTimingPhase phase) const {
        return phases_[(size_t) phase];
    }

    // One bounded aggregate line. Phase distributions are p50/p95/p99/max, never individual RPCs.
    std::string report(const char* side) const;

private:
    bool enabled_ = false;
    uint64_t calls_ = 0, successes_ = 0, failures_ = 0;
    uint64_t request_bytes_ = 0, reply_bytes_ = 0, tokens_ = 0, entries_ = 0, experts_ = 0;
    std::array<uint64_t, 9> token_shapes_{}; // index 0 is invalid/empty; 1..8 are protocol T
    std::array<RpcLatencyDistribution, (size_t) RpcTimingPhase::Count> phases_{};
};

static_assert(sizeof(RpcTimingCounters) < 4096, "RPC timing must remain fixed and small");

} // namespace strata::worker
