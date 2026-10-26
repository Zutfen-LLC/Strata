#pragma once
#include "strata/worker/expert_tier.hpp"
#include "strata/worker/cpu_endpoint.hpp"
#include <memory>

namespace strata::worker {
// Fixed, engine-owned CPU partitions. Each member retains its existing socket/request thread.
// Single-endpoint legacy mode continues to use ExpertWorkerTier directly.
class ExpertWorkerGroup {
public:
    using Open = ExpertWorkerTier::Open;
    using Stats = ExpertWorkerTier::Stats;
    ExpertWorkerGroup() = default;
    ~ExpertWorkerGroup();
    ExpertWorkerGroup(const ExpertWorkerGroup&) = delete;
    ExpertWorkerGroup& operator=(const ExpertWorkerGroup&) = delete;
    static bool validate(const std::vector<Open>&, const std::vector<uint8_t>& expected, std::string&);
    // delegated is an immutable GPU execution subset of expected; setup partitions stay unchanged.
    bool open(const std::vector<Open>&, const std::vector<uint8_t>& expected, std::string&,
              const std::vector<uint8_t>& delegated = {});
    static bool validate_delegation(const std::vector<Open>&, const std::vector<uint8_t>& expected,
                                    const std::vector<uint8_t>& delegated, std::string&);
    bool executes(int64_t layer, int32_t expert) const;
    bool deferred_rows() const { return deferred_rows_; }
    bool rows_ready() const { return rows_ready_ && ready() && !posted_; }
    const std::vector<float>& entry_rows() const { return rows_; } // dense original-entry rows after rows_ready()
    const std::vector<uint8_t>& entry_mask() const { return owned_; }
    bool probe_idle(std::string&); // serialized with the existing keeper; no second socket owner
    void close();
    bool holds(int64_t layer, int32_t expert) const;
    bool ready() const { return ready_ && !dead(); }
    bool dead() const { return failed_.load(std::memory_order_acquire); }
    bool posted() const { return posted_; }
    bool served() const { return served_; }
    bool post(int64_t, const float*, const int32_t*, const float*, const uint8_t*, int, int, std::string&);
    bool finish(std::string&);
    const float* partials() const { return served_ && !deferred_rows_ ? combined_.data() : nullptr; }
    size_t size() const { return peers_.size(); }
    const Stats& stats() const { return stats_; }
    const Stats& peer_stats(size_t i) const { return peers_.at(i)->stats(); } // only between windows
    bool peer_status(size_t i, WorkerStatusInfo& out, std::string& err);
    void report(std::string&) const;
private:
    void keepalive();
    void fail(const std::string&);
    std::string error() const;
    std::vector<std::unique_ptr<ExpertWorkerTier>> peers_;
    std::vector<CpuEndpoint> endpoints_;
    std::vector<std::vector<uint8_t>> masks_;
    std::vector<float> combined_, rows_, weights_;
    std::vector<uint8_t> owned_, delegated_;
    std::vector<int> owners_;
    std::vector<uint8_t> active_;
    uint16_t layers_ = 0, experts_ = 0, H_ = 0;
    int T_ = 0, K_ = 0, interval_ms_ = 50;
    bool entry_rows_=false, deferred_rows_=false, rows_ready_=false;
    strata::kernels::cpu::NativeFmt packed_fmt_{};
    bool packed_requested_=false;
    uint64_t pack_calls_=0, quantized_rows_=0, pack_ns_=0;
    std::string pack_reason_;
    double row_reduce_ms_=0;
    uint64_t posted_ns_ = 0;
    bool ready_ = false, posted_ = false, served_ = false;
    Stats stats_;
    std::atomic<bool> failed_{false};
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false, window_ = false;
    std::string reason_;
    std::thread keeper_;
};
} // namespace strata::worker
