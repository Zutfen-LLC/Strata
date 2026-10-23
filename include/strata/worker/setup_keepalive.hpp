#pragma once
#include "strata/worker/cpu_group.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>

namespace strata::worker {
// Owns the CPU client's socket exclusively while the engine acquires its network-GPU subset.
// STATUS keeps the existing per-frame idle receive deadline alive. It does not renew the CPU
// resident arena's TTL: an expired/mismatched session cancels acquisition and fails setup.
// No other thread may call this tier until finish()/stop() has joined the status thread.
class CpuSetupKeepalive {
public:
    explicit CpuSetupKeepalive(ExpertWorkerTier& tier, int interval_ms = 50)
        : tier_(&tier), sessions_{tier.stats().session}, interval_ms_(std::max(1, interval_ms)),
          thread_([this] { run(); }) {}
    explicit CpuSetupKeepalive(CpuWorkerGroup& group, int interval_ms = 50)
        : group_(&group), interval_ms_(std::max(1, interval_ms)),
          thread_([this] { run(); }) {}
    ~CpuSetupKeepalive() { stop(); }
    CpuSetupKeepalive(const CpuSetupKeepalive&) = delete;
    CpuSetupKeepalive& operator=(const CpuSetupKeepalive&) = delete;
    bool failed() const { return failed_.load(std::memory_order_acquire); }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
    // The final serialized status catches expiry between the last periodic probe and GPU READY.
    bool finish(std::string& err) {
        stop();
        if (!failed()) probe();
        if (failed()) err = why_;
        return !failed();
    }
private:
    void probe() {
        if (group_ != nullptr && sessions_.empty()) {
            for (size_t i = 0; i < group_->size(); ++i) sessions_.push_back(group_->stats(i).session);
        }
        for (size_t i = 0; i < sessions_.size(); ++i) {
            WorkerStatusInfo status;
            std::string err;
            const bool ok = group_ != nullptr ? group_->status(i, status, err) : tier_->status(status, err);
            if (!ok || !sessions_[i] || status.session != sessions_[i] || status.ready != 1) {
                why_ = "CPU worker session lost during selective CUDA acquisition";
                if (group_ != nullptr) why_ += " (worker " + std::to_string(i) + ")";
                if (!err.empty()) why_ += ": " + err;
                failed_.store(true, std::memory_order_release);
                return;
            }
        }
    }
    void run() {
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_) return;
            }
            probe();
            if (failed()) return;
            std::unique_lock<std::mutex> lock(mutex_);
            if (wake_.wait_for(lock, std::chrono::milliseconds(interval_ms_), [this] { return stopping_; })) return;
        }
    }
    ExpertWorkerTier* tier_ = nullptr;
    CpuWorkerGroup* group_ = nullptr;
    std::vector<uint64_t> sessions_;
    const int interval_ms_;
    std::atomic<bool> failed_{false};
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    std::string why_;
    std::thread thread_; // starts only after every field it uses has been initialized
};
} // namespace strata::worker
