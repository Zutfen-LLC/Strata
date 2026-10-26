#pragma once
#include "strata/worker/gpu_wire.hpp"
#include <atomic>
#include <memory>

namespace strata::worker {
// Test seam substitutes only execution, never protocol validation/cache/session logic.
class GpuExecutor {
public:
    virtual ~GpuExecutor() = default;
    virtual bool prepare(const GpuPlan&, uint64_t reserve, GpuCache&, GpuCounters&, std::string&) = 0;
    virtual bool infer(const GpuPlan&, const InferRequest&, InferReply&, std::string&) = 0;
    virtual bool supports_rows() const { return false; }
    virtual bool infer_rows(const GpuPlan&, const InferRequest&, EntryRowsReply&, std::string& err) {
        err="CUDA executor does not support entry rows"; return false;
    }
    virtual void release(GpuCounters&) = 0;
};
std::unique_ptr<GpuExecutor> create_cuda_gpu_executor(int device);
struct GpuWorkerOptions {
    GpuPlan plan; // trusted operator metadata, immutable for the life of this process
    std::string cache_root;
    uint64_t cache_budget = 24ull << 30;
    uint64_t runtime_reserve = 1ull << 30;
    int deadline_ms = 10000, setup_ms = 600000;
};
class GpuWorkerService {
public:
    GpuWorkerService(GpuWorkerOptions options, std::unique_ptr<GpuExecutor> executor);
    ~GpuWorkerService();
    bool listen(uint16_t& port, std::string& err);
    void run();
    void stop() { stop_.store(true); }
    const GpuCounters& counters() const { return counters_; }
private:
    void serve(uintptr_t fd);
    bool dispatch(const WireFrame&, WireFrame&, std::string&);
    void reset();
    GpuWorkerOptions options_;
    std::unique_ptr<GpuExecutor> executor_;
    GpuCache cache_;
    GpuCounters counters_;
    uintptr_t listener_ = ~(uintptr_t)0;
    std::atomic<bool> stop_{false};
    bool negotiated_ = false, setup_ = false, ready_ = false, rows_ = false;
    uint64_t session_ = 0, sequence_ = 0, session_serial_ = 0;
    uint64_t setup_started_ = 0;
    std::vector<uint8_t> missing_, upload_;
    size_t uploading_ = SIZE_MAX;
};
} // namespace strata::worker
