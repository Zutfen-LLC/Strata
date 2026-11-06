#pragma once
#include "strata/worker/gpu_wire.hpp"
#include <atomic>
#include <memory>

namespace strata::worker {
// Opt-in offline diagnostic snapshot of the actual metadata submitted to grouped CUDA.
struct GpuExecutionTrace {
    std::vector<int32_t> experts,asset_slots,start,dst,tok,original;
};
// Test seam substitutes only execution, never protocol validation/cache/session logic.
using GpuAssetReader = std::function<bool(const GpuAsset&, std::vector<uint8_t>&, std::string&)>;
class GpuExecutor {
public:
    virtual ~GpuExecutor() = default;
    virtual bool prepare(const GpuPlan&, uint64_t reserve, GpuCache&, GpuCounters&, std::string&) = 0;
    // In-process loading without a network or persistent asset cache.
    virtual bool prepare_source(const GpuPlan&, const GpuAssetReader&, GpuCounters&, std::string& err) {
        err="executor does not support direct local assets"; return false;
    }
    virtual bool infer(const GpuPlan&, const InferRequest&, InferReply&, std::string&) = 0;
    virtual bool supports_rows() const { return false; }
    // A worker must never advertise native arithmetic for another executor.
    virtual bool native_q8() const { return false; }
    virtual bool infer_rows(const GpuPlan&, const InferRequest&, EntryRowsReply&, std::string& err) {
        err="CUDA executor does not support entry rows"; return false;
    }
    virtual bool infer_rows_diagnostic(const GpuPlan&,const InferRequest&,EntryRowsReply&,GpuExecutionTrace&,std::string& err) {
        err="CUDA executor does not support diagnostic metadata";return false;
    }
    virtual void release(GpuCounters&) = 0;
};
std::unique_ptr<GpuExecutor> create_cuda_gpu_executor(int device);
struct GpuWorkerOptions {
    TlsConfig tls;
    GpuPlan plan; // trusted operator metadata, immutable for the life of this process
    std::string cache_root;
    std::string bind_ip = "127.0.0.1";
    std::vector<std::string> allow_peer_ips; // exact numeric coordinator addresses
    bool accepted_tcp_nodelay = false; // explicit transport experiment; default unchanged
    uint64_t cache_budget = 24ull << 30;
    uint64_t runtime_reserve = 1ull << 30;
    int deadline_ms = 10000, setup_ms = 600000;
    // Opt-in idle-session budget; 0 keeps the deadline-only default. When set, it
    // bounds only the wait for the first byte of the next request frame
    // after READY. Once any byte of a frame arrives, deadline_ms alone
    // bounds that frame's remaining header and payload, and sends and the TLS
    // handshake keep deadline_ms. setup_ms still bounds whole setup separately.
    int idle_ms = 0;
    bool vulkan = false; // reserve must be zero
    bool native_q8 = false; // explicit Vulkan native numerical contract
};
// Upper bound for the opt-in idle budget, matching the setup budget's bound.
inline constexpr int kGpuWorkerIdleMaxMs = 86400000;
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
    std::vector<uint32_t> allowed_peers_;
    uintptr_t listener_ = ~(uintptr_t)0;
    std::atomic<bool> stop_{false};
    bool negotiated_ = false, setup_ = false, ready_ = false, rows_ = false;
    uint64_t session_ = 0, sequence_ = 0, session_serial_ = 0;
    uint64_t setup_started_ = 0;
    std::vector<uint8_t> missing_, upload_;
    size_t uploading_ = SIZE_MAX;
};
} // namespace strata::worker
