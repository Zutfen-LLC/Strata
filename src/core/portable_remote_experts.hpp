#pragma once
#include "strata/core/portable_ffn.hpp"
#include "strata/worker/gpu_client.hpp"
#include <memory>
#include <vector>

namespace strata::core::portable_detail {
// Private adapter for the existing unweighted portable FFN row boundary.
// The caller owns the source and fallback for the adapter's full lifetime.
class RemoteExperts {
public:
    using Complement = std::function<bool(int64_t,const std::vector<PortableRoute>&,
        const std::vector<uint8_t>&,std::vector<float>&,std::string&)>;
    using SourceGuard = std::function<bool(std::string&)>;
    RemoteExperts() = default;
    ~RemoteExperts() = default;
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;
    bool open(worker::GpuWorkerTier::Open, const worker::GpuPlan& expected,
              const std::vector<worker::GpuPlan>& local, size_t required_k, Complement, SourceGuard, std::string&);
    bool run(int64_t, const PortableRoute&, std::vector<float>&, std::string&);
    bool run_batch(int64_t, const std::vector<PortableRoute>&, std::vector<float>&, std::string&);
    const worker::GpuWorkerTier::Stats& stats() const { return tier_.stats(); }
    uint64_t cpu_entries() const { return cpu_entries_; }
    bool usable() const { return ready_ && tier_.ready() && !tier_.dead(); }
    bool status(worker::GpuCounters& out,std::string& error) { return tier_.status(out,error); }
    bool close_orderly(std::string& error) { const bool ok=tier_.close_orderly(error); ready_=false; return ok; }
private:
    worker::GpuPlan plan_;
    worker::GpuWorkerTier tier_;
    Complement complement_;
    SourceGuard source_guard_;
    bool ready_=false;
    uint64_t cpu_entries_=0;
    size_t required_k_=0;
};
} // namespace strata::core::portable_detail
