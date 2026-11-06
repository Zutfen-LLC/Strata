#pragma once
#include "strata/core/portable_ffn.hpp"
#include "strata/worker/gpu_client.hpp"
#include "strata/worker/gpu_group.hpp"
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
    bool open(std::vector<worker::GpuWorkerTier::Open>, const std::vector<worker::GpuPlan>& expected,
              const std::vector<worker::GpuPlan>& local, size_t required_k, Complement, SourceGuard, std::string&);
    bool run(int64_t, const PortableRoute&, std::vector<float>&, std::string&);
    bool run_batch(int64_t, const std::vector<PortableRoute>&, std::vector<float>&, std::string&);
    worker::GpuWorkerTier::Stats stats() const { return group_ ? group_->stats() : tier_.stats(); }
    uint64_t complement_entries() const { return complement_entries_; }
    bool usable() const { return ready_ && !terminal_ && (group_ ? group_->ready() && !group_->dead() : tier_.ready() && !tier_.dead()); }
    bool status(worker::GpuCounters& out,std::string& error);
    std::vector<worker::GpuPeerSnapshot> snapshots() const;
    bool close_orderly(std::string& error) { const bool ok=group_ ? group_->close_orderly(error) : tier_.close_orderly(error); ready_=false; if(!ok){terminal_=true;if(group_)group_->close();else tier_.close();} return ok; }
private:
    worker::GpuPlan plan_;
    worker::GpuPeerSnapshot single_snapshot_;
    worker::GpuWorkerTier tier_;
    std::unique_ptr<worker::GpuWorkerGroup> group_;
    Complement complement_;
    SourceGuard source_guard_;
    bool ready_=false, terminal_=false;
    uint64_t complement_entries_=0;
    size_t required_k_=0;
};
} // namespace strata::core::portable_detail
