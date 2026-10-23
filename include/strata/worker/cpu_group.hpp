#pragma once

#include "strata/worker/cpu_endpoint.hpp"
#include "strata/worker/expert_tier.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::worker {

/// Split `complement` into disjoint, exhaustive partitions. Profile entries are ranked hottest first.
/// Memory quotas follow capacity; within those quotas the greedy hotness assignment follows speed and
/// spreads ties across layers. Unranked entries are striped by layer. The same inputs always produce the
/// same bitmaps.
bool place_cpu_partitions(const std::vector<uint8_t>& complement, uint16_t n_layers, uint16_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& profile,
                          const std::vector<CpuEndpoint>& endpoints,
                          std::vector<std::vector<uint8_t>>& partitions, std::string& err);

/// Engine-owned CPU worker collection. ExpertWorkerTier already owns a persistent communication thread,
/// so post() publishes every peer's copied request before finish() waits for any peer. Results remain
/// separate until every active peer succeeds; callers merge them in endpoint order.
class CpuWorkerGroup {
public:
    CpuWorkerGroup();
    ~CpuWorkerGroup();
    CpuWorkerGroup(const CpuWorkerGroup&) = delete;
    CpuWorkerGroup& operator=(const CpuWorkerGroup&) = delete;

    static bool validate(const std::vector<ExpertWorkerTier::Open>& options,
                         const std::vector<uint8_t>& complement, std::string& err);
    bool open(const std::vector<ExpertWorkerTier::Open>& options,
              const std::vector<uint8_t>& complement, std::string& err);
    void close();
    bool holds(int64_t layer, int32_t expert) const;
    bool ready() const;
    bool dead() const;
    bool post(int64_t layer, const float* x, const int32_t* ids, const float* weights,
              const uint8_t* owned, int n_tok, int k, std::string& err);
    bool finish(std::string& err);
    size_t size() const;
    bool posted() const;
    bool active(size_t index) const;
    bool served(size_t index) const;
    const uint8_t* mask(size_t index) const;
    const float* partials(size_t index) const;
    const ExpertWorkerTier::Stats& stats(size_t index) const;
    bool status(size_t index, WorkerStatusInfo& out, std::string& err);
    void report(std::string& out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::worker
