#pragma once
#include "strata/worker/gpu_client.hpp"
#include "strata/worker/gpu_endpoint.hpp"
#include <memory>

namespace strata::worker {
// Engine-owned bounded group. One communication thread owns each tier/socket,
// including idle STATUS. The caller posts copied inputs, then finishes all work
// before reading configuration-ordered partials/masks or starting another window.
class GpuWorkerGroup {
public:
    GpuWorkerGroup();
    ~GpuWorkerGroup();
    GpuWorkerGroup(const GpuWorkerGroup&) = delete;
    GpuWorkerGroup& operator=(const GpuWorkerGroup&) = delete;
    static bool validate(const std::vector<GpuWorkerTier::Open>&, std::string&);
    bool open(const std::vector<GpuWorkerTier::Open>&, std::string&);
    void close(); // joins before closing any tier; never races descriptor reuse
    bool holds(int64_t layer, int32_t expert) const;
    bool ready() const;
    bool dead() const;
    bool post(int64_t, const float*, const int32_t*, const float*, const uint8_t*, int, int, std::string&);
    bool finish(std::string&);
    size_t size() const;
    bool active(size_t) const;
    const uint8_t* mask(size_t) const;
    const float* partials(size_t index = 0) const;
    void report(std::string&) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::worker
