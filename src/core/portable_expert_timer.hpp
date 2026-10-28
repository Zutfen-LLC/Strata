#pragma once
#include <chrono>
#include <cstdint>
namespace strata::core::portable_detail {
// Two optional host-call boundaries only. No GPU synchronization or worker timer.
// Clock parameter permits deterministic capture/disabled-path tests.
template<class Clock = std::chrono::steady_clock> class ScopedExpertTimer {
    double& elapsed_;
    bool enabled_;
    typename Clock::time_point start_{};
public:
    ScopedExpertTimer(bool enabled, double& elapsed, uint64_t& calls) : elapsed_(elapsed), enabled_(enabled) {
        if (enabled_) { start_ = Clock::now(); ++calls; }
    }
    ~ScopedExpertTimer() {
        if (enabled_) elapsed_ += std::chrono::duration<double, std::milli>(Clock::now()-start_).count();
    }
    ScopedExpertTimer(const ScopedExpertTimer&) = delete;
    ScopedExpertTimer& operator=(const ScopedExpertTimer&) = delete;
};
} // namespace strata::core::portable_detail
