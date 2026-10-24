#pragma once

namespace strata::core {
// The dispatch's two existing submissions. CPU post copies its buffers and starts
// its request thread; GPU group post queues copied requests. Only submission
// failure drains CPU here; success keeps the finish and GPU-then-CPU merge order.
template<class CpuPost, class GpuPost, class DrainCpu>
bool cpu_first_remote_post(CpuPost&& cpu_post, GpuPost&& gpu_post, DrainCpu&& drain_cpu) {
    if (!cpu_post()) return false;
    try {
        if (gpu_post()) return true;
    } catch (...) {
        drain_cpu();
        throw;
    }
    drain_cpu();
    return false;
}
} // namespace strata::core
