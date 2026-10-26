#pragma once
#include "strata/worker/gpu_worker.hpp"

namespace strata::worker {
// Explicit in-process experiment only; not wired into any worker/listener/default.
// One trusted Q8_0 expert, H2560/FF640, one token and one active indexed entry.
// Hybrid host activation boundaries; Vulkan owns weights and all three matvecs.
// Single caller, no concurrent prepare/infer/release. reserve must be zero.
std::unique_ptr<GpuExecutor> create_vulkan_gpu_executor(int device);
#ifdef STRATA_AMD_EXECUTOR_HOST_TEST
// CPU graph substitution for host lifecycle tests only; never a Vulkan fallback.
std::unique_ptr<GpuExecutor> create_vulkan_executor_cpu_test();
#endif
} // namespace strata::worker
