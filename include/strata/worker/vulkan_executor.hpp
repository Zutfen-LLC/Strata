#pragma once
#include "strata/worker/gpu_worker.hpp"
#include "ggml-backend.h"

namespace strata::worker {
// Explicit in-process experiment only; not wired into any worker/listener/default.
// Trusted resident Q8_0 experts, H2560/FF640, one token and indexed entries.
// Hybrid host activation boundaries; Vulkan owns weights and all three matvecs.
// Single caller, no concurrent prepare/infer/release. reserve must be zero.
// Budget bounds aggregate owned tensor buffers, including alignment/work tensors.
// It excludes host storage and backend/driver allocations; it is not total VRAM.
std::unique_ptr<GpuExecutor> create_vulkan_gpu_executor(int device, uint64_t tensor_budget = 16ull << 20);
// Borrowed backend must outlive the executor; serial use only. Strict Vulkan.
std::unique_ptr<GpuExecutor> create_vulkan_executor_on_backend(ggml_backend_t backend, uint64_t tensor_budget);
#ifdef STRATA_AMD_EXECUTOR_HOST_TEST
// CPU graph substitution for host lifecycle tests only; never a Vulkan fallback.
std::unique_ptr<GpuExecutor> create_vulkan_executor_cpu_test(uint64_t tensor_budget = 16ull << 20);
#endif
} // namespace strata::worker
