#pragma once
#include "ggml-backend.h"
#include <cstddef>

namespace strata::core::portable_detail {
// Defined only inside the pin-checked Vulkan translation unit. No Vulkan handle
// escapes, no backend/device is created, and the borrowed backend must outlive it.
struct StrataVkLogitsGuard;
StrataVkLogitsGuard *strata_vk_logits_guard_create(ggml_backend_t backend, size_t device_budget);
bool strata_vk_logits_guard_check(StrataVkLogitsGuard *, const ggml_tensor *logits);
void strata_vk_logits_guard_destroy(StrataVkLogitsGuard *) noexcept;
size_t strata_vk_logits_guard_device_bytes(const StrataVkLogitsGuard *);
size_t strata_vk_logits_guard_host_bytes(const StrataVkLogitsGuard *);
} // namespace strata::core::portable_detail
