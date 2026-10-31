#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeHc;
// Standalone component and experimental serial bridge. Numerical contract v1;
// no universal rsqrt bit identity or full-HC qualification.
// Normal-or-zero arithmetic domain only. No Q8/REQ8/V3/pending-write fallback.
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* effective_spv,size_t metadata_budget=65536);
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*);
void strata_vk_native_hc_destroy(StrataVkNativeHc*) noexcept;
// Raw BF16 matrix packed in I32; all other tensors F32. Fixed H2560/HC4/T1..8.
// Synchronous call; host domain preflight is diagnostic-only and bounded (<1 MiB).
// Tensors must be disjoint contiguous spans on the same Vulkan device.
void strata_vk_native_hc_run(StrataVkNativeHc*,const ggml_tensor* residual,const ggml_tensor* norm_weight,
 const ggml_tensor* bf16_weight,ggml_tensor* normalized,ggml_tensor* rs,ggml_tensor* inject,uint32_t tokens);
}
