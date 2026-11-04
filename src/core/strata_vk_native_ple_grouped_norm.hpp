#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativePleGroupedNorm;
// Standalone bounded observation. It neither advances PLE history nor commits a token.
StrataVkNativePleGroupedNorm* strata_vk_native_ple_grouped_norm_create(ggml_backend_t,const char* effective_spv,size_t budget);
void strata_vk_native_ple_grouped_norm_destroy(StrataVkNativePleGroupedNorm*) noexcept;
size_t strata_vk_native_ple_grouped_norm_bytes(const StrataVkNativePleGroupedNorm*);
uint64_t strata_vk_native_ple_grouped_norm_calls(const StrataVkNativePleGroupedNorm*) noexcept;
// Contiguous Vulkan spans: input/gamma/output F32[4,2560], diagnostic F32[4,3].
// The read-only inputs may overlap; each output must be disjoint from all other spans.
// Status zero means this bounded finite-normal candidate completed, not CUDA intrinsic parity.
uint32_t strata_vk_native_ple_grouped_norm_run(StrataVkNativePleGroupedNorm*,const ggml_tensor* input,const ggml_tensor* gamma,
                                      ggml_tensor* output,ggml_tensor* diagnostic);
}
