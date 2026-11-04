#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativePleGate;
// Standalone bounded observation. It neither advances PLE history nor commits a token.
StrataVkNativePleGate* strata_vk_native_ple_gate_create(ggml_backend_t,const char* effective_spv,size_t budget);
void strata_vk_native_ple_gate_destroy(StrataVkNativePleGate*) noexcept;
size_t strata_vk_native_ple_gate_bytes(const StrataVkNativePleGate*);
uint64_t strata_vk_native_ple_gate_calls(const StrataVkNativePleGate*) noexcept;
// Contiguous Vulkan spans: key/query F32[4,2560], gate F32[4], diagnostic F32[4,4].
// The read-only inputs may overlap; each output must be disjoint from all other spans.
// Status zero means this bounded finite-normal candidate completed, not CUDA intrinsic parity.
uint32_t strata_vk_native_ple_gate_run(StrataVkNativePleGate*,const ggml_tensor* key,const ggml_tensor* query,
                                      ggml_tensor* gate,ggml_tensor* diagnostic);
}
