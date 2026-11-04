#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
// Standalone T1 observation. Exact disjoint spans: history F32[92160],
// normalized F32[10240], packed F16 weights I32[20480], hidden/gated
// F32[10240], candidate history F32[92160], and conv/result/pre/exp F32[10240].
// Input history is never committed by this component.
struct StrataVkNativePleConv;
StrataVkNativePleConv* strata_vk_native_ple_conv_create(ggml_backend_t,const char*,size_t budget=2u<<20);
void strata_vk_native_ple_conv_destroy(StrataVkNativePleConv*) noexcept;
size_t strata_vk_native_ple_conv_bytes(const StrataVkNativePleConv*);
void strata_vk_native_ple_conv_run(StrataVkNativePleConv*,ggml_tensor* const[10]);
}
