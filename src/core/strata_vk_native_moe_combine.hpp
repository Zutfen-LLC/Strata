#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
// Standalone observation component; available only in the explicit strict-FMA
// native-stage build. No PortableFfn integration or default selection yet.
struct StrataVkNativeMoeCombine;
StrataVkNativeMoeCombine* strata_vk_native_moe_combine_create(ggml_backend_t,const char* effective_spv_path);
size_t strata_vk_native_moe_combine_bytes(const StrataVkNativeMoeCombine*);
void strata_vk_native_moe_combine_destroy(StrataVkNativeMoeCombine*) noexcept;
// Contiguous device F32: parts [T,10,2560], weights [T,10], shared/output [T,2560].
// Shared is already gated; output must be disjoint from every input.
void strata_vk_native_moe_combine_run(StrataVkNativeMoeCombine*,const ggml_tensor* parts,
    const ggml_tensor* weights,const ggml_tensor* shared,ggml_tensor* output,uint32_t tokens);
}
