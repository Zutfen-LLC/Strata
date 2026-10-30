#pragma once
#include "ggml-backend.h"
#include <cstdint>
namespace strata::core::portable_detail {
// Standalone observation only. No FFN/model option. Never qualified by host math.
struct StrataVkNativeRouter;
StrataVkNativeRouter* strata_vk_native_router_create(ggml_backend_t,const char* effective_spv_path);
void strata_vk_native_router_destroy(StrataVkNativeRouter*) noexcept;
// Contiguous F32 logits [T,512], I32 IDs [T,10], F32 weights [T,10], T1..8.
// Same backend/device, 4-byte aligned, all spans disjoint. Nonfinite logits refuse.
void strata_vk_native_router_run(StrataVkNativeRouter*,const ggml_tensor*,ggml_tensor*,ggml_tensor*,uint32_t);
}
