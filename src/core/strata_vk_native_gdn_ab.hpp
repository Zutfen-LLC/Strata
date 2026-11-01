#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeGdnAb;
StrataVkNativeGdnAb* strata_vk_native_gdn_ab_create(ggml_backend_t,const char*,size_t budget=65536);
void strata_vk_native_gdn_ab_destroy(StrataVkNativeGdnAb*) noexcept;
size_t strata_vk_native_gdn_ab_bytes(const StrataVkNativeGdnAb*);
uint64_t strata_vk_native_gdn_ab_calls(const StrataVkNativeGdnAb*) noexcept;
// Fixed T1: x, packed BF16 wa/wb, dt, ssm_a, gate, beta, packed diagnostics.
// No state commit. Nonzero status refuses actual unsupported arithmetic classes.
uint32_t strata_vk_native_gdn_ab_run(StrataVkNativeGdnAb*,ggml_tensor*const[8]);
}
