#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeGdnLog1p;
StrataVkNativeGdnLog1p* strata_vk_native_gdn_log1p_create(ggml_backend_t,const char*,size_t budget=65536);
void strata_vk_native_gdn_log1p_destroy(StrataVkNativeGdnLog1p*) noexcept;
// Fixed disjoint F32[256]: x, result, rounded u, log(u), numerator, denominator.
// No state commit. Returns numerical status after actual outputs are available.
uint32_t strata_vk_native_gdn_log1p_run(StrataVkNativeGdnLog1p*,ggml_tensor*const[6]);
}
