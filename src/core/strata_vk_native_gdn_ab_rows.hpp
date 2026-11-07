#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeGdnAbRows;
StrataVkNativeGdnAbRows* strata_vk_native_gdn_ab_rows_create(ggml_backend_t,const char*,size_t);
void strata_vk_native_gdn_ab_rows_destroy(StrataVkNativeGdnAbRows*) noexcept;
size_t strata_vk_native_gdn_ab_rows_bytes(const StrataVkNativeGdnAbRows*);
// mixed[T*2560], BF16 alpha/beta weights[48*2560], dt/A[48],
// gate/beta[T*48], diagnostics[T*336]. Returns shader domain status.
uint32_t strata_vk_native_gdn_ab_rows_run(StrataVkNativeGdnAbRows*,ggml_tensor*const[8],uint32_t);
}
