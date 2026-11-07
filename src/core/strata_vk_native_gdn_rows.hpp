#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeGdnRows;
StrataVkNativeGdnRows* strata_vk_native_gdn_rows_create(ggml_backend_t,const char*,size_t);
void strata_vk_native_gdn_rows_destroy(StrataVkNativeGdnRows*) noexcept;
size_t strata_vk_native_gdn_rows_bytes(const StrataVkNativeGdnRows*);
// T1 recurrence inputs/outputs in native order; t[0] and t[6] are distinct
// state banks. Per-token fields t[1..4], t[7..12] are row-major. t[8],
// t[10..12] are disjoint views of auxiliary. One workgroup owns each head.
void strata_vk_native_gdn_rows_run(StrataVkNativeGdnRows*,ggml_tensor*const[13],ggml_tensor* auxiliary,uint32_t rows);
}
