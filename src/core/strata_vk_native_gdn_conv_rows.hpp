#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
// Borrowed backend/tensors. Separate immutable history and candidate history.
struct StrataVkNativeGdnConvRows;
StrataVkNativeGdnConvRows* strata_vk_native_gdn_conv_rows_create(ggml_backend_t,const char*,size_t budget);
void strata_vk_native_gdn_conv_rows_destroy(StrataVkNativeGdnConvRows*) noexcept;
size_t strata_vk_native_gdn_conv_rows_bytes(const StrataVkNativeGdnConvRows*);
// T=1 or 2: history[30720], qkv[T*10240], weights[40960],
// next-history[30720], h/pre/exp/silu[T*10240], rs/arg[T*32].
void strata_vk_native_gdn_conv_rows_run(StrataVkNativeGdnConvRows*,ggml_tensor* const[10],uint32_t rows);
}
