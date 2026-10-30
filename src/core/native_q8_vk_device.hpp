#pragma once
#include "native_q8_integer_dot.hpp"
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
#include "native_q8_strict_fma.hpp"
#endif
// Included only before ggml_vk_get_device in the exact pinned build-tree copy.
// The base ABI stays unchanged for other backend TUs. shared_ptr retains this
// derived deleter; GPU resources and backend ownership remain the base's.
struct StrataVkQ8Device final : vk_device_struct {
    strata::core::portable_detail::NativeQ8IntegerDotState strata_q8_dot;
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
    strata::core::portable_detail::NativeQ8StrictFmaState strata_q8_fma;
#endif
};
