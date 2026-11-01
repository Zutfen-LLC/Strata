#pragma once
#include "ggml-backend.h"
#include <cstddef>
namespace strata::core::portable_detail {
// Experimental fixed S128/HK16/HV48/T1, non-QFUSE self-commit arithmetic.
// Immutable input state; candidate is committed by the existing layer transaction only.
struct StrataVkNativeGdn;
StrataVkNativeGdn* strata_vk_native_gdn_create(ggml_backend_t,const char*,size_t budget=65536);
void strata_vk_native_gdn_destroy(StrataVkNativeGdn*) noexcept;
size_t strata_vk_native_gdn_bytes(const StrataVkNativeGdn*);
// 0..5 state,h(q/k/v),gate,beta,z,gamma. 6..12 candidate,y,exp(g),oc,rs,exp(-z),sigmoid.
// Auxiliary owns disjoint views 8/10/11/12, within one descriptor (GGML limit12).
// Fixed F32 contiguous disjoint Vulkan spans, strict FP controls; no live state mutation.
void strata_vk_native_gdn_run(StrataVkNativeGdn*,ggml_tensor* const tensors[13],ggml_tensor* auxiliary);
}
