#pragma once
#include "strata_vk_native_router.hpp"
namespace strata::core::portable_detail {
// Available only with STRATA_ROUTER_DIAGNOSTIC. Separate pipeline; original
// run/destroy ownership and dispatch geometry are reused, never selected by default.
// Inputs must be the exact synthetic T8 fixture; the standalone runner verifies it.
StrataVkNativeRouter* strata_vk_native_router_trace_create(ggml_backend_t,const char*);
void strata_vk_native_router_trace_run(StrataVkNativeRouter*,const ggml_tensor*,ggml_tensor*,ggml_tensor*,uint32_t* trace26);
}
