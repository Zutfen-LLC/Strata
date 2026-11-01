#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeLlvmLog1p;
StrataVkNativeLlvmLog1p* strata_vk_native_llvm_log1p_create(ggml_backend_t,const char*,size_t budget=65536);
void strata_vk_native_llvm_log1p_destroy(StrataVkNativeLlvmLog1p*) noexcept;
// Fixed disjoint x/result F32[256], wide I32[512], path I32[256].
// No state commit. FP64 accurate/non-FMA LLVM primitive only.
void strata_vk_native_llvm_log1p_run(StrataVkNativeLlvmLog1p*,ggml_tensor*const[4]);
}
