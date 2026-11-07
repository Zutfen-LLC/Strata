#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeShared;
// Standalone opt-in observation. Owner/weights/scratch must survive synchronous
// calls; caller serializes this component and its backend. Model use is opt-in.
StrataVkNativeShared* strata_vk_native_shared_create(ggml_backend_t,const char* effective_spv_path,size_t metadata_budget=65536);
// Separate matrix variant: exactly one 2560-wide input and 128 or 512 BF16 rows.
StrataVkNativeShared* strata_vk_qsa_indexer_bf16_create(ggml_backend_t,const char* effective_spv_path,size_t budget);
StrataVkNativeShared* strata_vk_qsa_indexer_bf16_rows_create(ggml_backend_t,const char*,size_t);
// Separate PLE value geometry; same existing matrix shader.
StrataVkNativeShared* strata_vk_ple_value_bf16_create(ggml_backend_t,const char* effective_spv_path,size_t budget);
uint64_t strata_vk_native_shared_submitted(const StrataVkNativeShared*) noexcept;
uint64_t strata_vk_native_shared_completed(const StrataVkNativeShared*) noexcept;
uint64_t strata_vk_native_shared_uncertain(const StrataVkNativeShared*) noexcept;
size_t strata_vk_native_shared_bytes(const StrataVkNativeShared*);
void strata_vk_native_shared_destroy(StrataVkNativeShared*) noexcept;
// mode0: dense Q8_0/I32 weights, Q8_1/I32 input -> F32 output; aux F32[1].
// mode1: raw BF16 packed in I32 weights, original F32 input -> raw scalar F32[T]; aux F32[1].
// mode2: raw scalar F32[T] weights, down F32[T,ni] -> scaled F32[T,ni], sigmoid aux F32[T].
// All four spans must be disjoint, contiguous and on this Vulkan device.
void strata_vk_native_shared_run(StrataVkNativeShared*,uint32_t mode,const ggml_tensor* weights,
    const ggml_tensor* input,ggml_tensor* output,ggml_tensor* aux,uint32_t ni,uint32_t no,uint32_t tokens);
}
