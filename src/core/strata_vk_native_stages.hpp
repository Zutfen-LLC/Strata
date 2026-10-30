#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
// Standalone diagnostic component. Borrowed backend and tensors must outlive it.
// Explicit unqualified provider observation can also use this owner. No CPU fallback.
struct StrataVkNativeStages;
StrataVkNativeStages* strata_vk_native_stages_create(ggml_backend_t);
// Explicit observation only; never called as a fallback. Writes final module receipt.
StrataVkNativeStages* strata_vk_native_stages_create_observation(ggml_backend_t,const char* effective_spv_path);
// Separate observation probe: first 16 words of hidden hold amax/d/ratio/rounded
// for four fixed input elements. Run only mode0 on the complete H2560 fixture.
StrataVkNativeStages* strata_vk_native_stages_create_quant_trace_observation(ggml_backend_t,const char* effective_spv_path);
// Explicit corrected quantizer diagnostic: only amax/127 and xi/d use integer
// IEEE binary32 nearest-even division; activation arithmetic stays unchanged.
StrataVkNativeStages* strata_vk_native_stages_create_quant_rn_observation(ggml_backend_t,const char* effective_spv_path);
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*);
void strata_vk_native_stages_destroy(StrataVkNativeStages*) noexcept;
// mode 0: quantize all input; 1: separate SwiGLU all capacity; 2: fused
// SwiGLU + quantize [starts[0],starts[count]). Shape is bounded, rows % 32 == 0.
// starts/count are host-validated and uploaded by this component, never pointers.
void strata_vk_native_stages_run(StrataVkNativeStages*, uint32_t mode,
    const ggml_tensor* gate, const ggml_tensor* up, ggml_tensor* hidden, ggml_tensor* packed,
    uint32_t n_ff, const int32_t* starts, size_t n_starts);
}
