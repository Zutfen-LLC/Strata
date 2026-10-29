#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
// Standalone diagnostic component. Borrowed backend and tensors must outlive it.
// Not connected to model execution. No CPU fallback.
struct StrataVkNativeStages;
StrataVkNativeStages* strata_vk_native_stages_create(ggml_backend_t);
void strata_vk_native_stages_destroy(StrataVkNativeStages*) noexcept;
// mode 0: quantize all input; 1: separate SwiGLU all capacity; 2: fused
// SwiGLU + quantize [starts[0],starts[count]). Shape is bounded, rows % 32 == 0.
// starts/count are host-validated and uploaded by this component, never pointers.
void strata_vk_native_stages_run(StrataVkNativeStages*, uint32_t mode,
    const ggml_tensor* gate, const ggml_tensor* up, ggml_tensor* hidden, ggml_tensor* packed,
    uint32_t n_ff, const int32_t* starts, size_t n_starts);
}
