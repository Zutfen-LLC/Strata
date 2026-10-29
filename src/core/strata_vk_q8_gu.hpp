#pragma once
#include "ggml-backend.h"
#include "native_q8_gu_layout.hpp"
namespace strata::core::portable_detail {
// Standalone native_gu_kernel<8> port. No quantization, activation, down or
// product integration. Borrowed backend/tensors outlive this serial owner.
struct StrataVkQ8Gu;
StrataVkQ8Gu* strata_vk_q8_gu_create(ggml_backend_t,const char* effective_spv_path);
void strata_vk_q8_gu_destroy(StrataVkQ8Gu*) noexcept;
void strata_vk_q8_gu_run(StrataVkQ8Gu*,const ggml_tensor* weights,const ggml_tensor* packed,
    ggml_tensor* gate,ggml_tensor* up,const Q8GuLayout&);
}
