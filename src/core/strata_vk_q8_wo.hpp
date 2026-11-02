#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
// Standalone native T1 Q8_0 candidate. All tensors/backend are borrowed.
struct StrataVkQ8Wo;
StrataVkQ8Wo* strata_vk_q8_wo_create(ggml_backend_t,const char* effective_spv_path,size_t budget);
// Same native ordering for the distinct 2,560-input Q/K/V projection geometry.
StrataVkQ8Wo* strata_vk_qsa_qkv_create(ggml_backend_t,const char* effective_spv_path,size_t budget);
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*);
uint64_t strata_vk_q8_wo_submitted(const StrataVkQ8Wo*) noexcept;
uint64_t strata_vk_q8_wo_completed(const StrataVkQ8Wo*) noexcept;
uint64_t strata_vk_q8_wo_submission_uncertain(const StrataVkQ8Wo*) noexcept;
void strata_vk_q8_wo_destroy(StrataVkQ8Wo*) noexcept;
void strata_vk_q8_wo_run(StrataVkQ8Wo*,const ggml_tensor* weights,const ggml_tensor* packed,
                         ggml_tensor* output,size_t rows);
}
