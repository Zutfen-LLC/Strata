#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include "native_q8_gu_layout.hpp"
namespace strata::core::portable_detail {
size_t strata_vk_q8_resident_block_limit(ggml_backend_t);
// native_gu_kernel<8> port. Borrowed backend/tensors outlive this serial owner.
struct StrataVkQ8Gu;
StrataVkQ8Gu* strata_vk_q8_gu_create(ggml_backend_t,const char* effective_spv_path);
size_t strata_vk_q8_gu_bytes(const StrataVkQ8Gu*);
const ggml_tensor* strata_vk_q8_gu_metadata_tensor(const StrataVkQ8Gu*);
void strata_vk_q8_gu_destroy(StrataVkQ8Gu*) noexcept;
void strata_vk_q8_gu_run(StrataVkQ8Gu*,const ggml_tensor* weights,const ggml_tensor* packed,
    ggml_tensor* gate,ggml_tensor* up,const Q8GuLayout&);
// One checked owner per group, in original group order. No bank regrouping.
void strata_vk_q8_gu_run_blocks(StrataVkQ8Gu*,const std::vector<const ggml_tensor*>& weights,const ggml_tensor* packed,
    ggml_tensor* gate,ggml_tensor* up,const Q8GuLayout&);
}
