#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include "native_q8_down_layout.hpp"
namespace strata::core::portable_detail {
struct StrataVkQ8Down;
StrataVkQ8Down* strata_vk_q8_down_create(ggml_backend_t,const char* effective_spv_path);
size_t strata_vk_q8_down_bytes(const StrataVkQ8Down*);
const ggml_tensor* strata_vk_q8_down_metadata_tensor(const StrataVkQ8Down*);
void strata_vk_q8_down_destroy(StrataVkQ8Down*) noexcept;
void strata_vk_q8_down_run(StrataVkQ8Down*,const ggml_tensor* weights,const ggml_tensor* hq,
    ggml_tensor* output,const Q8DownLayout&);
// One checked owner per group, in original group order. No bank regrouping.
void strata_vk_q8_down_run_blocks(StrataVkQ8Down*,const std::vector<const ggml_tensor*>& weights,const ggml_tensor* packed,
    ggml_tensor* output,const Q8DownLayout&);
}
