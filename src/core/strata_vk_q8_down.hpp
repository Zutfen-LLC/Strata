#pragma once
#include "ggml-backend.h"
#include "native_q8_down_layout.hpp"
namespace strata::core::portable_detail {
struct StrataVkQ8Down;
StrataVkQ8Down* strata_vk_q8_down_create(ggml_backend_t,const char* effective_spv_path);
void strata_vk_q8_down_destroy(StrataVkQ8Down*) noexcept;
void strata_vk_q8_down_run(StrataVkQ8Down*,const ggml_tensor* weights,const ggml_tensor* hq,
    ggml_tensor* output,const Q8DownLayout&);
}
