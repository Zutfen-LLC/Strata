#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeQsaNormRope;
StrataVkNativeQsaNormRope* strata_vk_native_qsa_norm_rope_create(ggml_backend_t,const char*,size_t budget=65536);
void strata_vk_native_qsa_norm_rope_destroy(StrataVkNativeQsaNormRope*)noexcept;
size_t strata_vk_native_qsa_norm_rope_bytes(const StrataVkNativeQsaNormRope*);
uint64_t strata_vk_native_qsa_norm_rope_calls(const StrataVkNativeQsaNormRope*)noexcept;
// x[rows*stride],gamma[cols],I32 positions[rows],normalized/rotated[rows*cols],
// diagnostics[rows*130]. T1 text/None scaling, n_rot64, cols128/256,
// rows1..24, stride=cols or2cols, positions0..255, epsilon1e-6.
// Input immutable; disjoint outputs are candidates, never a state commit.
uint32_t strata_vk_native_qsa_norm_rope_run(StrataVkNativeQsaNormRope*,ggml_tensor*const[6],uint32_t cols,uint32_t rows,uint32_t stride,uint32_t theta_scale);
}
