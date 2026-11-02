#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeQsaAttention;
StrataVkNativeQsaAttention* strata_vk_native_qsa_attention_create(ggml_backend_t,const char*chunk,const char*merge,const char*gate,size_t budget=196608);
void strata_vk_native_qsa_attention_destroy(StrataVkNativeQsaAttention*)noexcept;
size_t strata_vk_native_qsa_attention_bytes(const StrataVkNativeQsaAttention*);
uint64_t strata_vk_native_qsa_attention_calls(const StrataVkNativeQsaAttention*,unsigned stage)noexcept;
// Immutable inputs K,V F16[ceil(cells/4)*2*4*256], Q F32[6144], pages I32[ceil(cells/4)],
// IDs I32[2051], step I32[4], Q-full F32[12288]. Disjoint candidates:
// scratch F32[204400], chunk diagnostics F32[101376], merge diagnostics F32[6984],
// attention F32[6144], gate diagnostics F32[12288], gated output F32[6144].
// No persistent state/commit. Returns nonzero stage status without dispatching
// later stages; completed stages are paid even if a later stage refuses.
uint32_t strata_vk_native_qsa_attention_run(StrataVkNativeQsaAttention*,ggml_tensor*const[13],uint32_t cells);
}
