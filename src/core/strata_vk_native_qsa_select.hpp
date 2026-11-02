#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeQsaSelect;
StrataVkNativeQsaSelect* strata_vk_native_qsa_select_create(ggml_backend_t,const char*score_path,const char*select_path,size_t budget=131072);
void strata_vk_native_qsa_select_destroy(StrataVkNativeQsaSelect*)noexcept;
size_t strata_vk_native_qsa_select_bytes(const StrataVkNativeQsaSelect*);
uint64_t strata_vk_native_qsa_score_calls(const StrataVkNativeQsaSelect*)noexcept;
uint64_t strata_vk_native_qsa_select_calls(const StrataVkNativeQsaSelect*)noexcept;
// pooled F32[blocks*128],dead F32[128],query F32[512],step I32[4],score F32[blocks].
uint32_t strata_vk_native_qsa_scores_run(StrataVkNativeQsaSelect*,ggml_tensor*const[5],uint32_t cells);
// score F32[blocks],step I32[4],candidate IDs I32[2051]. Leaves inactive IDs intact.
// Native order_key handles NaN/Inf/signed zero; computed score arithmetic has a
// separate finite/normal contract. Neither operation changes model state.
uint32_t strata_vk_native_qsa_select_run(StrataVkNativeQsaSelect*,ggml_tensor*const[3],uint32_t cells);
}
