#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace strata::core::portable_detail {
struct StrataVkNativeHc;
// Standalone component and experimental serial bridge. Numerical contract v1;
// no universal rsqrt bit identity or full-HC qualification.
// Normal-or-zero arithmetic domain only. No Q8/REQ8/V3/pending-write fallback.
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* effective_spv,size_t metadata_budget=65536);
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*);
void strata_vk_native_hc_destroy(StrataVkNativeHc*) noexcept;
// Raw BF16 matrix packed in I32; all other tensors F32. Fixed H2560/HC4/T1..8.
// Synchronous call; host domain preflight is diagnostic-only and bounded (<1 MiB).
// Tensors must be disjoint contiguous spans on the same Vulkan device.
void strata_vk_native_hc_run(StrataVkNativeHc*,const ggml_tensor* residual,const ggml_tensor* norm_weight,
 const ggml_tensor* bf16_weight,ggml_tensor* normalized,ggml_tensor* rs,ggml_tensor* inject,uint32_t tokens);
// T1 materialized-head normalization only. The injection matrix and projection
// are absent; normalized and rs remain owned by the caller.
void strata_vk_native_hc_norm_only(StrataVkNativeHc*,const ggml_tensor* residual,const ggml_tensor* norm_weight,
 ggml_tensor* normalized,ggml_tensor* rs);
}
namespace strata::core::portable_detail {
struct StrataVkNativeHcMix;
// Thrown only after all three dispatches completed and current outputs were
// synchronized. Other failures must never expose possibly stale scratch.
struct NativeHcMixDomainRefusal : std::invalid_argument {
    uint32_t status;
    explicit NativeHcMixDomainRefusal(uint32_t value):std::invalid_argument(
        "native HC mix: domain refusal status="+std::to_string(value)+"; no result accepted"),status(value) {}
};
// Standalone T=1 or T=2, inferred from normalized span; no model bridge.
// Row-major normalized[T,10240], shared BF16 down/up [320,10240]/[10240,320].
// Outputs pre/low/exp[T,320], up pre/gate/exp[T,10240], mixed[T,2560].
// All spans disjoint; zero/normal domain; derived exp/div guards in native_hc_mix_limits.h.
StrataVkNativeHcMix* strata_vk_native_hc_mix_create(ggml_backend_t,const char* effective_spv,size_t budget=65536);
// Separate head-only candidate; existing mixer users retain the original shader.
StrataVkNativeHcMix* strata_vk_native_head_hc_mix_rn_create(ggml_backend_t,const char* effective_spv,size_t budget=65536);
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*);
void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix*) noexcept;
// Test-only identical preactivation probe; same production shader activation.
void strata_vk_native_hc_sigmoid_probe(StrataVkNativeHcMix*,const ggml_tensor* pre,ggml_tensor* exponential,ggml_tensor* gate);
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor* normalized,
 const ggml_tensor* down,const ggml_tensor* up,ggml_tensor* low_pre,ggml_tensor* low,ggml_tensor* low_exp,
 ggml_tensor* up_pre,ggml_tensor* gates,ggml_tensor* up_exp,ggml_tensor* mixed);
}

namespace strata::core::portable_detail {
// Standalone contiguous T=1 or T=2 candidate. Row count is inferred from the
// residual span; all other spans must match. Original R/bo/inject stay immutable until the
// caller explicitly commits after all consumers succeed; no product integration.
struct StrataVkNativeHcPending;
StrataVkNativeHcPending* strata_vk_native_hc_pending_create(ggml_backend_t,const char* effective_spv,size_t budget=65536);
// Explicit standalone final-write mode; same disjoint candidate/run/destroy contract.
// Preserves native_gr_post scale-zero-bias operations. No model enablement.
StrataVkNativeHcPending* strata_vk_native_hc_write_create(ggml_backend_t,const char* effective_spv,size_t budget=65536);
size_t strata_vk_native_hc_pending_bytes(const StrataVkNativeHcPending*);
void strata_vk_native_hc_pending_destroy(StrataVkNativeHcPending*) noexcept;
void strata_vk_native_hc_pending_run(StrataVkNativeHcPending*,const ggml_tensor* residual,const ggml_tensor* block_out,
 const ggml_tensor* injection,ggml_tensor* candidate,ggml_tensor* gate,ggml_tensor* exponential);
}
