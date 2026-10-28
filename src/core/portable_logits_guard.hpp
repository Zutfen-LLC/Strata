#pragma once
#include "ggml.h"

namespace strata::core::portable_detail {
// T=1 IEEE F32 status reduction for the pinned CPU/Vulkan graph backend.
// Each finite x gives x-x == 0 (including FLT_MAX); Inf/NaN gives NaN.
// Separate SUB and SUM kernels must preserve nonfinite values: no fast-math
// rewrite to zero is allowed. Qualify this graph on each backend/driver before
// admitting production. Unlike native's exponent-bit kernel, this uses floating
// arithmetic; it never feeds back into or changes the logits tensor.
inline ggml_tensor *logits_status(ggml_context *ctx, ggml_tensor *logits) {
    auto *status = ggml_sum(ctx, ggml_sub(ctx, logits, logits));
    ggml_set_name(status, "portable_logits_status");
    ggml_set_output(status);
    return status;
}
} // namespace strata::core::portable_detail
