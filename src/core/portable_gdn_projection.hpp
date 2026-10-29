#pragma once
#include "portable_graph.hpp"
namespace strata::core::portable_detail {
// The pinned Vulkan vector selector may quantize contiguous multi-column F32
// inputs to Q8_1 even when F32 source/accumulator precision is requested. Keep
// the existing F32 values in a strided view; its supported packing path copies
// them as F32 before the F32-activation kernel. Only batched GDN uses this seam.
inline ggml_tensor* gdn_f32_projection_columns(ggml_context* c,ggml_tensor* x) {
    require(x && x->type==GGML_TYPE_F32 && x->ne[1]>=2 && x->ne[1]<=8 && x->ne[2]==1 && x->ne[3]==1,
            "portable GDN: invalid F32 projection column shape");
    const int64_t K=x->ne[0];
    auto* padded=ggml_pad(c,x,1,0,0,0);
    ggml_set_name(padded,"gdn-f32-activation-storage");
    auto* columns=ggml_view_2d(c,padded,K,x->ne[1],size_t(K+1)*sizeof(float),0);
    ggml_set_name(columns,"gdn-f32-activation-columns");
    return columns;
}
} // namespace strata::core::portable_detail
