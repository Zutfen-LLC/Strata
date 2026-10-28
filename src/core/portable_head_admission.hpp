#pragma once
#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core::portable_detail {
struct PhysicalStorageLimit {
    std::string pci;
    uint64_t max_storage_buffer_range = 0;
};
// Ambiguous/missing properties never earn admission credit. This seam also lets
// host tests exercise physical matching without a Vulkan instance or device.
inline PhysicalStorageLimit match_storage_limit(const std::string& pci,
                                                const std::vector<PhysicalStorageLimit>& observations) {
    PhysicalStorageLimit result{pci, 0};
    size_t matches = 0;
    for (const auto& item : observations) {
        if (!pci.empty() && item.pci == pci) {
            ++matches;
            result.max_storage_buffer_range = item.max_storage_buffer_range;
        }
    }
    if (matches != 1) result.max_storage_buffer_range = 0;
    return result;
}
// Read-only physical properties via the linked Vulkan loader, matched to the
// actual GGML backend PCI identity. Non-Vulkan/failed queries return no proof.
PhysicalStorageLimit head_storage_limit(ggml_backend_t backend);

inline bool direct_head_q8_matvec(const ggml_tensor* node, const PhysicalStorageLimit& limit) {
    if (!node || node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1] || limit.pci.empty())
        return false;
    const auto* x = node->src[0];
    const auto* y = node->src[1];
    int32_t parameters[2]{};
    std::memcpy(parameters, node->op_params, sizeof(parameters));
    // Deliberately only the verified qwen4exp output head at T=1. Other graphs
    // keep the old bound even if a broader fast-path argument might apply.
    return x->type == GGML_TYPE_Q8_0 && y->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
           x->ne[0] == 2560 && x->ne[1] == 248320 && x->ne[2] == 1 && x->ne[3] == 1 &&
           y->ne[0] == 2560 && y->ne[1] == 1 && y->ne[2] == 1 && y->ne[3] == 1 &&
           node->ne[0] == 248320 && node->ne[1] == 1 && node->ne[2] == 1 && node->ne[3] == 1 &&
           ggml_is_contiguous(x) && ggml_is_contiguous(y) && ggml_is_contiguous(node) &&
           !x->view_src && parameters[0] == 0 && parameters[1] == 0 &&
           ggml_nbytes(x) == 675430400 && limit.max_storage_buffer_range >= ggml_nbytes(x);
}
inline size_t head_conversion_bound(size_t output_f32, size_t hc_f32, bool direct) {
    return std::max(direct ? size_t(0) : output_f32, hc_f32);
}
} // namespace strata::core::portable_detail
