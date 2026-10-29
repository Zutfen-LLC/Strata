#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend;
namespace strata { class GgufModel; }

namespace strata::core {

// Opt-in T=1 qwen4exp FFN coordinator. This is the after-attention sublayer,
// not a complete layer/model/session. No CUDA runtime or graph capture is used.
struct PortableFfnGeometry {
    int64_t hidden = 0, streams = 4, rank = 0, shared_ff = 0, experts = 0, selected = 0;
    float epsilon = 1e-6f;
};

// GGUF layout: dimension zero varies fastest. FFN accepts F32 (0)/Q8_0 (8).
// QSA additionally accepts BF16 (30) only for indexer Q/K projections and
// widens them exactly to F32. Payloads are copied at construction.
struct PortableTensorView {
    std::string name;
    std::vector<int64_t> shape;
    uint32_t type = 0;
    const void* data = nullptr;
    size_t bytes = 0;
};

struct PortableFfnOptions {
    int threads = 1;
    bool overlap_shared_cpu = false; // submit shared graph before synchronous expert dispatch; default off
    // Borrowed backend, e.g. ggml_backend_vk_init(die). Caller retains lifetime.
    ggml_backend* accelerator = nullptr;
    bool request_accelerator = false;
    bool allow_cpu_fallback = false;
    size_t weight_budget = 256ull << 20;
    size_t graph_budget = 256ull << 20; // aggregate scheduler buffers, both phases
    bool resident_dense_weights = false; // model-lifetime weights on requested backend; default off
};

struct PortablePlacement {
    std::string phase, operation, tensor, type, backend, fallback_reason;
    std::vector<int64_t> shape;
    bool metadata_only = false; // reshape/view/permute/transpose alias storage
    int64_t layer = -1; // model report: 0..47, 48 for global head
};

struct PortableMemoryUsage {
    size_t host_weights = 0, host_graph = 0, device_graph = 0;
    // Conservative admission for backend workspaces, metadata and host state.
    size_t host_extra_bound = 0, device_extra_bound = 0;
    // One token-local scheduler, destroyed on layer return. It may overlap
    // expert work inside that layer, but never another layer's temporary.
    // Backend-private allocations are excluded: they remain device_extra_bound.
    size_t device_token_temporary_bound = 0;
    // Only the verified Vulkan backend-context X conversion buffer. Other
    // backend/private buffers remain additive in device_extra_bound.
    size_t device_shared_conversion_bound = 0;
    size_t device_state = 0; // actual aligned persistent session state
    size_t device_weights = 0; // actual aligned persistent weights, separate from scheduler storage
};

struct PortableRoute {
    std::vector<float> activation;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

// One unweighted hidden-width row per selected expert, in route.ids order.
// Compatible with the existing indexed-row worker boundary. No ownership or
// transport policy is selected here; the product supplies its existing dispatch.
using PortableExpertRows = std::function<bool(int64_t layer, const PortableRoute& route,
                                             std::vector<float>& rows, std::string& error)>;

class PortableFfn {
public:
    static std::unique_ptr<PortableFfn> create(PortableFfnGeometry geometry, int64_t layer,
        const std::vector<PortableTensorView>& weights, PortableFfnOptions options, std::string& error);
    // Maps existing split GGUFs and copies only this sublayer's nine tensors.
    // Does not touch routed expert payloads, the PLE table, or attention weights.
    static std::unique_ptr<PortableFfn> from_gguf(const GgufModel& model, int64_t layer,
        PortableFfnOptions options, std::string& error);
    ~PortableFfn();
    PortableFfn(const PortableFfn&) = delete;
    PortableFfn& operator=(const PortableFfn&) = delete;

    bool reset(const std::vector<float>& residual, std::string& error);
    // Transactional residual: dispatch/validation failure leaves it unchanged.
    bool step(const PortableExpertRows& dispatch, PortableRoute& route, std::string& error);
    const std::vector<float>& residual() const;
    const std::vector<PortablePlacement>& placements() const;
    size_t weight_bytes() const;
    size_t graph_bytes() const;
    PortableMemoryUsage memory_usage() const;

private:
    struct Impl;
    explicit PortableFfn(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core
