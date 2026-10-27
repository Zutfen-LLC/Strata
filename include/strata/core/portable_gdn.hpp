#pragma once
#include "strata/core/portable_ffn.hpp"

namespace strata::core {
struct PortableGdnOptions {
    PortableFfnOptions execution;
    size_t state_budget = 16ull << 20; // host committed/candidate state and residuals
};
struct PortableGdnTrace {
    std::vector<float> qkv, convolution, query, key, value, decay, beta, attention;
};

// Complete qwen4exp layer 0, T=1: HC/GDN attention followed by PortableFfn.
// Exact fleet geometry only. Other layers (including PLE/QSA) are rejected.
class PortableGdnLayer {
  public:
    static std::unique_ptr<PortableGdnLayer> create(int64_t layer, const std::vector<PortableTensorView> &tensors,
                                                    PortableGdnOptions options, std::string &error);
    static std::unique_ptr<PortableGdnLayer> from_gguf(const GgufModel &model, int64_t layer,
                                                       PortableGdnOptions options, std::string &error);
    ~PortableGdnLayer();
    PortableGdnLayer(const PortableGdnLayer &) = delete;
    PortableGdnLayer &operator=(const PortableGdnLayer &) = delete;
    bool reset(std::string &error);
    // A fresh token residual (four streams) is supplied on every call. Public
    // persistent state and last_residual commit together only after FFN success.
    bool step(const std::vector<float> &residual, const PortableExpertRows &dispatch, PortableRoute &route,
              std::string &error, PortableGdnTrace *trace = nullptr);
    const std::vector<float> &recurrent_state() const;
    const std::vector<float> &convolution_history() const;
    const std::vector<float> &last_residual() const;
    uint64_t tokens() const;
    const std::vector<PortablePlacement> &placements() const;
    size_t weight_bytes() const;
    size_t graph_bytes() const;
    size_t state_bytes() const;

  private:
    struct Impl;
    explicit PortableGdnLayer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
