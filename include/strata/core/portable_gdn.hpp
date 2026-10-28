#pragma once
#include "strata/core/portable_ple.hpp"

namespace strata::core {
struct PortableGdnOptions {
    PortableFfnOptions execution;
    size_t state_budget = 16ull << 20;           // host committed/candidate state and residuals
    std::shared_ptr<PortablePleTable> ple_table; // required for create(layer=1); host only
};
struct PortableGdnTrace {
    std::vector<float> qkv, convolution, query, key, value, decay, beta, attention;
    std::array<uint32_t, 16> ple_rows{};
    std::vector<float> ple_embedding, ple_key, ple_value, ple_gate, ple_normalized, ple_convolution, ple_residual;
};

// Complete qwen4exp layers 0/1, T=1: optional configured PLE, HC/GDN,
// PortableFfn. Exact fleet geometry only. QSA and other layers are rejected.
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
    // Token-aware entry for layer 1. Supply the configured image token ID for
    // image positions. EOS cuts hash predecessors on subsequent tokens; it
    // does not clear recurrent/convolution state. reset() starts a new sequence.
    bool step(int32_t token, const std::vector<float> &residual, const PortableExpertRows &dispatch,
              PortableRoute &route, std::string &error, PortableGdnTrace *trace = nullptr);
    const std::vector<float> &ple_convolution_history() const;
    const std::array<int32_t, 2> &token_history() const;
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
