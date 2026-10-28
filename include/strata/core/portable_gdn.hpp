#pragma once
#include "strata/core/portable_ple.hpp"

namespace strata::core {
struct PortableGdnOptions {
    PortableFfnOptions execution;
    size_t state_budget = 16ull << 20;           // host committed/candidate state and residuals
    std::shared_ptr<PortablePleTable> ple_table; // enables configured layer-1 PLE; host only
    bool resident_state = false; // session-lifetime GDN/PLE state; host mirrors stay validated
    size_t resident_state_budget = 4ull << 20; // aligned persistent backend inputs, separate from graphs
    bool resident_attention_weights = false; // opt-in; CPU execution retains CPU allocation
};
// Cumulative host-call wall scopes. Compute is nested in attention/exchange;
// full FFN includes the provider. Reset does not clear caller-owned totals.
struct PortableGdnWork {
    double attention_exchange_ms = 0, attention_compute_ms = 0, ffn_ms = 0;
    // Candidate-only reads (state, convolution history, residual, optional PLE
    // history). Readback and finite scopes are disjoint within exchange; trace
    // reads are excluded. Commit encloses existing device copies and wait, after FFN.
    double candidate_readback_ms = 0, candidate_finite_ms = 0, state_commit_ms = 0;
    uint64_t candidate_readback_calls = 0, candidate_finite_calls = 0, state_commit_calls = 0;
    uint64_t attention_exchange_calls = 0, attention_compute_calls = 0, ffn_calls = 0;
};
struct PortableGdnTrace {
    std::vector<float> qkv, convolution, query, key, value, decay, beta, attention;
    std::array<uint32_t, 16> ple_rows{};
    std::vector<float> ple_embedding, ple_key, ple_value, ple_gate, ple_normalized, ple_convolution, ple_residual;
};

// Complete qwen4exp non-QSA layers 0..47, T=1: configured PLE, HC/GDN,
// PortableFfn. Exact fleet geometry only. QSA layers are rejected.
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
              std::string &error, PortableGdnTrace *trace = nullptr, PortableGdnWork *work = nullptr);
    // Token-aware entry for configured PLE. Supply the configured image token ID for
    // image positions. EOS cuts hash predecessors on subsequent tokens; it
    // does not clear recurrent/convolution state. reset() starts a new sequence.
    bool step(int32_t token, const std::vector<float> &residual, const PortableExpertRows &dispatch,
              PortableRoute &route, std::string &error, PortableGdnTrace *trace = nullptr, PortableGdnWork *work = nullptr);
    const std::vector<float> &ple_convolution_history() const;
    const std::array<int32_t, 2> &token_history() const;
    const std::vector<float> &recurrent_state() const;
    const std::vector<float> &convolution_history() const;
    const std::vector<float> &last_residual() const;
    uint64_t tokens() const;
    const std::vector<PortablePlacement> &placements() const;
    size_t weight_bytes() const;
    size_t graph_bytes() const;
    PortableMemoryUsage memory_usage() const;
    size_t state_bytes() const;

  private:
    struct Impl;
    explicit PortableGdnLayer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
