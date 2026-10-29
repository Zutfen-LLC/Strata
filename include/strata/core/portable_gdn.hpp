#pragma once
#include "strata/core/portable_ple.hpp"

namespace strata::core {
enum class PortableGdnPolicy { eager_diagnostic, device_production };
struct PortableGdnOptions {
    PortableFfnOptions execution;
    size_t state_budget = 16ull << 20; // host state/scratch bound; policy-specific mirrors
    std::shared_ptr<PortablePleTable> ple_table; // enables configured layer-1 PLE; host only
    bool resident_state = false; // session-lifetime GDN/PLE state; diagnostic host mirrors stay validated
    size_t resident_state_budget = 4ull << 20; // aligned persistent backend inputs, separate from graphs
    bool resident_attention_weights = false; // opt-in; CPU execution retains CPU allocation
    PortableGdnPolicy policy = PortableGdnPolicy::eager_diagnostic;
};
// Cumulative host-call wall scopes. Compute is nested in attention/exchange;
// full FFN includes the provider. Reset does not clear caller-owned totals.
struct PortableGdnWork {
    double attention_exchange_ms = 0, attention_compute_ms = 0, ffn_ms = 0;
    // Candidate-only reads: diagnostic state/history plus residual; production
    // reads only the residual. These count explicit reads, excluding backend
    // staging internal to copies and scheduler execution. Readback and finite scopes are disjoint within exchange; trace
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

// Complete qwen4exp non-QSA layers 0..47, T1 and causal T1..8 chunks: configured PLE, HC/GDN,
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
    bool poisoned() const;
    PortableGdnPolicy policy() const;
    // A fresh token residual (four streams) is supplied on every call. Public
    // metadata and last_residual publish only after FFN and device commit succeed.
    // Production failure may leave changed device state; reset/replay is required.
    bool step(const std::vector<float> &residual, const PortableExpertRows &dispatch, PortableRoute &route,
              std::string &error, PortableGdnTrace *trace = nullptr, PortableGdnWork *work = nullptr);
    // Token-aware entry for configured PLE. Supply the configured image token ID for
    // image positions. EOS cuts hash predecessors on subsequent tokens; it
    // does not clear recurrent/convolution state. reset() starts a new sequence.
    bool step(int32_t token, const std::vector<float> &residual, const PortableExpertRows &dispatch,
              PortableRoute &route, std::string &error, PortableGdnTrace *trace = nullptr, PortableGdnWork *work = nullptr);
    // Opt-in causal layer chunk, T1..8. residual_rows is [H,HC,T] and is
    // replaced only on success. Position must match tokens(). Failed chunks
    // poison either policy until reset/replay; public metadata stays committed.
    bool run_chunk(const std::vector<int32_t>& tokens, uint64_t position0,
                   std::vector<float>& residual_rows, const PortableExpertRowsBatch& dispatch,
                   std::vector<PortableRoute>& routes, std::string& error);
    PortableLayerChunkWork chunk_work() const;
    // Tensor-state observers require eager_diagnostic and throw in production.
    // Token history and last_residual remain current in both policies. Production
    // rejects traces, poisons on any failed step, and requires successful reset.
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
