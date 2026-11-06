#pragma once
#include "strata/core/portable_ple.hpp"

namespace strata::core {
// Native SessionState::gdn transient family. Borrowed backends must outlive all layers.
class PortableGdnScratch {
  public:
    static std::shared_ptr<PortableGdnScratch> create(PortableFfnOptions options, std::string &error, bool additional_scheduler = false);
    ~PortableGdnScratch();
    size_t graph_bytes() const;
    PortableMemoryUsage memory_usage() const;
    static size_t host_metadata_bound(bool additional_scheduler = false);
  private:
    struct Impl;
    explicit PortableGdnScratch(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class PortableGdnLayer;
};
enum class PortableGdnPolicy { eager_diagnostic, device_production };
struct PortableGdnOptions {
    static constexpr size_t native_ple_value_host_bound = 16u << 20; // BF16 role and bounded controls
    static constexpr size_t native_recurrence_host_bound=4u<<20;
    static constexpr size_t native_preparation_host_bound=2u<<20; // streamed conversion/preflight/first receipts
    static constexpr size_t native_input_projection_host_bound=1u<<20;
    static constexpr size_t native_output_projection_host_bound=1u<<20;
    static constexpr size_t native_ple_key_host_bound=1u<<20;
    static constexpr size_t native_ple_conv_host_bound=2u<<20;
    static constexpr size_t native_ple_gate_host_bound=1u<<20;
    static constexpr size_t native_ple_norm_host_bound=1u<<20;
    static constexpr size_t native_hc_host_bound = 1u << 20; // explicit metadata/preflight/trace envelope
    // Largest streamed BF16 conversion/readback is 6,553,600 B, plus the
    // existing 1 MiB metadata/trace bound; no simultaneous down/up host banks.
    static constexpr size_t native_hc_read_host_bound = 8u << 20;
    static constexpr size_t native_hc_pending_host_bound = native_hc_read_host_bound + (128u << 10);
    PortableFfnOptions execution;
    // Experimental fixed T1 eager/resident-state path; private or shared serial scratch.
    bool native_ple_value_t1 = false; // opt-in layer-1 PLE BF16 value projection
    std::string native_ple_value_receipt;
    bool native_recurrence_t1 = false;
    bool native_preparation_t1=false;
    bool native_input_projection_t1=false; // ordinary T1 native QKV and gate-input projection
    std::string native_input_projection_receipt;
    bool native_input_projection_record_first=false;
    bool native_output_projection_t1=false;
    std::string native_output_projection_receipt;
    bool native_ple_gate_t1=false; // default-off layer-1 PLE gate observation
    std::string native_ple_gate_receipt;
    bool native_ple_hc_read_t1=false; // materialized PLE residual, apply=false
    bool native_ple_norm_t1=false; // bounded native-order observation, not bitwise CUDA parity
    std::string native_ple_norm_receipt;
    bool native_ple_key_t1=false; // layer-1 Q8_0 key projection, diagnostic and default off
    std::string native_ple_key_receipt;
    bool native_ple_conv_t1=false; // layer-1 convolution/history experiment, default off
    std::string native_ple_conv_receipt;
    bool native_output_projection_record_first=false;
    bool native_preparation_weights=false; // shared control/on BF16 replacement, ordinary T1 only
    std::string native_preparation_receipt;
    bool native_preparation_record_first=false;
    std::string native_recurrence_receipt;
    bool native_recurrence_record_first=false; // bounded model layer0 own-input/commit receipt
    bool resident_ffn_weights = false; // selected FFN bank only; attention placement unchanged
    bool defer_ffn_hc_write = false; // layer1 outgoing ordinary boundary; PLE remains unchanged
    bool native_hc_norm_inject = false; // experimental layer-0 serial T1 only; mixed remains GGML
    bool native_hc_mix = false; // experimental attention read; requires native norm/injection
    bool native_hc_ffn_write = false; // materialize actual layer0 FFN output before next PLE consumer
    bool native_hc_ffn_read = false; // actual layer-0 GDN producer -> native pending FFN read, T1 only
    bool native_hc_pending = false; // explicit tuple: layer0 diagnostic or ordinary layer2 consumer
    std::string native_hc_receipt_prefix;
    bool causal_chunk = false; // production/non-PLE T2..8; requires batched projections
    bool batch_input_projections = false; // opt-in chunk T>1, non-PLE only; T1 stays serial
    size_t state_budget = 16ull << 20; // host state/scratch bound; policy-specific mirrors
    std::shared_ptr<PortablePleTable> ple_table; // enables configured layer-1 PLE; host only
    bool resident_state = false; // session-lifetime GDN/PLE state; diagnostic host mirrors stay validated
    size_t resident_state_budget = 4ull << 20; // aligned persistent backend inputs, separate from graphs
    bool resident_attention_weights = false; // opt-in; CPU execution retains CPU allocation
    PortableGdnPolicy policy = PortableGdnPolicy::eager_diagnostic;
    std::shared_ptr<PortableGdnScratch> scratch; // omit for private standalone execution
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
    PortableFfnHcTrace ffn_hc;
    bool native_hc_norm_inject = false;
    bool native_hc_mix = false;
    bool native_hc_pending = false;
    std::vector<float> hc_pending_input, hc_previous_output, hc_previous_inject, hc_pending_gate, hc_pending_exp;
    std::vector<float> hc_residual, hc_normalized, hc_rs, hc_inject;
    std::vector<float> hc_low_pre, hc_low, hc_low_exp, hc_up_pre, hc_gates, hc_up_exp, hc_mixed;
    std::vector<float> qkv, convolution, query, key, value, decay, beta, attention;
    std::array<uint32_t, 16> ple_rows{};
    std::vector<float> ple_embedding, ple_key, ple_value, ple_gate, ple_normalized, ple_convolution, ple_residual;
};

class PortableGdnLayer;
// Producer-issued value object. Copies share immutable values and a lifetime/generation
// guard; only PortableGdnLayer can construct one.
class PortableGdnSnapshot {
  public:
    int64_t layer() const { return layer_; }
    uint64_t position() const { return position_; }
    uint64_t generation() const { return generation_; }
    const PortableHcPendingWrite& tuple() const { return tuple_; }
    bool valid_for(const PortableGdnLayer& expected_owner, int64_t consumer_layer, uint64_t position) const;
  private:
    struct Owner;
    PortableGdnSnapshot(std::shared_ptr<Owner> owner, int64_t layer, uint64_t position,
                        uint64_t generation, const PortableHcPendingWrite& tuple);
    std::shared_ptr<Owner> owner_;
    int64_t layer_;
    uint64_t position_, generation_;
    PortableHcPendingWrite tuple_;
    friend class PortableGdnLayer;
};

// Complete qwen4exp non-QSA layers 0..47, T1 and causal T1..8 chunks: configured PLE, HC/GDN,
// PortableFfn. Exact fleet geometry only. QSA layers are rejected.
class PortableGdnLayer {
    friend class PortableGdnSnapshot;
  public:
    static std::unique_ptr<PortableGdnLayer> create(int64_t layer, const std::vector<PortableTensorView> &tensors,
                                                    PortableGdnOptions options, std::string &error);
    static std::unique_ptr<PortableGdnLayer> from_gguf(const GgufModel &model, int64_t layer,
                                                       PortableGdnOptions options, std::string &error);
    ~PortableGdnLayer();
    PortableGdnLayer(const PortableGdnLayer &) = delete;
    PortableGdnLayer &operator=(const PortableGdnLayer &) = delete;
    // Build availability only; not a physical-device or arithmetic qualification.
    static bool native_hc_compiled();
    bool reset(std::string &error);
    bool poisoned() const;
    PortableGdnPolicy policy() const;
    // A fresh token residual (four streams) is supplied on every call. Public
    // metadata and last_residual publish only after FFN and device commit succeed.
    // Production failure may leave changed device state; reset/replay is required.
    bool step(const std::vector<float> &residual, const PortableExpertRows &dispatch, PortableRoute &route,
              std::string &error, PortableGdnTrace *trace = nullptr, PortableGdnWork *work = nullptr);
    // Diagnostic T1 pending tuple. Original caller R/bo/inject are immutable;
    // R-prime is used by BOTH HC and graph residual consumers. Public state and
    // last_residual commit only after the existing complete layer transaction.
    // Requires native_hc_pending + composed attention roles (ordinary native boundary; PLE remains explicit).
    bool step_pending(const std::vector<float>& residual,const std::vector<float>& previous_output,
                      const std::vector<float>& previous_injection,const PortableExpertRows& dispatch,
                      PortableRoute& route,std::string& error,PortableGdnTrace* trace=nullptr,
                      PortableGdnWork* work=nullptr);
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
    // Allocate/replace one chunk shape without advancing sequence state.
    // Caller pre-admits the module bound before invoking this planning seam.
    bool prepare_chunk(size_t tokens, std::string& error);
    PortableLayerChunkWork chunk_work() const;
    // Actual projection graph submissions/columns, including work before a failed
    // chunk; reset preserves these cumulative execution receipts.
    uint64_t input_projection_calls() const;
    uint64_t native_ple_key_calls() const;
    uint64_t native_ple_gate_calls() const;
    uint64_t native_ple_norm_calls() const;
    uint64_t native_ple_conv_calls() const;
    uint64_t native_ffn_hc_pending_calls() const;
    uint64_t native_ffn_hc_write_calls() const;
    const PortableHcPendingWrite& pending_write() const;
    // Null unless the latest attempt committed a deferred tuple. Owned snapshots
    // survive later successful commits, but not producer reset/destruction.
    std::shared_ptr<const PortableGdnSnapshot> pending_snapshot(std::string& error) const;
    // The caller must identify the actual expected producer, not just its layer.
    // Validate its identity and generation, adjacent layer and consumer position.
    bool step_pending(const PortableGdnLayer& producer, const PortableGdnSnapshot& snapshot, const PortableExpertRows& dispatch,
                      PortableRoute& route, std::string& error, PortableGdnTrace* trace=nullptr,
                      PortableGdnWork* work=nullptr);
    uint64_t deferred_hc_outputs() const;
    uint64_t native_convolution_calls() const;
    uint64_t native_ab_calls() const;
    uint64_t native_recurrence_calls() const; // completed component work, even if later FFN refuses
    uint64_t native_hc_pending_calls() const; // paid successful work, including later graph/provider refusal
    uint64_t native_hc_mix_calls() const;
    uint64_t native_hc_calls() const; // actual successful component calls; cumulative across reset
    uint64_t causal_chunk_calls() const;
    uint64_t causal_residual_handoffs() const;
    uint64_t causal_state_commits() const;
    uint64_t input_projection_columns() const;
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
    size_t native_input_projection_bytes() const;
    uint64_t native_input_projection_calls(unsigned counter) const;
    size_t native_output_projection_bytes() const;
    uint64_t native_output_projection_calls(unsigned counter) const;
    const PortableGdnScratch *scratch_owner() const;

  private:
    struct Impl;
    bool step_impl(int32_t token,const std::vector<float>& residual,
                   const std::vector<float>* previous_output,const std::vector<float>* previous_injection,
                   const PortableExpertRows& dispatch,PortableRoute& route,std::string& error,
                   PortableGdnTrace* trace,PortableGdnWork* work);
    explicit PortableGdnLayer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
