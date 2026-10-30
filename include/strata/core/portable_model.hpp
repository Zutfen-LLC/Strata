#pragma once
#include "strata/core/portable_gdn.hpp"
#include "strata/core/portable_qsa.hpp"

namespace strata::core {
struct PortableHeadGeometry {
    int64_t hidden = 2560, streams = 4, rank = 320, vocabulary = 0;
    float epsilon = 1e-6f;
};
// Portable global HC/output projection; owns copied small/head weights and
// scheduler allocations. No NativeHead/NativeEmbed or CUDA allocation dependency.
class PortableHead {
  public:
    // guard_logits adds bounded exponent-bit status checks before publishing logits;
    // the model enables it for production state policies or prompt head elision. Host output checks remain.
    static std::unique_ptr<PortableHead> create(PortableHeadGeometry geometry,
                                                const std::vector<PortableTensorView> &weights,
                                                PortableFfnOptions options, std::string &error, bool guard_logits = false);
    // Metadata-only selected-backend estimate, including padding/alignment.
    // Uses the same four tensors and allocator sizing as create; no weight upload.
    static size_t weight_allocation_size(ggml_backend* backend, PortableHeadGeometry geometry,
                                        const std::vector<PortableTensorView>& weights);
    ~PortableHead();
    bool run(const std::vector<float> &streams, std::vector<float> &logits, std::string &error);
    size_t weight_bytes() const;
    size_t graph_bytes() const;
    PortableMemoryUsage memory_usage() const;
    const std::vector<PortablePlacement> &placements() const;

  private:
    struct Impl;
    explicit PortableHead(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
struct PortableExpertDevice {
    size_t accelerator = 0; // index in accelerators, not another physical device
    std::string plan_path; // existing immutable GpuPlan format
    size_t tensor_cap = 256ull << 20;
    size_t workspace_cap = 128ull << 20; // additional backend/runtime allowance, separate from tensors
    size_t free_reserve = 512ull << 20;
};
struct PortableExpertDeviceReport {
    size_t accelerator = 0, assigned = 0;
    uint64_t weight_bytes = 0, tensor_bytes = 0, tensor_scratch_bytes = 0, tensor_cap = 0, workspace_cap = 0;
    uint64_t metadata_bound = 0, staging_peak = 0, free_before = 0, free_after = 0, peak_growth = 0;
    uint64_t requests = 0, entries = 0, row_checks = 0;
    double row_max_abs = 0, row_relative_l2 = 0;
    double infer_rows_ms = 0; // optional host-call wall time, includes existing executor waits
    uint64_t infer_rows_calls = 0;
};
enum class PortableDenseTopology { split, primary };
struct PortableModelOptions {
    size_t context = 16; // bounded initial proof, at most 256 text positions
    int threads = 1, expert_workers = 1;
    // Borrowed accelerator lifetimes must cover the session. Empty = CPU.
    // Default: one owns all layers; two use [0,split) and [split,48).
    // primary: slot 0 owns every dense layer/state/head; slot 1 serves only
    // its explicit expert plan. Expert ownership never follows the dense split.
    std::vector<ggml_backend *> accelerators;
    size_t split = 32;
    PortableDenseTopology dense_topology = PortableDenseTopology::split;
    bool allow_cpu_fallback = false;
    // Separate caps per die; explicit caller limits, never pooled VRAM.
    std::vector<size_t> device_caps;
    // Optional CLI evidence for refusal diagnostics; never used as admission credit.
    std::vector<size_t> device_policy_caps, device_free_reserves;
    size_t host_cap = 32ull << 30;
    size_t layer_weight_cap = 256ull << 20, layer_graph_cap = 256ull << 20;
    size_t head_weight_cap = 1ull << 30, head_graph_cap = 4ull << 30;
    // Comparison oracle only: preserve the former copy/digest on every dispatch.
    bool expert_dispatch_oracle = false;
    size_t expert_staging_cap = 64ull << 20;
    bool expert_check_rows = false, expert_record_routes = false;
    std::string expert_native_q8_observation; // explicit receipt prefix; unqualified, unspecified FP32 denorm
    bool expert_timings = false; // opt-in coarse provider/executor host-call boundaries
    std::vector<PortableExpertDevice> expert_devices; // opt-in; empty preserves CPU path
    bool gdn_resident_weights = false; // opt-in GDN attention only
    bool gdn_resident_state = false; // opt-in; independent of weight residency
    bool dense_resident_weights = false; // opt-in attention, FFN and head weights
    bool gdn_causal_chunk = false; // production/non-PLE; requires batched projections and chunks
    bool gdn_batch_input_projections = false; // requires prompt chunks; non-PLE layers only
    size_t prompt_chunk_size = 0; // default off; 2..8 implies prompt head elision
    bool prompt_head_elision = false; // opt-in: only final prompt position projects vocabulary
    bool qsa_production = false; // device KV/indexer authority; requires strict accelerator and reuse
    bool qsa_reuse_attention = false; // compatibility flag; model sessions always share attention scratch
    bool gdn_production = false; // opt-in device authority; requires strict resident accelerator execution
};
struct PortableExpertWork {
    uint64_t hits = 0, verifications = 0, copied_bytes = 0, digest_bytes = 0;
    uint64_t native_batches = 0, source_invalidations = 0;
    // Explicit batch adapter receipts; jobs count distinct experts, not tokens.
    uint64_t batch_logical_rows = 0, batch_native_jobs = 0, batch_max_nt = 0;
    uint64_t gpu_entries = 0, cpu_entries = 0;
    // Existing ExpertPool host-wall accumulators; phases include synchronization.
    // elapsed encloses the three native phases: never add it to their sum.
    double pool_gu_ms = 0, pool_quant_ms = 0, pool_down_ms = 0, pool_elapsed_ms = 0;
    double provider_ms = 0; // encloses CPU pool and local GPU expert calls; never add overlapping times
    uint64_t provider_calls = 0;
};
// Whole-call wall totals are optional; head outcome counters are unconditional.
// Layer totals include matched provider intervals.
struct PortableModelWork {
    uint64_t heads_executed = 0, heads_skipped = 0; // unconditional, cumulative across reset
    uint64_t completed_forward_calls = 0, completed_layer_calls = 0;
    uint64_t completed_ffn_pre_calls = 0, completed_provider_calls = 0, completed_ffn_post_calls = 0;
    uint64_t logical_tokens = 0, prompt_chunks = 0, prompt_chunk_tokens = 0;
    uint64_t chunk_layer_calls = 0, chunk_ffn_pre_calls = 0, chunk_provider_calls = 0, chunk_ffn_post_calls = 0;
    uint64_t gdn_causal_submissions = 0, gdn_causal_residual_handoffs = 0, gdn_causal_state_commits = 0;
    PortableGdnWork gdn_phases; // nested GDN scopes; enabled only by expert_timings
    double step_ms = 0, gdn_ms = 0, qsa_ms = 0, head_ms = 0;
    double gdn_provider_ms = 0, qsa_provider_ms = 0;
    uint64_t step_calls = 0, gdn_calls = 0, qsa_calls = 0, head_calls = 0;
};
struct PortableModelReport {
    // Unique mapped shard file bytes; PLE may alias the same backing file.
    bool chunk_memory_is_shape_envelope = false;
    size_t mapped_bytes = 0, host_weights = 0, host_state_bound = 0, expert_staging = 0;
    size_t graph_admission_bound = 0;
    size_t host_scheduler = 0;
    size_t gdn_scratch_resources = 0;
    size_t gdn_scratch_host_scheduler = 0, gdn_scratch_device_scheduler = 0;
    size_t qsa_attention_resources = 0;
    size_t qsa_attention_host_scheduler = 0, qsa_attention_device_scheduler = 0;
    std::vector<size_t> device_scheduler, device_workspace_bound, device_token_temporary_bound, device_shared_conversion_bound, device_admission;
    std::vector<size_t> device_state; // actual aligned session state, separate from schedulers/weights
    std::vector<size_t> device_weights; // actual persistent dense weights per physical backend
    uint64_t retained_manifest_identity_a = 0, retained_manifest_identity_b = 0;
    // Header/range checks plus first-use selected expert digests; no new
    // whole-file hash campaign. PLE mapping is file-backed, not an arena.
    std::string identity_evidence;
    std::vector<PortableExpertDeviceReport> expert_devices;
};
enum class PortableStop { none, eos, context, max_new };
enum class PortableTokenKind { input, generated };
struct PortableGeneration {
    std::vector<int32_t> tokens;
    PortableStop stop = PortableStop::none;
};
// Model source files must remain immutable while open (see docs/PORTABLE_MODEL.md).
// One serial text session and its lifetime-owned six-shard model. Errors during
// step poison the session, including failures after earlier layers committed.
// reset() is the only way to recover; caller explicitly replays its token IDs.
class PortableModelSession {
  public:
    static std::unique_ptr<PortableModelSession> open(const std::vector<std::string> &shards,
                                                      const std::string &local_manifest, PortableModelOptions options,
                                                      std::string &error);
    ~PortableModelSession();
    // Input/teacher-forced turn separators never terminate generation.
    // Only a generated tokenizer EOS marks ended(); PLE handles every token
    // with its separate metadata history boundary, regardless of token kind.
    bool step(int32_t token, std::vector<float> &logits, std::string &error,
              PortableTokenKind kind = PortableTokenKind::input);
    // Input-only state advancement. No logits or logit finite check at this
    // position. Layer execution, poison/reset and position commit match step().
    bool advance(int32_t token, std::string &error);
    // Enabled-policy input-only chunk. No completed publication on failure;
    // earlier layers may have advanced, so any execution error poisons/reset.
    bool advance_chunk(const std::vector<int32_t>& tokens, std::string& error);
    bool generate(const std::vector<int32_t> &prompt, size_t max_new, PortableGeneration &result, std::string &error);
    bool reset(std::string &error);
    uint64_t position() const;
    bool poisoned() const;
    bool ended() const;
    int32_t eos() const;
    size_t vocabulary() const;
    // Cumulative across reset; digest_bytes counts input bytes, two FNV passes.
    PortableExpertWork expert_work() const;
    PortableModelWork model_work() const;
    const PortableModelReport &report() const;
    const std::vector<PortablePlacement> &placements() const;

  private:
    struct Impl;
    explicit PortableModelSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
