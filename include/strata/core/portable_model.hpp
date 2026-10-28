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
    static std::unique_ptr<PortableHead> create(PortableHeadGeometry geometry,
                                                const std::vector<PortableTensorView> &weights,
                                                PortableFfnOptions options, std::string &error);
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
};
struct PortableModelOptions {
    size_t context = 16; // bounded initial proof, at most 256 text positions
    int threads = 1, expert_workers = 1;
    // Borrowed accelerator lifetimes must cover the session. Empty = CPU.
    // One accelerator owns all layers; two use [0,split) and [split,48).
    std::vector<ggml_backend *> accelerators;
    size_t split = 32;
    bool allow_cpu_fallback = false;
    // Separate caps per die; explicit caller limits, never pooled VRAM.
    std::vector<size_t> device_caps;
    size_t host_cap = 32ull << 30;
    size_t layer_weight_cap = 256ull << 20, layer_graph_cap = 256ull << 20;
    size_t head_weight_cap = 1ull << 30, head_graph_cap = 4ull << 30;
    // Comparison oracle only: preserve the former copy/digest on every dispatch.
    bool expert_dispatch_oracle = false;
    size_t expert_staging_cap = 64ull << 20;
    bool expert_check_rows = false, expert_record_routes = false;
    std::vector<PortableExpertDevice> expert_devices; // opt-in; empty preserves CPU path
};
struct PortableExpertWork {
    uint64_t hits = 0, verifications = 0, copied_bytes = 0, digest_bytes = 0;
    uint64_t native_batches = 0, source_invalidations = 0;
    uint64_t gpu_entries = 0, cpu_entries = 0;
};
struct PortableModelReport {
    // Unique mapped shard file bytes; PLE may alias the same backing file.
    size_t mapped_bytes = 0, host_weights = 0, host_state_bound = 0, expert_staging = 0;
    size_t graph_admission_bound = 0;
    size_t host_scheduler = 0;
    std::vector<size_t> device_scheduler, device_workspace_bound, device_token_temporary_bound, device_shared_conversion_bound, device_admission;
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
    bool generate(const std::vector<int32_t> &prompt, size_t max_new, PortableGeneration &result, std::string &error);
    bool reset(std::string &error);
    uint64_t position() const;
    bool poisoned() const;
    bool ended() const;
    int32_t eos() const;
    // Cumulative across reset; digest_bytes counts input bytes, two FNV passes.
    PortableExpertWork expert_work() const;
    const PortableModelReport &report() const;
    const std::vector<PortablePlacement> &placements() const;

  private:
    struct Impl;
    explicit PortableModelSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
