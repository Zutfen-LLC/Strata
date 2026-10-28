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
    size_t expert_staging_cap = 64ull << 20;
};
struct PortableModelReport {
    // Unique mapped shard file bytes; PLE may alias the same backing file.
    size_t mapped_bytes = 0, host_weights = 0, host_state_bound = 0, expert_staging = 0;
    size_t graph_admission_bound = 0;
    size_t host_scheduler = 0;
    std::vector<size_t> device_scheduler, device_workspace_bound, device_admission;
    uint64_t retained_manifest_identity_a = 0, retained_manifest_identity_b = 0;
    // Header/range checks plus selected expert digest verification; no new
    // whole-file hash campaign. PLE mapping is file-backed, not an arena.
    std::string identity_evidence;
};
enum class PortableStop { none, eos, context, max_new };
enum class PortableTokenKind { input, generated };
struct PortableGeneration {
    std::vector<int32_t> tokens;
    PortableStop stop = PortableStop::none;
};
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
    const PortableModelReport &report() const;
    const std::vector<PortablePlacement> &placements() const;

  private:
    struct Impl;
    explicit PortableModelSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
