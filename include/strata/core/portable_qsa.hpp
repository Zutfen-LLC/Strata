#pragma once
#include "strata/core/portable_ffn.hpp"

namespace strata::core {
struct PortableQsaOptions {
    // graph_budget covers scheduler buffers/copies, CPU plans and a conservative
    // pinned-Vulkan tensor-workspace allowance. Larger active histories may
    // need a higher limit because the scheduler prices each view consumer.
    // Driver pipelines and other users of the borrowed backend are caller-owned.
    PortableFfnOptions execution;
    size_t capacity = 4096;
    size_t state_budget = 64ull << 20;
    // Opt-in capacity-primed scheduler/scratch reuse; exact active shapes, host KV staging.
    bool reuse_attention = false;
    // Device-authoritative state; requires reuse and strict selected accelerator.
    bool production = false;
    bool device_combine = false; // production only; append combine to exact active attention graph
};
struct PortableQsaGraphWork {
    uint64_t attention_submissions = 0, combine_submissions = 0, device_handoffs = 0;
};
// Explicit diagnostic download, including unused capacity, never a host mirror.
struct PortableQsaState {
    std::vector<uint16_t> keys, values;
    std::vector<float> pooled, spare, tail, bias;
    std::vector<int32_t> mapping;
};
struct PortableQsaTrace {
    std::vector<float> query, key, value, gate, index_query, index_raw_key;
    std::vector<float> pooled_key, cell_scores, selected, attention, gated_attention, attention_residual;
};
// Opt-in complete qwen4exp QSA layers 3/7/.../47, one text sequence, T1 or causal T1..8 chunks.
// Positions must be contiguous starting at zero. K/V retain the CUDA FP16
// storage policy; raw indexer tail and normalized/rotated pooled keys are F32.
class PortableQsaLayer {
  public:
    static std::unique_ptr<PortableQsaLayer> create(int64_t layer, const std::vector<PortableTensorView> &weights,
                                                    PortableQsaOptions options, std::string &error);
    static std::unique_ptr<PortableQsaLayer> from_gguf(const GgufModel &model, int64_t layer,
                                                       PortableQsaOptions options, std::string &error);
    // Metadata-only exact estimate for this backend, before allocating layer/device storage.
    // Throws on invalid capacity/backend or size/alignment overflow.
    static size_t device_state_allocation_size(ggml_backend *backend, size_t capacity);
    ~PortableQsaLayer();
    PortableQsaLayer(const PortableQsaLayer &) = delete;
    PortableQsaLayer &operator=(const PortableQsaLayer &) = delete;
    bool step(uint64_t position, const std::vector<float> &residual, const PortableExpertRows &dispatch,
              PortableRoute &route, std::string &error, PortableQsaTrace *trace = nullptr);
    // Opt-in T1..8 causal layer chunk. Token IDs bind the single-sequence
    // interface (QSA uses contiguous positions); rows are [H,HC,T]. Failure
    // preserves public receipts and poisons either policy until reset/replay.
    bool run_chunk(const std::vector<int32_t>& tokens, uint64_t position0,
                   std::vector<float>& residual_rows, const PortableExpertRowsBatch& dispatch,
                   std::vector<PortableRoute>& routes, std::string& error);
    // Allocate/replace one chunk shape without advancing sequence state.
    // Caller pre-admits the module bound before invoking this planning seam.
    bool prepare_chunk(size_t tokens, std::string& error);
    PortableLayerChunkWork chunk_work() const;
    // Actual successful graph submissions since construction (includes failed steps).
    PortableQsaGraphWork graph_work() const;
    bool reset(std::string &error);
    uint64_t tokens() const;
    bool poisoned() const;
    // Explicit full-capacity download; poisoned state may include uncommitted writes.
    bool inspect_state(PortableQsaState &state, std::string &error) const;
    // Host history getters below require eager diagnostics; production throws.
    // Token-major [cell,kv_head,256]. Only committed cells are exposed.
    const std::vector<uint16_t> &keys() const;
    const std::vector<uint16_t> &values() const;
    // [complete_block,128], followed by the moving spare/dead key:
    // weighted RMS of raw cell0, RoPE at position0. Zero only before first step.
    const std::vector<float> &pooled_keys() const;
    // Four raw keys; unused entries are zero, including the whole tail at
    // completion.
    const std::vector<float> &indexer_tail() const;
    const std::vector<float> &last_residual() const;
    const std::vector<PortablePlacement> &placements() const;
    size_t weight_bytes() const;
    // Scheduler allocations plus admitted workspace allowances, not total RSS/VRAM.
    size_t graph_bytes() const;
    PortableMemoryUsage memory_usage() const;
    // Host bound plus actual aligned device state; memory_usage separates them.
    size_t state_bytes() const;

  private:
    struct Impl;
    explicit PortableQsaLayer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
