#pragma once
#include "strata/core/portable_ffn.hpp"
#include <stdexcept>

namespace strata::core {
// Session-owned attention execution only. Backend handles are borrowed and must
// outlive this resource. Each operation borrows exclusively through its commit.
class PortableQsaAttention {
  public:
    static std::shared_ptr<PortableQsaAttention> create(PortableFfnOptions options, std::string &error);
    ~PortableQsaAttention();
    static size_t host_metadata_bound();
    PortableMemoryUsage memory_usage() const;
    size_t graph_bytes() const;
  private:
    struct Impl;
    explicit PortableQsaAttention(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class PortableQsaLayer;
};
struct PortableQsaOptions {
    static constexpr size_t native_hc_host_bound = PortableFfnOptions::native_hc_pending_host_bound + (128u<<10);
    static constexpr size_t native_norm_rope_host_bound=2u<<20;
    static constexpr size_t native_indexer_host_bound=1u<<20;
    bool native_indexer_weights=false, native_indexer_t1=false, native_indexer_record_first=false;
    static constexpr size_t native_indexer_projection_host_bound=4u<<20;
    bool native_indexer_projection_t1=false,native_indexer_projection_record_first=false;
    std::string native_indexer_projection_receipt_prefix;
    std::string native_indexer_receipt_prefix;
    static constexpr size_t native_attention_host_bound=8u<<20;
    bool native_attention_banks=false,native_attention_t1=false,native_attention_record_first=false;
    static constexpr size_t native_wo_host_bound=1u<<20;
    bool native_wo_t1=false,native_wo_record_first=false;
    std::string native_wo_receipt_prefix;
    static constexpr size_t native_qkv_host_bound=1u<<20;
    bool native_qkv_t1=false,native_qkv_record_first=false;
    std::string native_qkv_receipt_prefix;
    std::string native_attention_receipt_prefix;
    static constexpr size_t native_select_host_bound=1u<<20;
    // Existing fixed allowances cover the <=256 path. For larger native
    // histories add capacity-priced copies instead of assuming those fixed
    // envelopes still cover the detached candidates and capture buffers.
    static size_t native_indexer_host_for(size_t cells) {
        if (cells > 4096) throw std::runtime_error("native QSA: indexer capacity exceeds 4096");
        return native_indexer_host_bound + (cells > 256 ? 2 * (512 + (cells / 4 + 1) * 128 + 1) * sizeof(uint32_t) : 0);
    }
    static size_t native_select_host_for(size_t cells) {
        if (cells > 4096) throw std::runtime_error("native QSA: selection capacity exceeds 4096");
        return native_select_host_bound + (cells > 256 ? 2 * (cells / 4 + 2) * 128 * sizeof(uint32_t) + 2 * cells * sizeof(float) + 2051 * sizeof(int32_t) : 0);
    }
    static size_t native_attention_host_for(size_t cells) {
        if (cells > 4096) throw std::runtime_error("native QSA: attention capacity exceeds 4096");
        return native_attention_host_bound + (cells > 256 ? 2 * ((cells + 3) / 4) * 2048 * sizeof(uint16_t) : 0);
    }
    static size_t eager_state_growth(size_t cells) {
        if (cells > 4096) throw std::runtime_error("native QSA: capacity exceeds 4096");
        if (cells <= 256) return 0;
        return (cells - 256) * (4 * 512 * sizeof(uint16_t) + 8 * sizeof(float)) +
               (cells / 4 - 256 / 4) * 128 * 3 * sizeof(float);
    }
    bool native_select_banks=false,native_select_t1=false,native_select_record_first=false;
    std::string native_select_receipt_prefix;
    bool native_norm_rope_weights=false; // matched explicit resident F32 roles/candidate reserve
    bool native_norm_rope_t1=false;
    bool native_norm_rope_record_first=false;
    std::string native_norm_rope_receipt_prefix;
    bool native_hc_read = false; // opt-in serial family; native pending/norm/mix method
    bool native_hc_pending = false; // require explicit incoming R/BO/injection tuple
    std::string native_hc_receipt_prefix;
    bool native_hc_ffn_read = false; // actual post-output-projection tuple
    bool native_hc_ffn_write = false; // materialized endpoint/head control
    bool defer_ffn_hc_write = false; // ordinary outgoing native boundary

    // graph_budget covers scheduler buffers/copies, CPU plans and a conservative
    // pinned-Vulkan tensor-workspace allowance. Larger active histories may
    // need a higher limit because the scheduler prices each view consumer.
    // Driver pipelines and other users of the borrowed backend are caller-owned.
    PortableFfnOptions execution;
    bool resident_ffn_weights = false; // selected FFN bank only; attention placement unchanged
    size_t capacity = 4096;
    size_t state_budget = 64ull << 20;
    // Opt-in capacity-primed scheduler/scratch reuse; exact active shapes, host KV staging.
    bool reuse_attention = false;
    // Device-authoritative state; requires reuse and strict selected accelerator.
    bool production = false;
    // Optional session owner; requires reuse. Omit for private standalone scratch.
    std::shared_ptr<PortableQsaAttention> attention;
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
    PortableFfnHcTrace ffn_hc;
    std::vector<float> projected_output; // actual post-attn_output.weight producer
    PortableFfnHcTrace hc; // existing HC tuple/own-input field representation
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
    bool step_pending(uint64_t position,const PortableHcPendingWrite& pending,const PortableExpertRows& dispatch,
                      PortableRoute& route,std::string& error,PortableQsaTrace* trace=nullptr);
    uint64_t native_norm_rope_calls() const;
    uint64_t native_indexer_calls() const;
    uint64_t native_indexer_projection_calls(unsigned stage) const;
    uint64_t native_score_calls() const;
    uint64_t native_select_calls() const;
    uint64_t native_attention_calls(unsigned stage) const;
    uint64_t native_wo_calls(unsigned stage) const;
    uint64_t native_qkv_calls(unsigned stage) const;
    const std::vector<uint32_t>& native_indexer_state() const;
    uint64_t native_hc_pending_calls() const;
    uint64_t native_ffn_hc_pending_calls() const;
    uint64_t native_ffn_hc_write_calls() const;
    uint64_t deferred_hc_outputs() const;
    const PortableHcPendingWrite& pending_write() const;
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
    // Default: four F32 raw keys, cleared on completion. Native indexer opt-in:
    // three FP16-rounded/F32-expanded rows, retained across block completion.
    const std::vector<float> &indexer_tail() const;
    const std::vector<float> &last_residual() const;
    // Completed final FFN producer, borrowed until reset or the next step.
    ggml_tensor* final_residual_tensor() const;
    PortableFinalHcResidual final_residual_identity() const;
    bool record_head_producer(const PortableFinalHcResidual& expected, std::string& error);
    const std::vector<PortablePlacement> &placements() const;
    const PortableQsaAttention *attention_owner() const;
    size_t weight_bytes() const;
    // Scheduler allocations plus admitted workspace allowances, not total RSS/VRAM.
    size_t graph_bytes() const;
    PortableMemoryUsage memory_usage() const;
    // Host bound plus actual aligned device state; memory_usage separates them.
    size_t state_bytes() const;

  private:
    bool step_impl(uint64_t position,const std::vector<float>& residual,const PortableHcPendingWrite* pending,
                   const PortableExpertRows& dispatch,PortableRoute& route,std::string& error,PortableQsaTrace* trace);
    struct Impl;
    explicit PortableQsaLayer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core
