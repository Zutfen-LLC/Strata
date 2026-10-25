// include/strata/worker/expert_tier.hpp - inferswarm #349: the engine side of the remote CPU-expert mode.
//
// One opt-in tier that turns a #348 `strata-expert-worker` into the executor of a model's CPU-owned expert
// complement.  Everything here is CPU-only (it links strata_expert_worker and strata_kernels_cpu, never a
// GPU target), so the whole lifecycle - setup streaming, bounded staging, request construction, failure
// latching - is testable on a machine without a toolkit; the CUDA engine glue (expert_source.cpp,
// remote_expert_opt.cu, generate.cpp) only moves pointers between this tier and the verifier's buffers.
//
// THE MERGE CONTRACT (what the engine does with the reply, stated once so the engine side and the tests
// agree): the worker returns ONE already-weighted F32 partial sum per token - `acc = acc + w * o` over the
// entries it owns, in frame order, exactly `ExpertWorkerService::direct_native_partial`.  The engine zeroes
// those entries' per-entry rows (they leave the GPU's weighted combine as zero) and adds the partial on top
// of the combined result, after every helper tier's partial.  The masks are disjoint by construction: an
// entry is GPU-cache, helper, worker or local-pool work, never two.
#pragma once

#include "strata/worker/expert_client.hpp"
#include "strata/worker/rpc_timing.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::worker {
class CpuCapture;

/// The opt-in remote CPU-expert tier (inferswarm #349).  Default-off: the engine constructs it only for
/// `--remote-cpu-experts`; with no tier attached the dispatch's worker hooks are dead branches and every
/// existing path is unchanged.
class ExpertWorkerTier {
public:
    /// `model_name` is truncated to the manifest's 47 characters (the worker's log only).
    struct Open {
        std::string host = "127.0.0.1";   ///< the worker, or an SSH tunnel's local end
        uint16_t port = 7431;
        const LocalManifest *local = nullptr; // trusted reusable manifest, opt-in
        bool local_fallback = false;
        bool resident = false; // negotiate persistent cache; requires local manifest and worker TTL
        bool resident_reuse_only = false; // diagnostic: refuse empty cache, never load/upload
        bool diagnostic_regroup = false; // diagnostic: sequential virtual peers on the full cache
        int setup_ms = 600000;
        std::function<bool()> cancelled;
        int deadline_ms = 10000;          ///< per-frame deadline (setup, upload, infer)
        bool rpc_timing = false;           ///< fixed-memory per-host aggregates; default path has no clocks
        const char* model_name = "model";
        uint16_t gu_type = 8, d_type = 8; ///< GGML_TYPE_Q8_0 - the one layout the worker runs
        uint16_t n_layers = 0, n_expert = 0, n_embd = 0, n_ff = 0;
        std::vector<uint8_t> partition;   ///< n_layers * ceil(n_expert/8) bitmap, layer-major
        /// Fills exactly `bytes` (the SETUP_ACK's blob size) of one owned blob into `dst`; false aborts
        /// the setup.  The engine passes FileExpertSource::copy_blob behind this, so the upload streams
        /// through ONE blob-sized staging buffer - never a full expert mirror on the engine host.
        std::function<bool(int64_t layer, int64_t expert, uint8_t* dst, uint32_t bytes)> fetch;
    };

    ExpertWorkerTier() = default;
    ~ExpertWorkerTier();
    ExpertWorkerTier(const ExpertWorkerTier&) = delete;
    ExpertWorkerTier& operator=(const ExpertWorkerTier&) = delete;

    /// connect -> SETUP_BEGIN -> every owned blob through `fetch` -> READY.  Refuses (false, `err`) on any
    /// typed status, a fetch failure or a deadline; nothing is half-open.  Must succeed before inference.
    bool open(const Open& in, std::string& err);
    void close();

    bool holds(int64_t layer, int32_t expert) const {  ///< this tier's frozen partition
        return layer >= 0 && expert >= 0 && (size_t) layer < n_layers_ && (size_t) expert < n_expert_ &&
               partition_[(size_t) layer * stride_ + (size_t) expert / 8] & (1u << (expert % 8));
    }
    bool ready() const { return ready_; }
    /// Any refusal, deadline or disconnect latches dead: the engine must abort the request, never retry
    /// silently (a worker restart voided the session; only a full open() again is valid).
    bool dead() const { return dead_; }
    const std::string& dead_why() const { return dead_why_; }

    /// One window's request, posted AFTER the engine published the GPU's own work so the round trip
    /// overlaps the local CPU jobs.  `owned` is n_tok*k entries (1 = this worker executes the entry);
    /// entries not owned must not be marked or the worker refuses (`not-owned`) and the tier dies.
    /// Zero marked entries post nothing and return true - check `posted()`.
    bool post(int64_t layer, const float* x, const int32_t* ids, const float* weights, const uint8_t* owned,
              int n_tok, int k, std::string& err);
    bool posted() const { return posted_; }
    /// Whether the LAST window sent a frame and finish() served its reply (a zero-work window: false,
    /// and `partials()` is then not this window's - the contribution reads as zero).
    bool served() const { return served_; }

    /// Waits for the persistent request thread.  True: the window is served - `partials()` when the last post() actually
    /// sent a frame, otherwise (a zero-work window) the worker contributes nothing and the engine treats
    /// the partial as zero.  False: the tier is dead; the engine aborts without committing.  Always safe
    /// to call after post() - also after a post() that returned false.
    bool finish(std::string& err);
    /// The last reply's already-weighted partials (n_tok * n_embd).  Valid only when the last post()
    /// returned true with `posted()` true and finish() then returned true.
    const float* partials() const { return reply_.partials.data(); }
    // Diagnostic only. Bound before inference, lifetime exceeds this tier; never touched by socket thread.
    void set_capture(CpuCapture* p) { capture_=p; }
    CpuCapture* capture() const { return capture_; }

    /// The last worker STATUS (false when unreachable; the tier then stays usable if it was not dead).
    bool status(WorkerStatusInfo& out, std::string& err);
    // Group-owned idle probe. Skip an occupied slot; serialize the socket with post/status.
    bool keepalive_if_idle(std::string& err);

    struct Stats {
        uint64_t setup_blobs = 0;          ///< owned blobs available (loaded/verified or reused)
        uint64_t setup_weight_bytes = 0;   ///< blobs accepted by the worker (the SETUP_ACK's blob size x owned)
        bool cache_hit = false;
        uint64_t reused_bytes = 0, session = 0;
        uint64_t setup_network_bytes = 0, local_bytes = 0, local_hits = 0, local_misses = 0;
        double connect_setup_ms = 0, local_ms = 0, fetch_upload_ms = 0, ready_ms = 0;
        uint64_t local_read_ns = 0, local_verify_ns = 0;
        uint64_t requests = 0;             ///< nonempty INFER windows (normally one frame; diagnostic wire counts below)
        uint64_t entries = 0;              ///< routed (token, expert) entries this tier executed
        uint64_t failures = 0;
        uint64_t regroup_peer_requests[2]{}; // diagnostic wire calls; requests above counts logical windows
        uint64_t communication_thread_id = 0; ///< OS thread id for affinity inspection (0 before first work)
        uint64_t communication_thread_starts = 0; ///< one for the tier's lifetime after first nonempty post
        double setup_ms = 0;               ///< open()'s wall time (connect + upload + ready), kept apart
        double request_ms = 0;             ///< post->finish wall time, client side (network + service)
        double wait_ms = 0;                ///< the part of that the caller waited after its own work
        double queue_ms = 0;               ///< publish until the persistent communication thread takes the slot
        double exchange_ms = 0;            ///< client INFER send/reply duration on that thread
    };
    const Stats& stats() const { return st_; }
    const RpcTimingCounters& rpc_timing() const { return rpc_timing_; }
    bool rpc_timing_enabled() const { return rpc_timing_.enabled(); }
    void reset_rpc_timing() { rpc_timing_.reset(); }
    void record_result_merge(uint64_t ns) { rpc_timing_.add_phase(RpcTimingPhase::ClientResultMerge, ns); }
    /// One evidence line for the engine's report (identities and counters, never payloads).
    void report(std::string& out) const;

private:
    friend struct RpcWorkspaceTestAccess;
    void run_requests();
    void die(const std::string& why);

    ExpertWorkerClient client_;
    std::vector<uint8_t> partition_;
    uint16_t n_layers_ = 0, n_expert_ = 0, n_embd_ = 0;
    size_t stride_ = 0;
    uint64_t model_id_ = 0, partition_id_ = 0;
    uint64_t sequence_ = 0;
    uint64_t posted_ns_ = 0;
    bool ready_ = false;
    std::atomic<bool> dead_{false};
    std::string dead_why_;

    // One persistent communication thread and one in-flight request slot.  The thread is created by the
    // first nonempty post(), not open(), so it inherits the same dispatcher affinity as the former
    // per-request threads.  No affinity is changed here; communication_thread_id exposes it for inspection.
    mutable std::mutex request_mutex_;
    std::condition_variable request_wake_;
    std::condition_variable request_done_;
    std::thread th_;
    bool stopping_ = false;
    bool pending_ = false;
    bool busy_ = false;
    InferRequest req_;
    InferReply reply_;
    bool posted_ = false;
    bool served_ = false;
    Stats st_;
    RpcTimingCounters rpc_timing_;
    CpuCapture* capture_=nullptr;
    bool diagnostic_regroup_=false;
    std::vector<int8_t> regroup_owners_;
};

}  // namespace strata::worker
