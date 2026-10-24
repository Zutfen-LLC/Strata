// include/strata/worker/expert_worker.hpp - inferswarm #348: the resident CPU-expert worker service.
//
// Holds one model's CPU-owned expert blobs in a memory-budgeted arena and computes weighted partial sums
// with Strata's native CPU kernels (strata::kernels::cpu), over one TCP connection (loopback by default).
// The service
// is the server half of the wire contract in expert_wire.hpp; see docs/CPU_WORKER.md for the full protocol
// and the minimal local reproduction.
//
// Session rules (#348 scope 2): a SETUP_BEGIN starts a session and voids any previous one; blobs upload as
// bounded in-order chunks and a slot is verified against its declared digest when complete; readiness is
// exposed only after EVERY owned slot is present and verified, and freezes the session; inference frames
// carry ids, weights and activations - never weight bytes or pointers - and only execute slots this worker
// owns.  A disconnect voids the session; the next connection re-runs SETUP_BEGIN (a restart is a new
// session, and old session ids are refused). Opt-in residency retains only a complete verified local
// weight partition; a compatible new LOCAL_SETUP can bind it with fresh session state and READY.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/worker/expert_wire.hpp"
#include "strata/worker/rpc_timing.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::worker {

struct WorkerOptions {
    uint64_t memory_budget = 128ull << 30;   ///< the arena's byte budget; an assigned set over it is refused
    int workers = 0;      ///< ExpertPool worker threads (0: the pool's default plan for this machine)
    int deadline_ms = 10000;   ///< per-frame send/recv deadline
    int verbose = 0;      ///< 1: log each request's identity/status line (still no payloads)
    std::string local_root, local_manifest; // both empty by default; operator paths only
    bool local_upload_fallback = false;
    int local_setup_ms = 600000; // whole local setup budget, in addition to per-batch I/O budget
    int residency_ms = 0; ///< opt-in idle TTL, 1..86400000 ms; 0 preserves session-scoped weights
    bool residency_forever = false; ///< explicit service-owned residency; no idle expiry, never implicit at 0
    uint32_t upload_chunk_max = kUploadChunkMax;   ///< advertised data cap, 1..kUploadChunkMax
    bool accepted_tcp_nodelay = false; ///< opt-in checked TCP_NODELAY on CPU-worker accepted sockets only
    bool rpc_timing = false; ///< opt-in fixed-memory aggregate timing; never logs individual RPCs
    std::string bind_ip = "127.0.0.1"; ///< numeric unicast IPv4 only; never wildcard or DNS
    std::vector<std::string> allow_peer_ips; ///< exact numeric IPv4 peers; required for non-loopback bind
};

/// #348 scope 8: what the service counts.  Identities and statuses are logged; prompts, activations and
/// outputs never are.
struct WorkerCounters {
    uint64_t setup_weight_bytes = 0; ///< network weight bytes accepted into the arena
    uint64_t local_bytes = 0, local_hits = 0, local_misses = 0;
    uint64_t local_read_ns = 0, local_verify_ns = 0, local_setup_ns = 0;
    uint64_t infer_in_bytes = 0;       ///< inference frames received (headers included)
    uint64_t infer_out_bytes = 0;      ///< inference replies sent
    uint64_t requests = 0;             ///< MSG_INFER handled (accepted or rejected)
    uint64_t failures = 0;             ///< frames rejected or connections dropped mid-frame
    uint64_t cpu_service_ns = 0;       ///< time spent inside the native kernels (setup excluded)
    uint64_t cache_hits = 0, reused_bytes = 0, loaded_bytes = 0, verified_bytes = 0;
    uint64_t resident_bytes = 0, peak_arena_bytes = 0, arena_allocations = 0;
    uint64_t cache_expirations = 0, cache_unloads = 0;
    uint16_t last_error = ST_OK;
};

/// The service.  One listener, connections served one at a time, one outstanding request per connection;
/// there is no pipelining and no second concurrent client.
class ExpertWorkerService {
public:
    explicit ExpertWorkerService(WorkerOptions options = {});
    ~ExpertWorkerService();

    /// Binds options.bind_ip:port (0 = a free port, reported back). Non-loopback requires exact peers.
    bool listen(uint16_t& port, std::string& err);

    /// Serves connections until stop() (call from a dedicated thread); each connection to EOF.
    void run();

    /// Wakes run() (from another thread) and lets it return.
    void stop();

    const WorkerCounters& counters() const { return counters_; }
    const RpcTimingCounters& rpc_timing() const { return rpc_timing_; }
    void reset_rpc_timing() { rpc_timing_.reset(); }

    // ---- the execution pieces, exported so tests (and #349's verification) call the same code ----

    /// Weighted per-token accumulation over per-entry rows: partials[t] += mask[i] ? w[i] * rows[i] : 0,
    /// entries in frame order, `acc = acc + w * o`, one pass.  This IS the ordered-reduction contract.
    static void accumulate_partial(int T, int K, const uint8_t* mask, const float* weights,
                                   const float* rows, int n_embd, float* partials);

    /// The equivalent direct native CPU call (#348 scope 5): quantizes each token's activation once with
    /// native_quant_act, evaluates every owned entry with native_gu_rows / native_quant_h /
    /// native_down_rows over the expert's full row range, and accumulates with accumulate_partial.  The
    /// transported service output must be bitwise this.  `blob_of_entry` is T*K pointers, one per entry;
    /// only the mask-1 entries are read.
    static bool direct_native_partial(const strata::kernels::cpu::NativeFmt& fmt, int T, int K,
                                      const uint8_t* mask, const float* weights, const float* activations,
                                      const uint8_t* const* blob_of_entry, std::vector<float>& partials,
                                      std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    WorkerCounters counters_;
    RpcTimingCounters rpc_timing_;
};

}  // namespace strata::worker
