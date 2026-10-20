// include/strata/worker/expert_client.hpp - inferswarm #348: the engine side of the worker contract.
//
// One persistent connection, one outstanding request at a time: every call sends a frame and waits for
// its reply before returning (#348 scope 4).  The client enforces the same frame caps as the worker and
// the same deadlines, and surfaces the worker's typed status both as a bool and as the last status code,
// so the engine integration (#349) can fail closed on any refusal.
#pragma once

#include "strata/worker/expert_wire.hpp"
#include "strata/worker/local_source.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::worker {

class ExpertWorkerClient {
public:
    ExpertWorkerClient() = default;
    ~ExpertWorkerClient();
    ExpertWorkerClient(const ExpertWorkerClient&) = delete;
    ExpertWorkerClient& operator=(const ExpertWorkerClient&) = delete;

    /// Connects to host:port (normally 127.0.0.1 and the worker's port, or an SSH forward's local end).
    bool connect_host(const std::string& host, uint16_t port, int deadline_ms, std::string& err);

    bool connect_local(const std::string &numeric_host, uint16_t port, int deadline_ms,
                       const std::function<bool()> &cancelled, std::string &err);
    /// SETUP_BEGIN: starts a session (voiding any previous one the worker held).  On success the session
    /// id and the worker's chunk cap are remembered for the upload/infer calls.
    bool setup(const SetupManifest& m, std::string& err);

    /// resident negotiates complete verified reuse; unsupported peers fail without upload fallback.
    bool setup_local(const SetupManifest &m, Digest128 manifest, bool fallback, std::string &err, bool resident = false);
    // Hits are in request order. A zero means ENOENT and permits upload only if negotiated.
    bool load_local(const std::vector<std::pair<uint16_t, uint16_t>> &slots, std::vector<uint8_t> &hits,
                    std::string &err);
    struct LocalStats {
        uint64_t bytes = 0, hits = 0, misses = 0, read_ns = 0, verify_ns = 0, setup_ns = 0;
    };
    const LocalStats &local_stats() const { return local_stats_; }
    void set_setup_cancelled(std::function<bool()> fn) { setup_cancelled_ = std::move(fn); }
    void set_deadline_ms(int ms) { deadline_ms_ = ms; }
    /// Uploads one whole expert blob as bounded in-order chunks.  `blob` is `blob_bytes` long
    /// (the SETUP_ACK reports it); the digest is computed here.
    bool upload(int layer, int expert, const uint8_t *blob, uint32_t blob_bytes, std::string &err,
                const Digest128 *expected = nullptr);

    /// READY: freezes the session once every owned slot is present and verified.
    bool mark_ready(std::string& err);

    /// One bounded inference request; on refusal returns false with `last_status` set.
    bool infer(const InferRequest& r, InferReply& reply, std::string& err);

    bool status(WorkerStatusInfo& out, std::string& err);
    bool cache_status(WorkerCacheInfo& out, std::string& err);
    bool unload(std::string& err);
    bool cache_hit() const { return cache_hit_; }
    uint64_t reused_bytes() const { return reused_bytes_; }

    /// Asks the worker to stop (MSG_SHUTDOWN); mainly a test/tool knob.
    bool shutdown(std::string& err);

    /// Closes the connection.  A disconnect voids the worker-side session: call setup() again, never
    /// infer(). Verified weights may survive only with opt-in bounded residency.
    void close();

    uint64_t session() const { return session_; }
    uint32_t blob_bytes() const { return blob_bytes_; }
    uint16_t last_status() const { return last_status_; }

private:
  bool setup_impl(const SetupManifest &m, const Digest128 *identity, bool fallback, std::string &err, bool resident = false);
  bool send_frame(uint16_t msg, uint16_t st, const std::vector<uint8_t> &payload, std::string &err);
  bool recv_frame(uint16_t want_msg, WireFrame &f, std::string &err);

  uintptr_t fd_ = ~(uintptr_t)0;
  int deadline_ms_ = 10000;
  uint64_t session_ = 0;
  uint32_t blob_bytes_ = 0;
  uint32_t chunk_cap_ = kUploadChunkMax;
  uint16_t n_embd_ = 0;
  std::function<bool()> setup_cancelled_;
  bool cache_hit_ = false;
  uint64_t reused_bytes_ = 0;
  bool local_ = false;
  LocalStats local_stats_;
  uint16_t last_status_ = ST_OK;
};

}  // namespace strata::worker
