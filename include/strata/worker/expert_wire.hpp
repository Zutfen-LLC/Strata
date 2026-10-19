// include/strata/worker/expert_wire.hpp - inferswarm #348: the resident CPU-expert worker's wire contract.
//
// A small loopback service that holds the CPU-owned expert blobs of one model in an arena and computes
// their weighted partial sums with Strata's native CPU kernels (native_expert.hpp), so a GPU-only engine
// process can keep its experts on another machine's RAM.  Everything on the wire is a typed frame with a
// bounded length; an inference frame never carries weight bytes or process pointers - only ids, routing
// weights and finite F32 activations.
//
// THE DETERMINISM CONTRACT (what "the same bits" means here):
//   * distinct experts are GROUPED in first-appearance order of their entries; the pool may compute
//     their rows concurrently, but a row's arithmetic does not depend on which job or thread ran it
//     (for the supported layouts every row is ggml-cpu's own per-token vec_dot);
//   * a token's partial sum accumulates its entries' weighted rows in ENTRY order, `acc = acc + w * o`,
//     one pass, the shared accumulate_partial();
//   * the transported result is bitwise `worker::direct_native_partial()` (below): quantize each token's
//     activation once with native_quant_act, evaluate every entry with native_gu_rows / native_quant_h /
//     native_down_rows over the expert's full row range, then accumulate in the same order.  For the
//     supported layouts both paths run ggml-cpu's per-token vec_dot, whose bits do not depend on how many
//     tokens share a row call.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::worker {

/// 'S','E','W','1' - the Strata Expert Worker protocol, version 1.
inline constexpr uint32_t kWireMagic = 0x31574553u;   // "SEW1" little-endian
inline constexpr uint16_t kWireVersion = 1;

/// Every frame: magic (4) | msg (2) | status (2) | payload length (8).
inline constexpr size_t kHeaderBytes = 16;

enum MsgType : uint16_t {
    MSG_SETUP_BEGIN = 1,   ///< c->s: identity, geometry, layout, the ownership partition; starts a session
    MSG_SETUP_ACK = 2,     ///< s->c: the session id, the blob size, the chunk cap
    MSG_UPLOAD = 3,        ///< c->s: one bounded chunk of one expert blob, in order
    MSG_UPLOAD_ACK = 4,    ///< s->c: bytes of that slot now present
    MSG_READY = 5,         ///< c->s: verify the whole assigned set, freeze the session
    MSG_READY_ACK = 6,     ///< s->c: ready, or the number of slots still missing
    MSG_INFER = 7,         ///< c->s: one bounded inference request (no weight bytes)
    MSG_INFER_ACK = 8,     ///< s->c: one F32 weighted partial sum per token
    MSG_STATUS = 9,        ///< c->c->s: the counters
    MSG_STATUS_ACK = 10,
    MSG_SHUTDOWN = 11,     ///< c->s: polite stop (a test/tool knob; the engine just closes)
};

enum Status : uint16_t {
    ST_OK = 0,
    ST_PROTOCOL = 1,        ///< bad magic, version or frame length
    ST_WRONG_STATE = 2,     ///< the message does not fit the session's state
    ST_BAD_REQUEST = 3,     ///< a field is out of range or inconsistent
    ST_UNSUPPORTED_LAYOUT = 4,   ///< the quant/layout pair is not one this worker runs
    ST_OVER_BUDGET = 5,     ///< the assigned set needs more arena than the budget allows
    ST_DUPLICATE = 6,       ///< a slot was already fully uploaded
    ST_NOT_OWNED = 7,       ///< the (layer, expert) slot is outside this worker's partition
    ST_DIGEST_MISMATCH = 8, ///< the uploaded bytes do not match the declared digest
    ST_NOT_READY = 9,       ///< inference before the session froze as ready
    ST_INCOMPLETE = 10,     ///< readiness asked with slots still missing
    ST_STALE_SESSION = 11,  ///< the session (or model/partition identity) does not match
    ST_NONFINITE = 12,      ///< an activation or routing weight is not finite
    ST_DEADLINE = 13,       ///< reserved: a frame that misses its deadline closes the connection (the
                            ///< peer learns from the EOF; no reply frame is owed for an unreadable one)
    ST_INTERNAL = 14,
};

// ---- compiled bounds --------------------------------------------------------
// The worker's own caps, from the geometry the CPU pool is compiled for (expert.hpp: H, FF, MAXT) plus the
// protocol's partition tables.  A manifest outside them is refused, never truncated to fit.
inline constexpr int kMaxT = strata::kernels::cpu::MAXT;   ///< tokens per request (8)
inline constexpr int kMaxK = 10;                           ///< routed experts per token
inline constexpr int kMaxNEmbd = strata::kernels::cpu::H;  ///< 2560
inline constexpr int kMaxNFf = strata::kernels::cpu::FF;   ///< 640 (the pool's split buffers)
inline constexpr int kMaxLayers = 96;
inline constexpr int kMaxExperts = 1024;
inline constexpr uint32_t kUploadChunkMax = 64u * 1024u;   ///< payload bytes per UPLOAD chunk
/// The UPLOAD frame's fixed part: session(8) layer(2) expert(2) total(4) off(4) len(4) digest(16).
inline constexpr size_t kUploadHeadBytes = 40;

/// Bytes of ownership bitmap a manifest may carry (also the SETUP_BEGIN frame cap's variable part).
inline constexpr size_t kPartitionBytes = (size_t) kMaxLayers * ((kMaxExperts + 7) / 8);
inline constexpr size_t kSetupFrameMax = kHeaderBytes + 128 + kPartitionBytes;

/// The largest inference payload the schema can express, derived, not asserted: fixed fields + per-entry
/// mask/ids/weights + per-token activations, at T=8, K=10, n_embd=2560.
inline constexpr size_t kInferFixed = 8 + 8 + 8 + 8 + 2 + 1 + 1;
inline constexpr size_t kInferPayloadMax = kInferFixed + (size_t) kMaxT * kMaxK +
                                           (size_t) kMaxT * kMaxK * 2 + (size_t) kMaxT * kMaxK * 4 +
                                           (size_t) kMaxT * kMaxNEmbd * 4;
inline constexpr size_t kInferFrameMax = kHeaderBytes + kInferPayloadMax;
static_assert(kInferFrameMax < 96u * 1024u, "an inference frame must stay below 96 KiB (#348)");

inline constexpr size_t kReplyPayloadMax = 16 + (size_t) kMaxT * kMaxNEmbd * 4;
inline constexpr size_t kUploadFrameMax = kHeaderBytes + kUploadHeadBytes + kUploadChunkMax;
inline constexpr size_t kMaxFrameBytes = kSetupFrameMax > kInferFrameMax
                                             ? (kSetupFrameMax > kUploadFrameMax ? kSetupFrameMax
                                                                                 : kUploadFrameMax)
                                             : (kInferFrameMax > kUploadFrameMax ? kInferFrameMax
                                                                                 : kUploadFrameMax);

/// One frame as it travels: the decoded header plus the payload bytes.
struct WireFrame {
    uint16_t msg = 0;
    uint16_t status = ST_OK;
    std::vector<uint8_t> payload;
};

/// A 128-bit content digest (two independent FNV-1a passes).  An integrity check against truncated or
/// mismatched uploads - NOT a cryptographic primitive, and not presented as one.
struct Digest128 {
    uint64_t a = 0, b = 0;
    bool operator==(const Digest128& o) const { return a == o.a && b == o.b; }
    bool operator!=(const Digest128& o) const { return !(*this == o); }
};
Digest128 blob_digest(const void* data, size_t n);
inline Digest128 blob_digest(const std::vector<uint8_t>& v) { return blob_digest(v.data(), v.size()); }

/// The setup manifest: model/geometry/quant/layout identity and the fixed CPU ownership partition.
struct SetupManifest {
    uint64_t nonce = 0;          ///< client-chosen; combined into the session id
    uint64_t model_id = 0;       ///< the client's identity digest of the model (carried, checked on INFER)
    char model_name[48] = {};    ///< NUL-padded, for the worker's log only
    char quant_name[16] = {};    ///< e.g. "Q8_0"
    uint16_t gu_type = 0, d_type = 0;
    uint16_t n_layers = 0, n_expert = 0;
    uint16_t n_embd = 0, n_ff = 0;
    uint64_t partition_id = 0;   ///< the client's digest of the partition (carried, checked on INFER)
    std::vector<uint8_t> partition;   ///< n_layers * ceil(n_expert/8) bitmap, layer-major
    bool owns(int layer, int expert) const {
        if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return false;
        const size_t stride = ((size_t) n_expert + 7) / 8;
        return (partition[(size_t) layer * stride + (size_t) expert / 8] >> (expert % 8)) & 1u;
    }
    uint64_t owned_count() const {
        uint64_t n = 0;
        for (int l = 0; l < n_layers; ++l)
            for (int e = 0; e < n_expert; ++e) n += owns(l, e) ? 1 : 0;
        return n;
    }
};

/// One inference request.  `mask`/`ids`/`weights` are T*K entries, entry-major (token t, rank j at t*K+j).
struct InferRequest {
    uint64_t session = 0, model_id = 0, partition_id = 0, sequence = 0;
    uint16_t layer = 0;
    uint8_t T = 0, K = 0;
    std::vector<uint8_t> mask;      ///< 1 = this worker computes the entry (it must own ids[i])
    std::vector<uint16_t> ids;
    std::vector<float> weights;
    std::vector<float> activations; ///< T * n_embd finite F32
};

struct InferReply {
    uint64_t session = 0, sequence = 0;
    std::vector<float> partials;    ///< T * n_embd; zeros where a token had no owned entry
};

struct WorkerStatusInfo {
    uint64_t session = 0;
    uint64_t setup_weight_bytes = 0, infer_in_bytes = 0, infer_out_bytes = 0;
    uint64_t requests = 0, failures = 0, cpu_service_ns = 0;
    uint16_t last_error = ST_OK;
    uint8_t ready = 0;
};

// ---- payload codecs (little-endian, bounds-checked) --------------------------
bool encode_setup_begin(const SetupManifest& m, std::vector<uint8_t>& out);
bool decode_setup_begin(const uint8_t* p, size_t n, SetupManifest& m);
/// `n_embd` (the manifest's) sizes the activation block; the wire frame carries no geometry of its own.
bool encode_infer(const InferRequest& r, uint16_t n_embd, std::vector<uint8_t>& out);
bool decode_infer(const uint8_t* p, size_t n, uint16_t n_layers, uint16_t n_expert, uint16_t n_embd,
                  InferRequest& r);
std::vector<uint8_t> encode_infer_ack(const InferReply& r);
bool decode_infer_ack(const uint8_t* p, size_t n, uint16_t n_embd, uint8_t T, InferReply& r);

/// Builds a whole frame - the 16-byte header plus the payload.  The ONE place the header is written,
/// shared by the worker, the client and the tests, so the three cannot drift apart.
void encode_frame(std::vector<uint8_t>& out, uint16_t msg, uint16_t status, const void* payload, size_t n);

/// The name of a status code, for logs and tests.
const char* status_name(uint16_t s);

}  // namespace strata::worker
