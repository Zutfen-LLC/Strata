// tests/worker/expert_tier_test.cpp - inferswarm #349: the engine-side tier's focused checks.
//
// CPU-only, no GPU and no model file: Q8_0 packed fixtures are generated in-process with ggml's own
// quantizer (the #348 fixture), the REAL service runs on a real loopback socket, and the tier drives it
// through the REAL client - the same code path the CUDA engine's dispatch drives.  The transported
// partials are compared BIT FOR BIT with ExpertWorkerService::direct_native_partial() under the tier's
// own request construction (slicing, weights, repeated ids, consecutive windows, T=1/4/8, mixed, zero
// and all ownership), and the rest is the engine-side behaviour the issue asks for: bounded setup
// staging (one fetch per owned blob), no local blob dereference after readiness (a changed or
// unavailable source must not change the answer), zero-work windows sending no frame, and the failure
// latch (an out-of-partition marking, a worker that goes away, a deadline the worker never meets, an
// over-budget setup, an empty partition).
#include "strata/worker/expert_tier.hpp"
#include "strata/worker/expert_worker.hpp"
#include "strata/worker/expert_wire.hpp"

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}

using strata::worker::SetupManifest;
using strata::worker::WorkerOptions;
using strata::kernels::cpu::NativeFmt;

constexpr uint16_t kGu = 8, kD = 8;   // GGML_TYPE_Q8_0
constexpr uint16_t kLayers = 3, kExperts = 16, kEmbd = 2560, kFf = 640;

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 1u) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return (float) ((int32_t) (s >> 9)) / (float) (1 << 23) - 1.f;
    }
    uint32_t pick(uint32_t n) { s = s * 1664525u + 1013904223u; return s % n; }
    float unit() { return 0.5f * next() + 0.5f; }   // (0, 1): finite positive routing weights
};

/// The router's real shape: K DISTINCT experts per token (a repeated id inside one token's top-k is a
/// worker refusal, covered by #348's own controls; the same expert across tokens is normal and kept).
std::vector<int32_t> distinct_ids(Rng& r, int T, int K) {
    std::vector<int32_t> ids((size_t) T * K);
    std::vector<int32_t> pool((size_t) kExperts);
    for (int t = 0; t < T; ++t) {
        for (int32_t i = 0; i < kExperts; ++i) pool[(size_t) i] = i;
        for (int j = 0; j < K; ++j) {   // partial Fisher-Yates: the router's distinct top-k
            const uint32_t p = r.pick((uint32_t) (kExperts - j)) + (uint32_t) j;
            std::swap(pool[(size_t) j], pool[(size_t) p]);
            ids[(size_t) (t * K + j)] = pool[(size_t) j];
        }
    }
    return ids;
}

std::vector<uint8_t> make_blob(uint32_t seed, const NativeFmt& f) {
    std::vector<uint8_t> blob(f.bytes, 0);
    std::vector<float> src((size_t) std::max(f.n_embd, f.n_ff));
    Rng r(seed);
    auto pack = [&](int64_t n, uint8_t* dst) {
        for (int64_t i = 0; i < n; ++i) src[(size_t) i] = r.next();
        const size_t wrote = ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), dst, 0, 1, n, nullptr);
        check(wrote == (size_t) (n == f.n_embd ? f.gu_row : f.d_row), "ggml packed a whole Q8_0 row");
    };
    for (int64_t row = 0; row < f.n_ff; ++row) pack(f.n_embd, blob.data() + (size_t) row * f.gu_row);
    for (int64_t row = 0; row < f.n_ff; ++row)
        pack(f.n_embd, blob.data() + f.up_off + (size_t) row * f.gu_row);
    for (int64_t row = 0; row < f.n_embd; ++row)
        pack(f.n_ff, blob.data() + f.down_off + (size_t) row * f.d_row);
    return blob;
}

using BlobMap = std::map<std::pair<int, int>, std::vector<uint8_t>>;

const std::pair<int, int> kOwnedList[] = {{0, 2}, {0, 5}, {0, 9}, {1, 1}, {1, 2}, {1, 3}, {2, 7}};

std::vector<uint8_t> make_partition() {
    const size_t stride = ((size_t) kExperts + 7) / 8;
    std::vector<uint8_t> p((size_t) kLayers * stride, 0);
    for (const auto& [l, e] : kOwnedList) p[(size_t) l * stride + (size_t) e / 8] |= (uint8_t) (1u << (e % 8));
    return p;
}

BlobMap make_blobs(const NativeFmt& f) {
    BlobMap blobs;
    uint32_t seed = 7;
    for (const auto& [l, e] : kOwnedList) blobs[{l, e}] = make_blob(seed++, f);
    return blobs;
}

struct Svc {
    std::unique_ptr<strata::worker::ExpertWorkerService> svc;
    uint16_t port = 0;
    std::thread th;
    explicit Svc(WorkerOptions o = {}) {
        if (o.workers == 0) o.workers = 2; // synthetic checks must not consume a host-wide pool
        svc = std::make_unique<strata::worker::ExpertWorkerService>(o);
    }
    bool start() {
        std::string err;
        if (!svc->listen(port, err)) { std::fprintf(stderr, "FAIL: listen: %s\n", err.c_str()); ++fails; return false; }
        th = std::thread([this] { svc->run(); });
        return true;
    }
    void stop() {
        if (th.joinable()) {
            svc->stop();
            th.join();
        }
    }
    ~Svc() { stop(); }
};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/// The tier's Open with the fixture fetch: counts calls, can be switched off after setup.
struct Fetcher {
    BlobMap* blobs = nullptr;
    std::atomic<int> calls{0};
    std::atomic<bool> allow{true};
    strata::worker::ExpertWorkerTier::Open open(uint16_t port) {
        strata::worker::ExpertWorkerTier::Open o;
        o.host = "127.0.0.1";
        o.port = port;
        o.model_name = "tier-fixture";
        o.gu_type = kGu;
        o.d_type = kD;
        o.n_layers = kLayers;
        o.n_expert = kExperts;
        o.n_embd = kEmbd;
        o.n_ff = kFf;
        o.partition = make_partition();
        o.fetch = [this](int64_t l, int64_t e, uint8_t* dst, uint32_t bytes) {
            if (!allow.load()) return false;
            const auto it = blobs->find({(int) l, (int) e});
            if (it == blobs->end() || it->second.size() != (size_t) bytes) return false;
            std::memcpy(dst, it->second.data(), (size_t) bytes);
            ++calls;
            return true;
        };
        return o;
    }
};

/// One window through the tier; `owned[t*k+j]` says the worker executes the entry.  A zero-work window
/// yields zeros - the worker contributes nothing and the engine must read it that way.
bool run_window(strata::worker::ExpertWorkerTier& tier, int layer, const std::vector<int32_t>& ids,
                const std::vector<float>& weights, const std::vector<float>& x, const std::vector<uint8_t>& owned,
                int T, int K, std::vector<float>& out, std::string& err) {
    if (!tier.post(layer, x.data(), ids.data(), weights.data(), owned.data(), T, K, err)) return false;
    const bool sent = tier.posted();
    if (!tier.finish(err)) return false;
    if (!sent) {
        out.assign((size_t) T * kEmbd, 0.f);
        return true;
    }
    out.assign(tier.partials(), tier.partials() + (size_t) T * kEmbd);
    return true;
}

/// The direct reference for the same window (bitwise what the worker must answer).
bool direct_window(const NativeFmt& fmt, const BlobMap& blobs, int layer, const std::vector<int32_t>& ids,
                   const std::vector<float>& weights, const std::vector<float>& x,
                   const std::vector<uint8_t>& owned, int T, int K, std::vector<float>& out) {
    std::vector<const uint8_t*> entry_blobs((size_t) T * K, nullptr);
    for (int i = 0; i < T * K; ++i)
        if (owned[(size_t) i]) entry_blobs[(size_t) i] = blobs.at({layer, ids[(size_t) i]}).data();
    std::string err;
    if (!strata::worker::ExpertWorkerService::direct_native_partial(fmt, T, K, owned.data(), weights.data(),
                                                                    x.data(), entry_blobs.data(), out, err)) {
        std::fprintf(stderr, "FAIL: direct_native_partial: %s\n", err.c_str());
        ++fails;
        return false;
    }
    return true;
}

void owned_of(const std::vector<uint8_t>& partition, int layer, const std::vector<int32_t>& ids,
              std::vector<uint8_t>& owned) {
    const size_t stride = ((size_t) kExperts + 7) / 8;
    owned.assign(ids.size(), 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t e = ids[i];
        owned[i] = e >= 0 && e < kExperts &&
                           (partition[(size_t) layer * stride + (size_t) e / 8] & (1u << (e % 8)))
                       ? 1
                       : 0;
    }
}

// ---- the parity battery -------------------------------------------------------------------------

void parity_battery(const NativeFmt& fmt, BlobMap& blobs, uint16_t port) {
    std::string err;
    strata::worker::ExpertWorkerTier tier;
    Fetcher fetch{&blobs};
    const std::vector<uint8_t> partition = make_partition();
    auto tier_open = fetch.open(port);
    tier_open.rpc_timing = true;
    if (!tier.open(tier_open, err)) {
        std::fprintf(stderr, "FAIL: tier open against the real service: %s\n", err.c_str());
        ++fails;
        return;
    }
    check(tier.ready() && !tier.dead(), "tier ready after setup");
    check(fetch.calls == (int) blobs.size(), "one fetch per owned blob (bounded staging: no re-reads)");
    check(tier.stats().setup_blobs == blobs.size(), "setup blob count recorded");
    check(tier.stats().setup_weight_bytes == blobs.size() * fmt.bytes, "setup weight bytes recorded");

    // Zero work does not create the lazy communication thread.  The ordinary close at the end covers
    // cleanup after the persistent thread exists; failure_battery closes a freshly opened idle tier.
    {
        const int T = 1, K = 1;
        std::vector<int32_t> ids{4};
        std::vector<float> weights{0.5f}, x(kEmbd, 0.1f);
        std::vector<uint8_t> owned{0};
        check(tier.post(0, x.data(), ids.data(), weights.data(), owned.data(), T, K, err),
              "the first zero-work post succeeds");
        check(tier.finish(err), "the first zero-work finish succeeds");
        check(tier.stats().communication_thread_starts == 0,
              "zero work creates no communication thread");
        check(tier.rpc_timing().calls() == 0,
              "zero work creates no RPC timing call");
    }

    Rng r(99);
    for (int T : {1, 4, 8}) {
        const int K = 10;
        const std::vector<int32_t> ids = distinct_ids(r, T, K);
        std::vector<float> weights((size_t) T * K), x((size_t) T * kEmbd);
        for (int i = 0; i < T * K; ++i) weights[(size_t) i] = r.unit();
        for (auto& v : x) v = r.next();
        for (int layer = 0; layer < kLayers; ++layer) {   // consecutive windows, one per layer
            std::vector<uint8_t> owned;
            owned_of(partition, layer, ids, owned);
            std::vector<float> got, want;
            check(run_window(tier, layer, ids, weights, x, owned, T, K, got, err),
                  "a mixed-ownership window ran");
            check(direct_window(fmt, blobs, layer, ids, weights, x, owned, T, K, want),
                  "the direct reference ran");
            check(same_bits(want, got), "transported partials are bitwise the direct call");
        }
    }
    // a zero-ownership window (expert 4 of layer 0 is not owned): no frame, zero contribution
    {
        const int T = 4, K = 10;
        std::vector<int32_t> ids((size_t) T * K, 4);
        std::vector<float> weights((size_t) T * K, 0.1f), x((size_t) T * kEmbd, 0.05f);
        std::vector<uint8_t> owned((size_t) T * K, 0);
        std::vector<float> out;
        strata::worker::WorkerStatusInfo st;
        check(tier.status(st, err), "status before the zero-work window");
        const uint64_t before = st.requests;
        check(run_window(tier, 0, ids, weights, x, owned, T, K, out, err), "a zero-ownership window ran");
        check(!tier.posted(), "the zero-work window posted no request");
        check(std::all_of(out.begin(), out.end(), [](float v) { return v == 0.f; }),
              "the zero-work contribution reads as zero");
        check(tier.status(st, err), "status after the zero-work window");
        check(st.requests == before, "no INFER frame was sent for zero CPU work");
    }
    // an all-ownership window (layer 1 routes only owned experts, ids repeated across tokens)
    {
        const int T = 2, K = 3;
        std::vector<int32_t> ids{1, 2, 3, 1, 2, 3};
        std::vector<float> weights((size_t) T * K, 0.2f), x((size_t) T * kEmbd);
        Rng rr(5);
        for (auto& v : x) v = rr.next();
        std::vector<uint8_t> owned;
        owned_of(partition, 1, ids, owned);
        std::vector<float> got, want;
        check(run_window(tier, 1, ids, weights, x, owned, T, K, got, err), "an all-ownership window ran");
        check(direct_window(fmt, blobs, 1, ids, weights, x, owned, T, K, want), "the direct reference ran");
        check(same_bits(want, got), "all-ownership partials are bitwise the direct call");
    }
    // a wrong-token control: shuffled activations must give the direct answer of the SHUFFLED input
    // (a transposed or mis-sliced activation row would not)
    {
        const int T = 4, K = 10;
        Rng rr(11);
        const std::vector<int32_t> ids = distinct_ids(rr, T, K);
        std::vector<float> weights((size_t) T * K), x((size_t) T * kEmbd), xs = x;
        for (int i = 0; i < T * K; ++i) weights[(size_t) i] = rr.unit();
        for (auto& v : x) v = rr.next();
        xs = x;
        for (int t = 0; t < T / 2; ++t)   // swap two token rows
            for (int e = 0; e < kEmbd; ++e)
                std::swap(xs[(size_t) t * kEmbd + e], xs[(size_t) (T - 1 - t) * kEmbd + e]);
        std::vector<uint8_t> owned;
        owned_of(partition, 0, ids, owned);
        std::vector<float> got1, got2, want1, want2;
        check(run_window(tier, 0, ids, weights, x, owned, T, K, got1, err), "the window ran");
        check(run_window(tier, 0, ids, weights, xs, owned, T, K, got2, err), "the shuffled window ran");
        direct_window(fmt, blobs, 0, ids, weights, x, owned, T, K, want1);
        direct_window(fmt, blobs, 0, ids, weights, xs, owned, T, K, want2);
        check(same_bits(want1, got1), "the first activations match the direct call");
        check(same_bits(want2, got2), "the shuffled activations match their own direct call");
        check(!same_bits(got1, got2), "changed activations change the answer (the wrong-token control bites)");
    }
    // no local dereference after readiness: the source changes (or goes away) under us, the answers
    // must not - the worker holds the only bytes inference reads
    {
        const int T = 4, K = 10;
        Rng rr(21);
        const std::vector<int32_t> ids = distinct_ids(rr, T, K);
        std::vector<float> weights((size_t) T * K), x((size_t) T * kEmbd);
        for (int i = 0; i < T * K; ++i) weights[(size_t) i] = rr.unit();
        for (auto& v : x) v = rr.next();
        std::vector<uint8_t> owned;
        owned_of(partition, 0, ids, owned);
        std::vector<float> got, want;
        direct_window(fmt, blobs, 0, ids, weights, x, owned, T, K, want);
        const int calls_before = fetch.calls.load();
        BlobMap changed = blobs;
        for (auto& [k, v] : changed) v[0] ^= 0xff;
        const BlobMap original = blobs;
        *fetch.blobs = changed;   // the local source now holds different bytes
        check(run_window(tier, 0, ids, weights, x, owned, T, K, got, err), "the window ran");
        check(fetch.calls.load() == calls_before, "inference dereferenced no local blob");
        check(same_bits(want, got), "a changed local source cannot change the answer");
        fetch.allow.store(false);   // and an unavailable source is never asked
        check(run_window(tier, 0, ids, weights, x, owned, T, K, got, err), "the window ran with the source off");
        check(same_bits(want, got), "an unavailable source cannot change the answer");
        fetch.allow.store(true);
        *fetch.blobs = original;
    }
    // post() owns copies of every caller buffer.  Mutating them as soon as post returns cannot alter
    // the exchange that is already queued on the persistent thread.
    {
        const int T = 2, K = 3;
        std::vector<int32_t> ids{1, 2, 3, 1, 2, 3};
        std::vector<float> weights((size_t) T * K, 0.2f), x((size_t) T * kEmbd, 0.05f);
        std::vector<uint8_t> owned((size_t) T * K, 1);
        const auto original_ids = ids;
        const auto original_weights = weights;
        const auto original_x = x;
        const auto original_owned = owned;
        check(tier.post(1, x.data(), ids.data(), weights.data(), owned.data(), T, K, err),
              "the copied-lifetime window posted");
        std::fill(ids.begin(), ids.end(), 4);
        std::fill(weights.begin(), weights.end(), -9.f);
        std::fill(x.begin(), x.end(), 99.f);
        std::fill(owned.begin(), owned.end(), 0);
        check(tier.finish(err), "the copied-lifetime window finished");
        std::vector<float> got(tier.partials(), tier.partials() + (size_t) T * kEmbd), want;
        check(direct_window(fmt, blobs, 1, original_ids, original_weights, original_x, original_owned,
                            T, K, want), "the copied-lifetime reference ran");
        check(same_bits(want, got), "the queued request owns its caller inputs");
    }
    // A second post without a preceding finish waits for the first exchange and reuses the one slot.
    // The legacy contract discarded the first reply in this usage; finish returns the second reply.
    {
        const int T = 2, K = 3;
        std::vector<int32_t> ids{1, 2, 3, 1, 2, 3};
        std::vector<float> wa((size_t) T * K, 0.1f), wb((size_t) T * K, 0.3f);
        std::vector<float> xa((size_t) T * kEmbd, 0.02f), xb((size_t) T * kEmbd, 0.07f);
        std::vector<uint8_t> owned((size_t) T * K, 1);
        const uint64_t before = tier.stats().requests;
        check(tier.post(1, xa.data(), ids.data(), wa.data(), owned.data(), T, K, err),
              "the un-finished first request posted");
        check(tier.post(1, xb.data(), ids.data(), wb.data(), owned.data(), T, K, err),
              "the second post ordered behind the first");
        check(tier.finish(err), "the ordered second request finished");
        std::vector<float> got(tier.partials(), tier.partials() + (size_t) T * kEmbd), want;
        check(direct_window(fmt, blobs, 1, ids, wb, xb, owned, T, K, want),
              "the ordered second reference ran");
        check(same_bits(want, got), "finish exposes the ordered second reply");
        check(tier.stats().requests == before + 2, "both ordered requests used the one slot");
    }
    check(tier.stats().communication_thread_starts == 1,
          "repeated requests use one persistent communication thread");
    check(tier.stats().communication_thread_id != 0,
          "the communication OS thread id is exposed for affinity inspection");
    check(tier.stats().queue_ms >= 0 && tier.stats().exchange_ms > 0 && tier.stats().wait_ms >= 0,
          "aggregate queue, exchange and finish-wait timings are recorded");
    check(tier.rpc_timing().calls() == tier.stats().requests && tier.rpc_timing().failures() == 0,
          "opt-in client timing counts each successful production RPC once");
    check(tier.rpc_timing().phase(strata::worker::RpcTimingPhase::ClientPrepare).count ==
              tier.rpc_timing().calls() &&
              tier.rpc_timing().phase(strata::worker::RpcTimingPhase::ClientReceiveDecode).count ==
              tier.rpc_timing().calls(),
          "client preparation and receive/decode distributions cover successful RPCs");
    tier.close();
}

// ---- the failure battery ------------------------------------------------------------------------

/// A fake worker that answers setup with tiny 64-byte blobs, then injects one inference fault through
/// the real client.  None of these failures may expose a partial for the engine to merge.
struct AdverseWorker {
    enum Mode { refusal, malformed_reply, disconnect, timeout, stale_reply };
    Mode mode;
    int lfd = -1;
    uint16_t port = 0;
    std::thread th;
    explicit AdverseWorker(Mode m) : mode(m) {}
    void start() {
        lfd = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        if (::bind(lfd, (sockaddr*) &addr, sizeof addr) != 0 || ::listen(lfd, 1) != 0) return;
        socklen_t alen = sizeof addr;
        ::getsockname(lfd, (sockaddr*) &addr, &alen);
        port = ntohs(addr.sin_port);
        th = std::thread([this] { serve(); });
    }
    void serve() {
        using namespace strata::worker;
        int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd < 0) return;
        auto put64 = [](std::vector<uint8_t>& o, uint64_t v) {
            for (int b = 0; b < 8; ++b) o.push_back((uint8_t) (v >> (8 * b)));
        };
        auto put32 = [](std::vector<uint8_t>& o, uint32_t v) {
            for (int b = 0; b < 4; ++b) o.push_back((uint8_t) (v >> (8 * b)));
        };
        for (;;) {
            uint8_t head[16];
            if (::recv(cfd, head, 16, MSG_WAITALL) <= 0) break;
            const uint16_t msg = (uint16_t) (head[4] | head[5] << 8);
            uint64_t len = 0;
            for (int b = 0; b < 8; ++b) len |= (uint64_t) head[8 + b] << (8 * b);
            if (len > kMaxFrameBytes) break;
            std::vector<uint8_t> p((size_t) len);
            if (len && ::recv(cfd, p.data(), (size_t) len, MSG_WAITALL) <= 0) break;
            std::vector<uint8_t> ack;
            uint16_t reply = 0;
            if (msg == MSG_SETUP_BEGIN) {
                put64(ack, 0x88);
                put64(ack, 64);    // tiny blobs: one chunk per slot, no digests checked here
                put32(ack, 65536);
                reply = MSG_SETUP_ACK;
            } else if (msg == MSG_UPLOAD) {
                put64(ack, 0x88);
                put32(ack, 64);
                reply = MSG_UPLOAD_ACK;
            } else if (msg == MSG_READY) {
                put64(ack, 0x88);
                put64(ack, 0);
                reply = MSG_READY_ACK;
            } else if (msg == MSG_INFER) {
                if (mode == timeout) {
                    std::this_thread::sleep_for(std::chrono::seconds(4)); // client deadline must fire first
                    break;
                }
                if (mode == disconnect) break;
                reply = MSG_INFER_ACK;
                if (mode == refusal) {
                    std::vector<uint8_t> f;
                    encode_frame(f, reply, ST_STALE_SESSION, nullptr, 0);
                    ::send(cfd, f.data(), f.size(), 0);
                    break;
                }
                InferRequest request;
                if (!decode_infer(p.data(), p.size(), kLayers, kExperts, kEmbd, request)) break;
                InferReply response;
                response.session = mode == stale_reply ? 0x99 : request.session;
                response.sequence = request.sequence;
                response.partials.assign((size_t) request.T * kEmbd, 0.25f);
                ack = encode_infer_ack(response);
                if (mode == malformed_reply) ack.pop_back();
            } else {
                break;
            }
            std::vector<uint8_t> f;
            f.push_back('S'); f.push_back('E'); f.push_back('W'); f.push_back('1');
            f.push_back((uint8_t) reply); f.push_back((uint8_t) (reply >> 8));
            f.push_back(0); f.push_back(0);
            for (int b = 0; b < 8; ++b) f.push_back((uint8_t) (ack.size() >> (8 * b)));
            f.insert(f.end(), ack.begin(), ack.end());
            if (::send(cfd, f.data(), f.size(), 0) <= 0) break;
        }
        ::close(cfd);
    }
    ~AdverseWorker() {
        if (th.joinable()) th.join();
        if (lfd >= 0) ::close(lfd);
    }
};

void failure_battery(const NativeFmt& fmt, BlobMap& blobs) {
    std::string err;
    // an out-of-partition marking is caught BEFORE any frame (the engine's own bug, not the worker's)
    {
        Svc svc;
        check(svc.start(), "service for the marking control");
        strata::worker::ExpertWorkerTier tier;
        Fetcher fetch{&blobs};
        check(tier.open(fetch.open(svc.port), err), "tier open for the marking control");
        const int T = 1, K = 10;
        std::vector<int32_t> ids(K, 4);   // layer 0 expert 4: not owned
        std::vector<float> w(K, 0.1f), x((size_t) kEmbd, 0.05f);
        std::vector<uint8_t> owned(K, 1);   // wrongly marked
        check(!tier.post(0, x.data(), ids.data(), w.data(), owned.data(), T, K, err),
              "an out-of-partition marking is refused");
        check(tier.dead(), "the tier latched dead");
        check(!tier.post(0, x.data(), ids.data(), w.data(), owned.data(), T, K, err), "a later post fails fast");
        check(!tier.finish(err), "finish reports the latch");
        check(svc.svc->counters().requests == 0, "no INFER frame reached the worker");
    }
    // a worker that goes away mid-run: the tier dies, and a fresh tier on a fresh service is a NEW
    // session (the restart contract; the old session ids the old worker held are gone with it)
    {
        strata::worker::ExpertWorkerTier tier;
        Fetcher fetch{&blobs};
        const int T = 4, K = 10;
        Rng rr(31);
        const std::vector<int32_t> ids = distinct_ids(rr, T, K);
        std::vector<float> w((size_t) T * K, 0.25f), x((size_t) T * kEmbd, 0.1f);
        std::vector<uint8_t> owned;   // the entries of layer 0's owned experts
        owned_of(make_partition(), 0, ids, owned);
        std::vector<float> out;
        {
            Svc svc;
            check(svc.start(), "service for the restart control");
            check(tier.open(fetch.open(svc.port), err), "tier open for the restart control");
            check(run_window(tier, 0, ids, w, x, owned, T, K, out, err), "a window ran before the restart");
        }   // the service object is gone: its listener and connection are closed
        bool failed = false;
        for (int attempt = 0; attempt < 3 && !failed; ++attempt)
            failed = !run_window(tier, 0, ids, w, x, owned, T, K, out, err);
        check(failed, "a window after the worker left fails");
        check(tier.dead(), "the dropped worker latched the tier dead");
        check(!tier.served(), "a disconnect exposes no partial to merge");
        Svc fresh;
        check(fresh.start(), "the fresh service started");
        strata::worker::ExpertWorkerTier tier2;
        Fetcher fetch2{&blobs};
        check(tier2.open(fetch2.open(fresh.port), err), "a fresh tier opens against the fresh service");
        tier2.close();
    }
    // Typed refusal, malformed success, disconnect, timeout and stale successful identity all latch
    // without exposing a partial.  These run through the persistent communication thread.
    for (const auto mode : {AdverseWorker::refusal, AdverseWorker::malformed_reply,
                            AdverseWorker::disconnect, AdverseWorker::timeout,
                            AdverseWorker::stale_reply}) {
        AdverseWorker fake(mode);
        fake.start();
        check(fake.port != 0, "the adverse worker started");
        strata::worker::ExpertWorkerTier tier;
        auto o = Fetcher{&blobs}.open(0);
        o.port = fake.port;
        o.deadline_ms = 400;
        o.rpc_timing = true;
        // the fake advertises 64-byte blobs, so the fixture fetcher refuses them - feed it a fetch that
        // accepts the fake's size, just for the handshake
        o.fetch = [](int64_t, int64_t, uint8_t* dst, uint32_t bytes) {
            std::memset(dst, 0xAB, (size_t) bytes);
            return true;
        };
        check(tier.open(o, err), "tier open against the adverse worker");
        const int T = 2, K = 10;
        Rng rr(41);
        const std::vector<int32_t> ids = distinct_ids(rr, T, K);
        std::vector<float> w((size_t) T * K, 0.5f), x((size_t) T * kEmbd, 0.2f);
        std::vector<uint8_t> owned;   // layer 0's owned share: the request must carry real entries
        owned_of(make_partition(), 0, ids, owned);
        const auto t0 = std::chrono::steady_clock::now();
        check(tier.post(0, x.data(), ids.data(), w.data(), owned.data(), T, K, err), "the request was posted");
        bool failed = false;
        if (mode == AdverseWorker::timeout) {
            tier.close(); // in-flight destruction/close wakes and joins; the socket deadline bounds it
            failed = tier.dead();
        } else {
            failed = !tier.finish(err);
        }
        check(failed, "the adverse request failed");
        const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        check(tier.dead(), "the adverse response latched the tier dead");
        check(!tier.served(), "an adverse response exposes no partial to merge");
        check(!tier.post(0, x.data(), ids.data(), w.data(), owned.data(), T, K, err),
              "the dead tier sends no retry");
        check(tier.rpc_timing().calls() == 1 && tier.rpc_timing().failures() == 1,
              "the real client timing counts the adverse RPC once as failed");
        check(tier.rpc_timing().phase(strata::worker::RpcTimingPhase::ClientEncodeSend).count == 1 &&
                  tier.rpc_timing().phase(strata::worker::RpcTimingPhase::ClientReceiveDecode).count == 1,
              "the real refusal/disconnect/timeout path retains bounded exchange phases");
        if (mode == AdverseWorker::timeout)
            check(took < 3.0, "the deadline fired instead of hanging");
    }
    // setup refusals: over-budget is typed and latches; an empty partition is a config error, not a latch
    {
        Svc svc(WorkerOptions{.memory_budget = 1ull << 20});
        check(svc.start(), "service for the budget control");
        strata::worker::ExpertWorkerTier tier;
        Fetcher fetch{&blobs};
        check(!tier.open(fetch.open(svc.port), err), "an over-budget setup is refused");
        check(tier.dead(), "the refused setup latched the tier");
        check(err.find("refused setup") != std::string::npos, "the refusal says it was the setup");
        strata::worker::ExpertWorkerTier empty_tier;
        auto o2 = fetch.open(svc.port);
        o2.partition.assign(((size_t) kExperts + 7) / 8 * kLayers, 0);
        check(!empty_tier.open(o2, err), "an empty partition is refused");
        check(!empty_tier.dead(), "the empty partition is a config error, not a worker failure");
    }
}

}  // namespace

int main() {
    NativeFmt fmt;
    std::string err;
    check(strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, fmt, err), "Q8_0/Q8_0 is a native layout");
    if (fails) return 1;
    BlobMap blobs = make_blobs(fmt);
    {
        WorkerOptions timing_options;
        timing_options.rpc_timing = true;
        Svc svc(timing_options);
        check(svc.start(), "the service started");
        parity_battery(fmt, blobs, svc.port);
        svc.stop(); // join before reading service-owned timing state
        check(svc.svc->rpc_timing().calls() > 0 && svc.svc->rpc_timing().failures() == 0,
              "opt-in worker timing counts successful production RPCs");
        check(svc.svc->rpc_timing().phase(strata::worker::RpcTimingPhase::WorkerPool).count > 0 &&
                  svc.svc->rpc_timing().phase(strata::worker::RpcTimingPhase::WorkerRequest).count ==
                  svc.svc->rpc_timing().calls(),
              "worker pool and complete-handler distributions cover real requests");
    }
    failure_battery(fmt, blobs);
    if (fails) {
        std::fprintf(stderr, "expert_tier_test: %d failure(s)\n", fails);
        return 1;
    }
    std::printf("expert_tier_test: all checks passed\n");
    return 0;
}
