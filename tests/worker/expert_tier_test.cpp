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
#include "strata/worker/cpu_group.hpp"
#include "strata/worker/expert_worker.hpp"
#include "strata/worker/expert_wire.hpp"

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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
    if (!tier.open(fetch.open(port), err)) {
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
            } else if (msg == MSG_STATUS) {
                put64(ack,0x88);
                for(int i=0;i<6;++i)put64(ack,0);
                ack.push_back(0);ack.push_back(0);ack.push_back(1);
                reply=MSG_STATUS_ACK;
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

void group_battery(const NativeFmt& fmt, BlobMap& blobs) {
    using strata::worker::CpuEndpoint;
    using strata::worker::CpuWorkerGroup;
    std::string err;
    const std::vector<uint8_t> complement = make_partition();

    // Placement: one endpoint is exactly the legacy full partition; two/three endpoints are deterministic,
    // disjoint and exhaustive. Capacity is a hard bound and speed decides which worker receives hotter ranks.
    const std::vector<std::pair<int32_t,int32_t>> hot = {{0,2},{0,5},{0,9},{1,1},{1,2},{1,3},{2,7}};
    std::vector<std::vector<uint8_t>> one, two, three, again;
    check(strata::worker::place_cpu_partitions(complement,kLayers,kExperts,hot,{{"one",1,7,1}},one,err) &&
          one.size()==1 && one[0]==complement,"one CPU endpoint keeps the full complement");
    check(strata::worker::place_cpu_partitions(complement,kLayers,kExperts,hot,
          {{"a",1,4,1},{"b",2,3,2}},two,err),"two-worker capacity/speed placement");
    check(strata::worker::place_cpu_partitions(complement,kLayers,kExperts,hot,
          {{"a",1,3,1},{"b",2,2,2},{"c",3,2,1}},three,err) &&
          strata::worker::place_cpu_partitions(complement,kLayers,kExperts,hot,
          {{"a",1,3,1},{"b",2,2,2},{"c",3,2,1}},again,err) && three==again,
          "three-worker placement is deterministic");
    auto exact = [&](const std::vector<std::vector<uint8_t>>& parts) {
        std::vector<uint8_t> seen(complement.size(),0);
        for(const auto& p:parts)for(size_t i=0;i<p.size();++i){if(seen[i]&p[i])return false;seen[i]|=p[i];}
        return seen==complement;
    };
    check(exact(two)&&exact(three),"placements are disjoint and exhaustive");
    check(!strata::worker::place_cpu_partitions(complement,kLayers,kExperts,hot,
          {{"a",1,2,1},{"b",2,2,1}},again,err),"insufficient CPU capacity is refused");
    check(!strata::worker::place_cpu_partitions(complement,kLayers,kExperts,hot,
          {{"a",1,0,1},{"a",1,0,1}},again,err),"duplicate CPU endpoints are refused");
    std::vector<std::vector<uint8_t>> skewed;
    check(strata::worker::place_cpu_partitions(complement,kLayers,kExperts,hot,
          {{"small-a",1,1,1},{"small-b",2,1,1},{"large",3,1000,1}},skewed,err),
          "skewed capacity placement keeps every configured participant");
    auto bit_count=[](const std::vector<uint8_t>& p) {
        size_t n=0;for(uint8_t byte:p)for(;byte;byte&=(uint8_t)(byte-1))++n;return n;
    };
    check(skewed.size()==3&&bit_count(skewed[0])==1&&bit_count(skewed[1])==1&&bit_count(skewed[2])==5,
          "skewed capacity quotas reserve one expert per participant");

    Svc a, b, c;
    check(a.start()&&b.start()&&c.start(),"three CPU worker services started");
    std::vector<CpuEndpoint> endpoints={{"127.0.0.1",a.port,3,1},{"127.0.0.1",b.port,2,1},{"127.0.0.1",c.port,2,1}};
    Fetcher fetch{&blobs};

    // The compatibility path still runs through the group with one endpoint; two endpoints publish both
    // requests before either reply is awaited.
    const size_t stride=((size_t)kExperts+7)/8;
    std::vector<uint8_t> p0((size_t)kLayers*stride,0),p1=p0,sparse=p0;
    p0[0]|=(uint8_t)(1u<<2);p1[0]|=(uint8_t)(1u<<5);sparse[0]=p0[0];
    {
        auto o=fetch.open(a.port);o.partition=p0;
        CpuWorkerGroup single;
        check(single.open({o},sparse,err),"single-worker group opened");
        const int32_t id=2;const float weight=0.5f;const uint8_t owned=1;
        std::vector<float> x(kEmbd,0.1f);
        check(single.post(0,x.data(),&id,&weight,&owned,1,1,err)&&single.active(0),
              "single-worker compatibility request posted");
        check(single.finish(err)&&single.served(0),"single-worker compatibility request finished");
        single.close();
    }
    {
        sparse[0]=p0[0]|p1[0];
        auto o0=fetch.open(a.port);o0.partition=p0;
        auto o1=fetch.open(b.port);o1.partition=p1;
        CpuWorkerGroup pair;
        check(pair.open({o0,o1},sparse,err),"two-worker group opened");
        const int32_t ids[2]={2,5};const float weights[2]={0.4f,0.6f};const uint8_t owned[2]={1,1};
        std::vector<float> x(kEmbd,0.1f);
        check(pair.post(0,x.data(),ids,weights,owned,1,2,err)&&pair.active(0)&&pair.active(1)&&
              pair.stats(0).requests==1&&pair.stats(1).requests==1,
              "two-worker requests are both published before finish");
        check(pair.finish(err)&&pair.served(0)&&pair.served(1),"two-worker group finished");
        pair.close();
    }

    check(strata::worker::place_cpu_partitions(complement,kLayers,kExperts,hot,endpoints,three,err),
          "fixture three-way placement");
    std::vector<strata::worker::ExpertWorkerTier::Open> opens;
    for(size_t i=0;i<endpoints.size();++i){auto o=fetch.open(endpoints[i].port);o.partition=three[i];opens.push_back(o);}
    CpuWorkerGroup group;
    const bool group_open=group.open(opens,complement,err);
    check(group_open,"three-worker group opened");
    if(!group_open)return;
    const int T=1,K=3;
    const std::vector<int32_t> ids={2,5,9};
    const std::vector<float> weights={0.2f,0.3f,0.5f};
    std::vector<float> x(kEmbd);for(size_t i=0;i<x.size();++i)x[i]=(float)((int)(i%17)-8)*0.01f;
    const std::vector<uint8_t> all={1,1,1};
    check(group.post(0,x.data(),ids.data(),weights.data(),all.data(),T,K,err),"three-worker request posted");
    bool published_all=true;for(size_t i=0;i<group.size();++i)published_all &= group.active(i)&&group.stats(i).requests==1;
    check(published_all,"every participating CPU request is published before group finish waits");
    check(group.finish(err),"three-worker request finished");
    std::vector<float> merged(kEmbd,0), full;
    for(size_t i=0;i<group.size();++i){
        std::vector<uint8_t> mask(group.mask(i),group.mask(i)+K), reference;
        check(direct_window(fmt,blobs,0,ids,weights,x,mask,T,K,reference),"per-worker direct reference");
        check(std::memcmp(reference.data(),group.partials(i),reference.size()*sizeof(float))==0,
              "each worker partial is bitwise its direct reference");
        for(size_t j=0;j<merged.size();++j)merged[j]+=group.partials(i)[j]; // endpoint order is the contract
    }
    check(direct_window(fmt,blobs,0,ids,weights,x,all,T,K,full),"unsplit direct reference");
    bool close=true;for(size_t h=0;h<full.size();++h){
        float scale=std::fabs(full[h]);for(size_t i=0;i<group.size();++i)scale+=std::fabs(group.partials(i)[h]);
        const float rounding_bound=8.f*std::numeric_limits<float>::epsilon()*std::max(scale,1.f);
        close &= std::fabs(merged[h]-full[h])<=rounding_bound;
    }
    check(close,"ordered grouped reduction stays within its local F32 addition-rounding bound");
    const std::vector<int32_t> no_ids={0,4,8};const std::vector<uint8_t> none={0,0,0};
    check(group.post(0,x.data(),no_ids.data(),weights.data(),none.data(),T,K,err)&&!group.posted()&&group.finish(err),
          "a no-work group window sends no requests");
    group.close();

    // Once worker 0 is READY, setup of worker 1 may take longer than worker 0's idle frame deadline.
    // STATUS keepalives must preserve the first connection until all partitions are ready.
    {
        Svc early(WorkerOptions{.deadline_ms=150}),slow(WorkerOptions{.deadline_ms=1000});
        check(early.start()&&slow.start(),"services for sequential setup keepalive started");
        auto o0=fetch.open(early.port);o0.partition=p0;o0.deadline_ms=1000;
        auto o1=fetch.open(slow.port);o1.partition=p1;o1.deadline_ms=1000;
        const auto fetch_blob=o1.fetch;
        o1.fetch=[fetch_blob](int64_t l,int64_t e,uint8_t* dst,uint32_t bytes) {
            std::this_thread::sleep_for(std::chrono::milliseconds(350));
            return fetch_blob(l,e,dst,bytes);
        };
        sparse[0]=p0[0]|p1[0];
        CpuWorkerGroup kept;
        check(kept.open({o0,o1},sparse,err),"earlier CPU worker stayed alive during later setup");
        check(kept.ready()&&!kept.dead(),"sequentially loaded CPU group is ready");
    }

    // Missing/duplicate/out-of-range ownership is rejected before any connection is opened.
    auto bad=opens;bad[0].partition[0]&=(uint8_t)~(1u<<2);
    check(!CpuWorkerGroup::validate(bad,complement,err),"missing CPU ownership is refused");
    bad=opens;bad[0].partition[0]|=(uint8_t)(1u<<5);
    check(!CpuWorkerGroup::validate(bad,complement,err),"duplicate CPU ownership is refused");
    bad=opens;std::fill(bad[0].partition.begin(),bad[0].partition.end(),0);
    check(!CpuWorkerGroup::validate(bad,complement,err),"an empty configured CPU participant is refused");
    bad=opens;for(auto& o:bad)o.n_expert=15;bad[0].partition[1]|=0x80;
    std::vector<uint8_t> fifteen=complement;for(int l=0;l<kLayers;++l)fifteen[(size_t)l*2+1]&=0x7f;
    check(!CpuWorkerGroup::validate(bad,fifteen,err),"out-of-range CPU ownership is refused");
}

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
        if (mode == AdverseWorker::timeout)
            check(took < 3.0, "the deadline fired instead of hanging");
    }
    // Every adverse reply mode also runs through a real two-peer group. One failed participant drains the
    // successful peer and fails the whole group, so the caller has no success on which to commit a partial.
    for (const auto mode : {AdverseWorker::refusal, AdverseWorker::malformed_reply,
                            AdverseWorker::disconnect, AdverseWorker::timeout,
                            AdverseWorker::stale_reply}) {
        Svc good;check(good.start(),"good service for group failure control");
        AdverseWorker bad(mode);bad.start();
        const size_t stride=((size_t)kExperts+7)/8;
        std::vector<uint8_t> complement((size_t)kLayers*stride,0), p0=complement, p1=complement;
        p0[0]|=(uint8_t)(1u<<2);p1[0]|=(uint8_t)(1u<<5);complement[0]=p0[0]|p1[0];
        Fetcher fetch{&blobs};auto a=fetch.open(good.port);a.partition=p0;
        auto b=fetch.open(bad.port);b.partition=p1;b.deadline_ms=400;
        b.fetch=[](int64_t,int64_t,uint8_t* dst,uint32_t bytes){std::memset(dst,0xAB,bytes);return true;};
        strata::worker::CpuWorkerGroup group;
        const bool group_open=group.open({a,b},complement,err);
        check(group_open,"mixed-success CPU group opened");
        if(!group_open)return;
        const int32_t ids[2]={2,5};const float weights[2]={0.4f,0.6f};
        std::vector<float> x(kEmbd,0.1f);const uint8_t owned[2]={1,1};
        check(group.post(0,x.data(),ids,weights,owned,1,2,err),"mixed-success group posted both peers");
        const auto t0=std::chrono::steady_clock::now();
        check(!group.finish(err),"one CPU peer failure rejects the whole group");
        check(!group.posted()&&group.dead()&&group.served(0),"successful CPU peer was drained before failure propagated");
        check(group.stats(0).requests==1&&group.stats(1).requests==1,"both CPU peer requests were published");
        if(mode==AdverseWorker::timeout)check(std::chrono::duration<double>(
             std::chrono::steady_clock::now()-t0).count()<3.0,"group timeout failed without hanging");
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
        Svc svc;
        check(svc.start(), "the service started");
        parity_battery(fmt, blobs, svc.port);
    }
    group_battery(fmt, blobs);
    failure_battery(fmt, blobs);
    if (fails) {
        std::fprintf(stderr, "expert_tier_test: %d failure(s)\n", fails);
        return 1;
    }
    std::printf("expert_tier_test: all checks passed\n");
    return 0;
}
