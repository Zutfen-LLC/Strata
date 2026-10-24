// src/worker/expert_worker.cpp - inferswarm #348: the resident CPU-expert worker service.
//
// The session state machine, the arena and the native execution.  Every frame is bounds-checked against
// the wire contract before anything is allocated or executed, and every failure is a typed status plus a
// counter bump - a rejected request lets the connection continue, a malformed frame closes it.
#include "strata/worker/expert_worker.hpp"

#include "strata/worker/local_source.hpp"
#include "wire_sock.hpp"
#include <algorithm>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace strata::worker {
namespace {

using strata::kernels::cpu::kNativeActBytes;
using strata::kernels::cpu::NativeFmt;

// #348 scope 3: the layouts this worker runs are the ones the focused tests actually pack and compare -
// Q8_0 gate/up with Q8_0 down today.  A new format joins this table only with its own packed fixture and
// direct-versus-transport coverage, never by assumption.
struct SupportedLayout {
    uint16_t gu, d;
};
constexpr SupportedLayout kSupportedLayouts[] = {{8, 8}};   // GGML_TYPE_Q8_0

uint64_t mono_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool layout_supported(uint16_t gu, uint16_t d) {
    for (const SupportedLayout& s : kSupportedLayouts)
        if (s.gu == gu && s.d == d) return true;
    return false;
}

struct Cur {
    const uint8_t* p;
    size_t n, i = 0;
    bool take(size_t k) const { return i + k <= n; }
    uint64_t u64() {
        uint64_t v = 0;
        if (take(8)) {
            for (int b = 0; b < 8; ++b) v |= (uint64_t) p[i + b] << (8 * b);
            i += 8;
        }
        return v;
    }
    uint32_t u32() {
        uint32_t v = 0;
        if (take(4)) {
            for (int b = 0; b < 4; ++b) v |= (uint32_t) p[i + b] << (8 * b);
            i += 4;
        }
        return v;
    }
    uint16_t u16() {
        uint16_t v = 0;
        if (take(2)) {
            v = (uint16_t) (p[i] | p[i + 1] << 8);
            i += 2;
        }
        return v;
    }
};

void put64(std::vector<uint8_t>& o, uint64_t v) {
    for (int b = 0; b < 8; ++b) o.push_back((uint8_t) (v >> (8 * b)));
}
void put32(std::vector<uint8_t>& o, uint32_t v) {
    for (int b = 0; b < 4; ++b) o.push_back((uint8_t) (v >> (8 * b)));
}

}  // namespace

// ---- the ordered-reduction contract, shared by the service and the reference -------------

void ExpertWorkerService::accumulate_partial(int T, int K, const uint8_t* mask, const float* weights,
                                             const float* rows, int n_embd, float* partials) {
    for (int t = 0; t < T; ++t)
        for (int e = 0; e < n_embd; ++e) partials[(size_t) t * n_embd + e] = 0.f;
    for (int i = 0; i < T * K; ++i) {   // entries in frame order: the slot accumulation order
        if (mask[i] == 0) continue;
        const int t = i / K;
        const float w = weights[i];
        const float* row = rows + (size_t) i * n_embd;
        float* acc = partials + (size_t) t * n_embd;
        for (int e = 0; e < n_embd; ++e) acc[e] = acc[e] + w * row[e];
    }
}

bool ExpertWorkerService::direct_native_partial(const NativeFmt& fmt, int T, int K,
                                                const uint8_t* mask, const float* weights,
                                                const float* activations,
                                                const uint8_t* const* blob_of_entry,
                                                std::vector<float>& partials, std::string& err) {
    if (T < 1 || T > kMaxT || K < 1 || K > kMaxK) {
        err = "direct_native_partial: T or K outside the compiled bounds";
        return false;
    }
    partials.assign((size_t) T * fmt.n_embd, 0.f);
    std::vector<float> rows((size_t) T * K * fmt.n_embd, 0.f);
    std::vector<uint8_t> nact((size_t) T * strata::kernels::cpu::kNativeActBytes, 0);
    // one quantized activation per token with CPU work, exactly as the engine's dispatch does
    for (int t = 0; t < T; ++t) {
        bool any = false;
        for (int j = 0; j < K; ++j) any = any || mask[t * K + j] != 0;
        if (any)
            strata::kernels::cpu::native_quant_act(fmt, activations + (size_t) t * fmt.n_embd,
                                                   nact.data() + (size_t) t * kNativeActBytes);
    }
    std::vector<float> ff((size_t) fmt.n_ff, 0.f);
    std::vector<uint8_t> hq((size_t) fmt.h_bytes, 0);
    for (int i = 0; i < T * K; ++i) {
        if (mask[i] == 0) continue;
        const uint8_t* blob = blob_of_entry[i];
        const int t = i / K;
        const void* act = nact.data() + (size_t) t * kNativeActBytes;
        float* ffrow = ff.data();
        strata::kernels::cpu::native_gu_rows(fmt, blob, &act, 1, &ffrow, 0, (int) fmt.n_ff);
        strata::kernels::cpu::native_quant_h(fmt, ffrow, hq.data());
        const void* hqv = hq.data();
        float* outrow = rows.data() + (size_t) i * fmt.n_embd;
        strata::kernels::cpu::native_down_rows(fmt, blob, &hqv, 1, &outrow, 0, (int) fmt.n_embd);
    }
    accumulate_partial(T, K, mask, weights, rows.data(), (int) fmt.n_embd, partials.data());
    return true;
}

// ---- the service --------------------------------------------------------------------------

struct ExpertWorkerService::Impl {
    WorkerOptions opt;
    WorkerCounters* counters = nullptr;
    RpcTimingCounters* timing = nullptr;
    sock_fd listener = kBadFd;
    std::vector<uint32_t> allowed_peers;
    std::atomic<bool> stopping{false};

    enum class State { Empty, Setup, Ready };
    State state = State::Empty;
    uint64_t session = 0, model_id = 0, partition_id = 0, session_serial = 0;
    SetupManifest manifest{};
    std::unique_ptr<LocalSource> local;
    bool local_mode = false, fallback = false;
    uint64_t local_start_ns = 0;
    std::vector<uint8_t> local_attempt; // 0 untried, 1 local hit, 2 absent, 3 invalid
    NativeFmt fmt{};
    std::vector<uint64_t> slot_off;      ///< owned slots, layer-major order: their arena offsets
    std::vector<uint32_t> slot_filled;   ///< bytes present (Setup phase)
    std::vector<uint8_t> slot_bad;       ///< a digest already failed for this slot
    std::vector<int32_t> slot_of;        ///< (layer * n_expert + expert) -> slot index, -1 = not owned
    std::unique_ptr<uint8_t[]> arena;
    uint64_t arena_bytes = 0;
    uint64_t last_sequence = 0;
    bool have_sequence = false;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    uint64_t service_log_bytes = 0;
    bool service_log_suppressed = false;

    // Only the serving thread touches this storage, after validation and until the pool has parked.
    // Sizes are bounded by kMaxT/kMaxK and the negotiated geometry. Never retain an upload here.
    struct InferenceWorkspace {
        InferRequest request;
        InferReply reply;
        std::vector<uint8_t> nact, payload;
        std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
        std::vector<int16_t> job_of;
        std::vector<float> rows;
        std::vector<uint16_t> ids;
        std::vector<int> uses;
    } scratch;
    std::vector<uint8_t> send_buffer;

    void logf(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        if (!opt.residency_forever) {
            std::fputs("strata-worker: ", stderr);
            std::vfprintf(stderr, fmt, ap);
            std::fputc('\n', stderr);
        } else if (!service_log_suppressed) {
            char message[2048]{};
            const int size = std::vsnprintf(message, sizeof message, fmt, ap);
            message[sizeof message - 1] = 0;
            const size_t bytes = size < 0 ? 0 : std::min((size_t)size, sizeof message - 1);
            if (service_log_bytes + bytes + 16 <= 8u * 1048576) {
                std::fprintf(stderr,"strata-worker: %s\n",message);
                service_log_bytes += bytes + 16;
            } else {
                std::fputs("strata-worker: service detail log quota reached; further detail suppressed; use status counters and guardian journal\n",stderr);
                service_log_suppressed = true;
            }
        }
        std::fflush(stderr);
        va_end(ap);
    }

    Impl(WorkerOptions o, WorkerCounters* c, RpcTimingCounters* t)
        : opt(o), counters(c), timing(t),
          pool(std::make_unique<strata::kernels::cpu::ExpertPool>(o.workers)) {}

    void fail(uint16_t status) {
        ++counters->failures;
        counters->last_error = status;
    }

    // Session identity/replay/setup controls are never retained across connections.
    // Weight storage below survives ONLY after a complete persistent READY.
    bool persistent_session = false, resident_complete = false, cache_binding = false;
    Digest128 resident_identity{};
    uint64_t idle_since_ns = 0;

    bool resident_expired() const {
        return resident_complete && !opt.residency_forever &&
               mono_ns() - idle_since_ns >= (uint64_t)opt.residency_ms * 1000000;
    }
    void unload_weights(const char* reason) {
        scratch = {};
        if (arena_bytes) logf("cache unload: reason=%s bytes=%llu", reason, (unsigned long long)arena_bytes);
        resident_complete = false;
        resident_identity = {};
        idle_since_ns = 0;
        arena.reset();
        arena_bytes = counters->resident_bytes = 0;
        slot_off.clear();
        slot_filled.clear();
        slot_bad.clear();
        slot_of.clear();
        manifest = {};
        fmt = {};
    }
    void void_session() {
        scratch = {};
        std::vector<uint8_t>().swap(send_buffer);
        state = State::Empty;
        local_mode = false;
        local_attempt.clear();
        if (local)
            local->close_files();
        persistent_session = cache_binding = false;
        session = model_id = partition_id = last_sequence = 0;
        have_sequence = false;
        if (!resident_complete) unload_weights("session-end");
    }

    void expire_resident() {
        if (!resident_expired()) return;
        ++counters->cache_expirations;
        void_session();
        unload_weights("expiry");
    }

    // ---- frame io ----
    bool recv_frame(sock_fd fd, WireFrame& f, bool& malformed);
    bool send_frame(sock_fd fd, uint16_t msg, uint16_t status, const std::vector<uint8_t>& payload);

    // ---- handlers: each replies once and returns false only to close the connection ----
    void on_setup_begin(sock_fd fd, const uint8_t *p, size_t n, bool use_local = false);
    void on_local_load(sock_fd fd, const uint8_t *p, size_t n);
    bool local_expired() const {
        return local_mode && mono_ns() - local_start_ns >= (uint64_t)opt.local_setup_ms * 1000000;
    }
    void on_upload(sock_fd fd, const uint8_t* p, size_t n);
    void on_ready(sock_fd fd, const uint8_t* p, size_t n);
    void on_infer(sock_fd fd, const uint8_t* p, size_t n, uint64_t request_start_ns);
    void on_status(sock_fd fd);
    void on_cache_status(sock_fd fd);
    void serve_connection(sock_fd fd);
    struct ExecuteTiming { uint64_t prepare, pool, reduce; };
    bool execute(const InferRequest& r, std::vector<float>& partials, std::string& err, ExecuteTiming* timing);
};

bool ExpertWorkerService::Impl::recv_frame(sock_fd fd, WireFrame& f, bool& malformed) {
    malformed = false;
    uint8_t head[kHeaderBytes];
    const auto cancelled = [&] { return stopping.load(std::memory_order_acquire) || resident_expired() ||
               (persistent_session && state == State::Setup && local_expired()); };
    if (!sock::recv_all_controlled(fd, head, sizeof head, opt.deadline_ms, cancelled)) return false;
    const uint32_t magic = (uint32_t) head[0] | (uint32_t) head[1] << 8 | (uint32_t) head[2] << 16 |
                           (uint32_t) head[3] << 24;
    f.msg = (uint16_t) (head[4] | head[5] << 8);
    f.status = (uint16_t) (head[6] | head[7] << 8);
    uint64_t len = 0;
    for (int b = 0; b < 8; ++b) len |= (uint64_t) head[8 + b] << (8 * b);
    size_t cap = 0;
    switch (f.msg) {
    case MSG_LOCAL_SETUP:
        cap = kSetupFrameMax - kHeaderBytes + 24;
        break;
    case MSG_LOCAL_LOAD:
        cap = 12 + kLocalBatchMax * 4;
        break;
    case MSG_SETUP_BEGIN:
        cap = kSetupFrameMax - kHeaderBytes;
        break;
    case MSG_UPLOAD:
        cap = kUploadHeadBytes + opt.upload_chunk_max;
        break;
    case MSG_INFER:
        cap = kInferPayloadMax;
        break;
    case MSG_SETUP_ACK:
        cap = 64;
        break;
    case MSG_UPLOAD_ACK:
        cap = 32;
        break;
    case MSG_READY:
    case MSG_SHUTDOWN:
        cap = 16;
        break;
    case MSG_READY_ACK:
        cap = 32;
        break;
    case MSG_INFER_ACK:
        cap = kReplyPayloadMax;
        break;
    case MSG_CACHE_STATUS:
    case MSG_UNLOAD:
        cap = 0;
        break;
    case MSG_STATUS:
        cap = 16;
        break;
    case MSG_STATUS_ACK:
        cap = 128;
        break;
    default:
        malformed = true;
        return false;
    }
    if (magic != kWireMagic || len > cap) { malformed = true; return false; }
    // Avoid geometric growth from a smaller upload buffer spilling over the retained inference cap.
    if (f.msg == MSG_INFER) f.payload.reserve((size_t) len);
    f.payload.resize((size_t) len);
    if (len > 0 && !sock::recv_all_controlled(fd, f.payload.data(), f.payload.size(), opt.deadline_ms, cancelled)) return false;
    return true;
}

bool ExpertWorkerService::Impl::send_frame(sock_fd fd, uint16_t msg, uint16_t status,
                                           const std::vector<uint8_t>& payload) {
    if (payload.size() > kMaxFrameBytes) return false;
    auto& buf = send_buffer;
    encode_frame(buf, msg, status, payload.data(), payload.size());
    // An admitted inference reply is active work: its existing I/O deadline still applies.
    // Telemetry/setup/refusal replies must not pin idle weights behind a slow reader.
    const bool active_reply = msg == MSG_INFER_ACK && status == ST_OK;
    return sock::send_all_controlled(fd, buf.data(), buf.size(), opt.deadline_ms, [&] {
        return stopping.load(std::memory_order_acquire) || (!active_reply && resident_expired());
    });
}

void ExpertWorkerService::Impl::on_setup_begin(sock_fd fd, const uint8_t *p, size_t n, bool use_local) {
    expire_resident();
    void_session(); // even a refused setup invalidates the old session, never the verified cache
    const uint16_t ack = use_local ? MSG_LOCAL_SETUP_ACK : MSG_SETUP_ACK;
    Digest128 identity{};
    bool want_fallback = false, want_resident = false;
    if (use_local) {
        Cur c{p, n};
        const auto version = c.u32(), flags = c.u32();
        identity = {c.u64(), c.u64()};
        if (n < 24 || version != kLocalVersion || flags > 3 || !local ||
            ((flags & 1) && !opt.local_upload_fallback) ||
            ((flags & kLocalResident) && opt.residency_ms == 0 && !opt.residency_forever)) {
            fail(ST_LOCAL_UNAVAILABLE);
            send_frame(fd, ack, ST_LOCAL_UNAVAILABLE, {});
            return;
        }
        want_fallback = flags & 1;
        want_resident = flags & kLocalResident;
        p += 24;
        n -= 24;
    }
    SetupManifest m;
    if (!decode_setup_begin(p, n, m)) {
        fail(ST_BAD_REQUEST);
        send_frame(fd, ack, ST_BAD_REQUEST, {});
        return;
    }
    if (resident_complete) {
        SetupManifest candidate = m, frozen = manifest;
        candidate.nonce = frozen.nonce = 0;
        std::memset(candidate.model_name, 0, sizeof candidate.model_name);
        std::memset(frozen.model_name, 0, sizeof frozen.model_name); // log label, not identity
        std::vector<uint8_t> a, b;
        const bool same = want_resident && identity == resident_identity &&
                          encode_setup_begin(candidate, a) && encode_setup_begin(frozen, b) && a == b;
        if (!same) {
            fail(ST_CACHE_MISMATCH);
            logf("cache mismatch: retained %llu B; unload or wait for expiry", (unsigned long long)arena_bytes);
            send_frame(fd, ack, ST_CACHE_MISMATCH, {});
            return;
        }
    }
    if (use_local && (local->manifest().identity != identity || !local->manifest().matches(m))) {
        fail(ST_LOCAL_INVALID);
        send_frame(fd, ack, ST_LOCAL_INVALID, {});
        return;
    }
    if (!layout_supported(m.gu_type, m.d_type)) {
        logf("setup refused: layout gu=%u d=%u is not one this worker runs (model '%s')", m.gu_type,
             m.d_type, m.model_name);
        fail(ST_UNSUPPORTED_LAYOUT);
        send_frame(fd, ack, ST_UNSUPPORTED_LAYOUT, {});
        return;
    }
    NativeFmt f;
    std::string err;
    if (!strata::kernels::cpu::native_fmt(m.gu_type, m.d_type, m.n_embd, m.n_ff, f, err)) {
        logf("setup refused: %s (model '%s')", err.c_str(), m.model_name);
        fail(ST_UNSUPPORTED_LAYOUT);
        send_frame(fd, ack, ST_UNSUPPORTED_LAYOUT, {});
        return;
    }
    const uint64_t owned = m.owned_count();
    const uint64_t need = owned * (uint64_t) f.bytes;
    if (need > opt.memory_budget) {
        logf("setup refused: %llu owned blobs need %llu B, budget %llu B (model '%s')",
             (unsigned long long) owned, (unsigned long long) need, (unsigned long long) opt.memory_budget,
             m.model_name);
        fail(ST_OVER_BUDGET);
        send_frame(fd, ack, ST_OVER_BUDGET, {});
        return;
    }
    // a SETUP_BEGIN voids any previous session: a restart is a new session, old ids cannot continue
    void_session();
    local_mode = use_local;
    fallback = want_fallback;
    if (use_local) {
        local_start_ns = mono_ns();
        counters->local_bytes = counters->local_hits = counters->local_misses = 0;
        counters->local_read_ns = counters->local_verify_ns = counters->local_setup_ns = 0;
    }
    persistent_session = want_resident;
    cache_binding = resident_complete;
    if (!cache_binding) {
        manifest = m;
        resident_identity = identity;
        fmt = f;
        slot_of.assign((size_t) m.n_layers * m.n_expert, -1);
        slot_off.clear();
        for (int l = 0; l < m.n_layers; ++l)
            for (int e = 0; e < m.n_expert; ++e)
                if (m.owns(l, e)) {
                    slot_of[(size_t) l * m.n_expert + e] = (int32_t) slot_off.size();
                    slot_off.push_back((uint64_t) slot_off.size() * f.bytes);
                }
        slot_filled.assign(slot_off.size(), 0);
        slot_bad.assign(slot_off.size(), 0);
        local_attempt.assign(slot_off.size(), 0);
        if (need > 0) {
            arena.reset(new uint8_t[need]);   // exactly the assigned set: the budget was checked above
            arena_bytes = need;
            counters->resident_bytes = need;
            counters->peak_arena_bytes = std::max(counters->peak_arena_bytes, need);
            ++counters->arena_allocations;
        }
    }
    model_id = m.model_id;
    partition_id = m.partition_id;
    if (cache_binding) {
        ++counters->cache_hits;
        counters->reused_bytes += arena_bytes;
    }
    ++session_serial;
    session = m.nonce ^ (mono_ns() << 1) ^ (session_serial * 0x9e3779b97f4a7c15ull);
    if (session == 0) session = 1;
    state = State::Setup;
    logf("session %llx: model '%s' quant '%s' %ux%u gu=%u d=%u, %llu owned blobs of %llu B",
         (unsigned long long) session, m.model_name, m.quant_name, m.n_layers, m.n_expert, m.gu_type,
         m.d_type, (unsigned long long) owned, (unsigned long long) f.bytes);
    std::vector<uint8_t> reply;
    put64(reply, session);
    put64(reply, (uint64_t) f.bytes);
    put32(reply, opt.upload_chunk_max);
    if (use_local) {
        put32(reply, kLocalVersion);
        put32(reply, kLocalBatchMax);
        if (want_resident) {
            put32(reply, cache_binding ? 1 : 0);
            put64(reply, cache_binding ? arena_bytes : 0);
            logf("cache setup: hit=%d reused=%llu B resident=%llu B session=%llx", cache_binding,
                 (unsigned long long)(cache_binding ? arena_bytes : 0),
                 (unsigned long long)arena_bytes, (unsigned long long)session);
        }
    }
    send_frame(fd, ack, ST_OK, reply);
}

// Each batch writes at most 32 blobs directly into the assigned arena. Ranges are
// sorted by file and offset. There is no second weight buffer or mmap payload.
void ExpertWorkerService::Impl::on_local_load(sock_fd fd, const uint8_t *p, size_t n) {
    auto refuse = [&](uint16_t st) {
        fail(st);
        send_frame(fd, MSG_LOCAL_LOAD_ACK, st, {});
    };
    Cur c{p, n};
    const auto sid = c.u64();
    const auto count = c.u32();
    if (state != State::Setup || !local_mode || cache_binding || sid != session) {
        refuse(ST_WRONG_STATE);
        return;
    }
    if (!count || count > kLocalBatchMax || n != 12 + 4 * count) {
        refuse(ST_BAD_REQUEST);
        return;
    }
    struct Entry {
        uint16_t l, e;
        int32_t slot;
        bool hit = true;
        std::array<bool, 3> present{true, true, true};
    };
    std::vector<Entry> entries;
    uint32_t previous = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const auto l = c.u16(), e = c.u16();
        const uint32_t key = (uint32_t)l * manifest.n_expert + e;
        if (l >= manifest.n_layers || e >= manifest.n_expert || !manifest.owns(l, e)) {
            refuse(ST_NOT_OWNED);
            return;
        }
        const int32_t slot = slot_of[key];
        if ((i && key <= previous) || local_attempt[slot] || slot_filled[slot] || slot_bad[slot]) {
            refuse(ST_DUPLICATE);
            return;
        }
        previous = key;
        entries.push_back({l, e, slot, true});
    }
    const auto start = mono_ns();
    const uint64_t batch_ns = (uint64_t)opt.deadline_ms * 1000000;
    auto cancelled = [&] {
        if (stopping.load(std::memory_order_acquire) || local_expired() || mono_ns() - start >= batch_ns)
            return true;
#ifndef _WIN32
        char byte;
        const auto rc = ::recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
        if (rc == 0 || rc > 0 || (rc < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR))
            return true;
#endif
        return false;
    };
    auto abort = [&](uint16_t st, const std::string &why) {
        for (const auto &e : entries) {
            slot_bad[e.slot] = 1;
            local_attempt[e.slot] = 3;
        }
        counters->local_setup_ns = mono_ns() - local_start_ns;
        logf("local setup stopped: %s (%s)", why.c_str(), status_name(st));
        refuse(st);
    };
    struct Read {
        uint32_t file;
        uint64_t offset;
        size_t bytes;
        uint8_t *dst;
    };
    std::vector<Read> reads;
    std::string err;
    for (auto &e : entries) {
        size_t dst = 0;
        const auto &rr = local->manifest().layers[e.l];
        for (size_t ri = 0; ri < 3; ++ri) {
            const auto &r = rr[ri];
            if (cancelled()) {
                abort(ST_DEADLINE, "deadline or cancellation");
                return;
            }
            const int f = local->file(r.file, err);
            if (f == -2) {
                abort(ST_LOCAL_INVALID, err);
                return;
            }
            if (f == -1) {
                e.hit = false;
                e.present[ri] = false;
            }
        }
        if (!e.hit) {
            local_attempt[e.slot] = 2;
            ++counters->local_misses;
        }
        for (size_t ri = 0; ri < 3; ++ri) {
            const auto &r = rr[ri];
            if (e.present[ri])
                reads.push_back({r.file, r.offset + (uint64_t)e.e * r.bytes, (size_t)r.bytes,
                                 arena.get() + slot_off[e.slot] + dst});
            dst += r.bytes;
        }
    }
    std::sort(reads.begin(), reads.end(), [](const Read &a, const Read &b) {
        return a.file < b.file || (a.file == b.file && a.offset < b.offset);
    });
    for (const auto &r : reads)
        for (size_t i = 0; i < r.bytes;) {
            if (cancelled()) {
                abort(ST_DEADLINE, "deadline or cancellation");
                return;
            }
            const size_t bytes = std::min<size_t>(256u << 10, r.bytes - i);
            const auto t = mono_ns();
            const bool ok = local->read(r.file, r.offset + i, r.dst + i, bytes, err);
            counters->local_read_ns += mono_ns() - t;
            if (!ok) {
                abort(ST_LOCAL_INVALID, err);
                return;
            }
            counters->local_bytes += bytes;
            counters->loaded_bytes += bytes;
            i += bytes;
        }
    auto verify = [&](const uint8_t *data, size_t bytes, Digest128 want) {
        const auto t = mono_ns();
        Digest128 d{0xcbf29ce484222325ull, 0x9e3779b97f4a7c15ull};
        for (size_t i = 0; i < bytes;) {
            if (cancelled()) {
                abort(ST_DEADLINE, "deadline or cancellation");
                return false;
            }
            const size_t end = std::min<size_t>(bytes, i + (64u << 10));
            counters->verified_bytes += end - i;
            for (; i < end; ++i) {
                d.a = (d.a ^ data[i]) * 0x100000001b3ull;
                d.b = (d.b ^ data[bytes - 1 - i]) * 0x100000001b3ull;
            }
        }
        counters->local_verify_ns += mono_ns() - t;
        if (d != want) {
            abort(ST_DIGEST_MISMATCH, "local expert digest mismatch");
            return false;
        }
        return true;
    };
    for (const auto &e : entries) {
        const auto data = arena.get() + slot_off[e.slot];
        const size_t key = (size_t)e.l * manifest.n_expert + e.e;
        if (e.hit) {
            if (!verify(data, fmt.bytes, local->manifest().digests[key]))
                return;
            local_attempt[e.slot] = 1;
            slot_filled[e.slot] = fmt.bytes;
            ++counters->local_hits;
        } else {
            size_t dst = 0;
            for (size_t r = 0; r < 3; ++r) {
                const auto bytes = local->manifest().layers[e.l][r].bytes;
                if (e.present[r] && !verify(data + dst, bytes, local->manifest().role_digests[key][r]))
                    return;
                dst += bytes;
            }
        }
    }
    if (cancelled()) {
        abort(ST_DEADLINE, "deadline or cancellation");
        return;
    }
    counters->local_setup_ns = mono_ns() - local_start_ns;
    std::vector<uint8_t> reply;
    put64(reply, session);
    put32(reply, count);
    put64(reply, counters->local_bytes);
    put64(reply, counters->local_hits);
    put64(reply, counters->local_misses);
    put64(reply, counters->local_read_ns);
    put64(reply, counters->local_verify_ns);
    put64(reply, counters->local_setup_ns);
    for (const auto &e : entries)
        reply.push_back(e.hit ? 1 : 0);
    send_frame(fd, MSG_LOCAL_LOAD_ACK, ST_OK, reply);
}

void ExpertWorkerService::Impl::on_upload(sock_fd fd, const uint8_t* p, size_t n) {
    Cur c{p, n};
    const uint64_t sid = c.u64();
    const uint16_t layer = c.u16();
    const uint16_t expert = c.u16();
    const uint32_t total_len = c.u32();
    const uint32_t chunk_off = c.u32();
    const uint32_t chunk_len = c.u32();
    Digest128 declared{};
    if (c.take(16)) {
        for (int b = 0; b < 8; ++b) declared.a |= (uint64_t) p[c.i + b] << (8 * b);
        c.i += 8;
        for (int b = 0; b < 8; ++b) declared.b |= (uint64_t) p[c.i + b] << (8 * b);
        c.i += 8;
    } else {
        fail(ST_BAD_REQUEST);
        send_frame(fd, MSG_UPLOAD_ACK, ST_BAD_REQUEST, {});
        return;
    }
    if (n - c.i != chunk_len) {
        fail(ST_BAD_REQUEST);
        send_frame(fd, MSG_UPLOAD_ACK, ST_BAD_REQUEST, {});
        return;
    }
    if (state != State::Setup || cache_binding || sid != session) {
        fail(state == State::Empty ? ST_WRONG_STATE : ST_STALE_SESSION);
        send_frame(fd, MSG_UPLOAD_ACK, state == State::Empty ? ST_WRONG_STATE : ST_STALE_SESSION, {});
        return;
    }
    if (layer >= manifest.n_layers || expert >= manifest.n_expert ||
        slot_of[(size_t) layer * manifest.n_expert + expert] < 0) {
        logf("upload refused: slot (%u,%u) is outside this worker's partition", layer, expert);
        fail(ST_NOT_OWNED);
        send_frame(fd, MSG_UPLOAD_ACK, ST_NOT_OWNED, {});
        return;
    }
    const int32_t slot = slot_of[(size_t) layer * manifest.n_expert + expert];
    if (local_mode) {
        const auto &expected = local->manifest().digests[(size_t)layer * manifest.n_expert + expert];
        const uint16_t st =
            local_expired()
                ? ST_DEADLINE
                : (declared != expected ? ST_DIGEST_MISMATCH
                                        : (!fallback || local_attempt[slot] != 2 ? ST_WRONG_STATE : ST_OK));
        if (st != ST_OK) {
            fail(st);
            send_frame(fd, MSG_UPLOAD_ACK, st, {});
            return;
        }
    }
    if (total_len != (uint32_t) fmt.bytes || chunk_len == 0 || chunk_len > opt.upload_chunk_max ||
        (uint64_t) chunk_off + chunk_len > total_len) {
        fail(ST_BAD_REQUEST);
        send_frame(fd, MSG_UPLOAD_ACK, ST_BAD_REQUEST, {});
        return;
    }
    if (slot_bad[(size_t) slot]) {
        // the slot already failed its digest: only a session restart retries it
        fail(ST_DIGEST_MISMATCH);
        send_frame(fd, MSG_UPLOAD_ACK, ST_DIGEST_MISMATCH, {});
        return;
    }
    if (slot_filled[(size_t) slot] == total_len) {
        // a completed slot never re-receives: a retry is a session restart, not an overwrite
        fail(ST_DUPLICATE);
        send_frame(fd, MSG_UPLOAD_ACK, ST_DUPLICATE, {});
        return;
    }
    if (chunk_off != slot_filled[(size_t) slot]) {   // chunks arrive in order, from byte 0 up
        fail(ST_BAD_REQUEST);
        send_frame(fd, MSG_UPLOAD_ACK, ST_BAD_REQUEST, {});
        return;
    }
    std::memcpy(arena.get() + slot_off[(size_t) slot] + chunk_off, p + c.i, chunk_len);
    slot_filled[(size_t) slot] += chunk_len;
    counters->setup_weight_bytes += chunk_len;
    counters->loaded_bytes += chunk_len;
    uint16_t status = ST_OK;
    if (slot_filled[(size_t) slot] == total_len) {
        counters->verified_bytes += total_len;
        const Digest128 got = blob_digest(arena.get() + slot_off[(size_t) slot], total_len);
        if (got != declared) {
            logf("slot (%u,%u): digest mismatch (%llx/%llx vs %llx/%llx)", layer, expert,
                 (unsigned long long) got.a, (unsigned long long) got.b, (unsigned long long) declared.a,
                 (unsigned long long) declared.b);
            slot_bad[(size_t) slot] = 1;
            status = ST_DIGEST_MISMATCH;
            fail(ST_DIGEST_MISMATCH);
        }
    }
    std::vector<uint8_t> reply;
    put64(reply, session);
    put32(reply, slot_filled[(size_t) slot]);
    send_frame(fd, MSG_UPLOAD_ACK, status, reply);
}

void ExpertWorkerService::Impl::on_ready(sock_fd fd, const uint8_t* p, size_t n) {
    Cur c{p, n};
    const uint64_t sid = c.u64();
    if (n != 8 || state != State::Setup || sid != session) {
        const uint16_t s = state == State::Empty ? ST_WRONG_STATE : ST_STALE_SESSION;
        fail(s);
        send_frame(fd, MSG_READY_ACK, s, {});
        return;
    }
    if (local_expired()) {
        fail(ST_DEADLINE);
        send_frame(fd, MSG_READY_ACK, ST_DEADLINE, {});
        return;
    }
    uint64_t missing = 0;
    for (size_t s = 0; s < slot_filled.size(); ++s)
        if (slot_filled[s] != (uint32_t) fmt.bytes || slot_bad[s]) ++missing;
    if (missing > 0) {
        logf("readiness refused: %llu of %llu assigned slots missing or bad",
             (unsigned long long) missing, (unsigned long long) slot_filled.size());
        fail(ST_INCOMPLETE);
        std::vector<uint8_t> reply;
        put64(reply, session);
        put64(reply, missing);
        send_frame(fd, MSG_READY_ACK, ST_INCOMPLETE, reply);
        return;
    }
    if (local_mode) {
        counters->local_setup_ns = mono_ns() - local_start_ns;
        logf("local ready: read=%llu B network=%llu B hits=%llu misses=%llu read_s=%.3f verify_s=%.3f "
             "setup_s=%.3f",
             (unsigned long long)counters->local_bytes, (unsigned long long)counters->setup_weight_bytes,
             (unsigned long long)counters->local_hits, (unsigned long long)counters->local_misses,
             counters->local_read_ns / 1e9, counters->local_verify_ns / 1e9, counters->local_setup_ns / 1e9);
    }
    if (persistent_session) {
        resident_complete = true;
        idle_since_ns = mono_ns();
        local->close_files();
    }
    state = State::Ready;   // frozen: no further setup or upload is accepted on this session
    logf("session %llx ready: %llu blobs, %llu B verified", (unsigned long long) session,
         (unsigned long long) slot_filled.size(), (unsigned long long) arena_bytes);
    std::vector<uint8_t> reply;
    put64(reply, session);
    put64(reply, 0);
    send_frame(fd, MSG_READY_ACK, ST_OK, reply);
}

bool ExpertWorkerService::Impl::execute(const InferRequest& r, std::vector<float>& partials,
                                        std::string& err, ExecuteTiming* measured) {
    using strata::kernels::cpu::ExpertJobMulti;
    const uint64_t prepare_start = measured ? mono_ns() : 0;
    const int T = r.T, K = r.K, H = manifest.n_embd;
    partials.assign((size_t) T * H, 0.f);
    bool any = false;
    for (size_t i = 0; i < r.mask.size(); ++i) any = any || r.mask[i] != 0;
    if (!any) {
        if (measured) measured->prepare = mono_ns() - prepare_start;
        return true;   // zero CPU work: the reply is a zero partial, not an error
    }
    auto& nact = scratch.nact;
    nact.assign((size_t) T * strata::kernels::cpu::kNativeActBytes, 0);
    for (int t = 0; t < T; ++t) {
        bool tok = false;
        for (int j = 0; j < K; ++j) tok = tok || r.mask[(size_t) t * K + j] != 0;
        if (tok)
            strata::kernels::cpu::native_quant_act(fmt, r.activations.data() + (size_t) t * H,
                                                   nact.data() + (size_t) t * kNativeActBytes);
    }
    // distinct experts in first-appearance order; an expert's entries keep frame order (duplicates too)
    auto& jobs = scratch.jobs;
    jobs.clear();
    jobs.reserve(r.mask.size());
    auto& job_of = scratch.job_of;
    job_of.assign((size_t) manifest.n_expert, -1);
    auto& rows = scratch.rows;
    rows.assign((size_t) T * K * H, 0.f);
    for (int i = 0; i < T * K; ++i) {
        if (r.mask[i] == 0) continue;
        const uint16_t e = r.ids[i];
        if (!manifest.owns(r.layer, e) || slot_of[(size_t) r.layer * manifest.n_expert + e] < 0) {
            err = "a masked entry names a slot this worker does not own";
            return false;
        }
        if (job_of[e] < 0) {
            job_of[e] = (int16_t) jobs.size();
            ExpertJobMulti j;
            j.blob = arena.get() + slot_off[(size_t) slot_of[(size_t) r.layer * manifest.n_expert + e]];
            j.nt = 0;
            j.fmt = nullptr;
            jobs.push_back(j);
        }
        ExpertJobMulti& j = jobs[(size_t) job_of[e]];
        const int t = i / K;
        j.nact[j.nt] = nact.data() + (size_t) t * kNativeActBytes;
        j.out[j.nt] = rows.data() + (size_t) i * H;
        ++j.nt;
    }
    if (measured) measured->prepare = mono_ns() - prepare_start;
    const uint64_t t0 = mono_ns(); // existing cpu_service_ns counter, independent of diagnostic timing
    pool->run_split_multi_native(fmt, jobs.data(), (int) jobs.size());
    const uint64_t pool_ns = mono_ns() - t0;
    counters->cpu_service_ns += pool_ns;
    if (measured) measured->pool = pool_ns;
    const uint64_t reduce_start = measured ? mono_ns() : 0;
    accumulate_partial(T, K, r.mask.data(), r.weights.data(), rows.data(), H, partials.data());
    if (measured) measured->reduce = mono_ns() - reduce_start;
    return true;
}

void ExpertWorkerService::Impl::on_infer(sock_fd fd, const uint8_t* p, size_t n,
                                         uint64_t request_start_ns) {
    ++counters->requests;
    const bool measure = timing->enabled();
    const uint64_t decode_start = measure ? mono_ns() : 0;
    bool decode_recorded = false;
    uint8_t tokens = 0;
    uint32_t entries = 0, experts = 0;
    auto finish_decode = [&] {
        if (measure && !decode_recorded) {
            timing->add_phase(RpcTimingPhase::WorkerDecodeValidate, mono_ns() - decode_start);
            decode_recorded = true;
        }
    };
    // every reply path goes through here: status + payload, sent and counted as reply bytes
    auto reply = [&](uint16_t status, const std::vector<uint8_t>& payload, uint64_t encode_start_ns = 0) {
        finish_decode();
        if (measure && !encode_start_ns) encode_start_ns = mono_ns();
        counters->infer_out_bytes += kHeaderBytes + payload.size();
        const bool sent = send_frame(fd, MSG_INFER_ACK, status, payload);
        if (measure) {
            timing->add_phase(RpcTimingPhase::WorkerEncodeSend, mono_ns() - encode_start_ns);
            timing->add_phase(RpcTimingPhase::WorkerRequest, mono_ns() - request_start_ns);
        }
        if (measure)
            timing->finish_call(tokens, entries, experts, kHeaderBytes + n,
                                sent ? kHeaderBytes + payload.size() : 0, sent && status == ST_OK);
        if (sent && status == ST_OK && resident_complete)
            idle_since_ns = mono_ns(); // idle starts after the successful request's reply completes
    };
    if (state == State::Empty) {
        fail(ST_WRONG_STATE);
        reply(ST_WRONG_STATE, {});
        return;
    }
    auto& r = scratch.request;
    if (!decode_infer(p, n, manifest.n_layers, manifest.n_expert, manifest.n_embd, r)) {
        fail(ST_BAD_REQUEST);
        reply(ST_BAD_REQUEST, {});
        return;
    }
    if (measure) {
        tokens = r.T;
        const auto shape = rpc_request_shape(r.mask, r.ids);
        entries = shape.entries;
        experts = shape.distinct_experts;
    }
    if (r.session != session || r.model_id != model_id || r.partition_id != partition_id) {
        logf("inference refused: stale session or identity (session %llx, model %llx, partition %llx)",
             (unsigned long long) r.session, (unsigned long long) r.model_id,
             (unsigned long long) r.partition_id);
        fail(ST_STALE_SESSION);
        reply(ST_STALE_SESSION, {});
        return;
    }
    if (state != State::Ready) {
        fail(ST_NOT_READY);
        reply(ST_NOT_READY, {});
        return;
    }
    if (have_sequence && r.sequence <= last_sequence) {
        // a replayed or older sequence is how stale output would be served as fresh; refuse it outright
        logf("inference refused: sequence %llu is not past %llu", (unsigned long long) r.sequence,
             (unsigned long long) last_sequence);
        fail(ST_BAD_REQUEST);
        reply(ST_BAD_REQUEST, {});
        return;
    }
    for (size_t i = 0; i < r.mask.size(); ++i) {
        if (r.mask[i] == 0) continue;
        if (!manifest.owns(r.layer, r.ids[i])) {
            logf("inference refused: entry %zu names expert %u, owned by the GPU side", i, r.ids[i]);
            fail(ST_NOT_OWNED);
            reply(ST_NOT_OWNED, {});
            return;
        }
    }
    // one expert at most kMaxT times among the owned entries: the pool's job arrays are MAXT wide, so a
    // routing that repeats one expert more often than the window has tokens cannot be expressed
    {
        auto& id = scratch.ids;
        auto& uses = scratch.uses;
        id.clear();
        uses.clear();
        for (size_t i = 0; i < r.mask.size(); ++i) {
            if (r.mask[i] == 0) continue;
            size_t q = 0;
            for (; q < id.size(); ++q)
                if (id[q] == r.ids[i]) break;
            if (q == id.size()) {
                id.push_back(r.ids[i]);
                uses.push_back(0);
            }
            if (++uses[q] > kMaxT) {
                logf("inference refused: expert %u appears more than %d times among the owned entries",
                     r.ids[i], kMaxT);
                fail(ST_BAD_REQUEST);
                reply(ST_BAD_REQUEST, {});
                return;
            }
        }
    }
    for (size_t i = 0; i < r.weights.size(); ++i)
        if (!std::isfinite(r.weights[i])) { fail(ST_NONFINITE); reply(ST_NONFINITE, {}); return; }
    for (size_t i = 0; i < r.activations.size(); ++i)
        if (!std::isfinite(r.activations[i])) { fail(ST_NONFINITE); reply(ST_NONFINITE, {}); return; }

    auto& rep = scratch.reply;
    std::string err;
    finish_decode();
    ExecuteTiming measured;
    if (measure) measured = {};
    if (!execute(r, rep.partials, err, measure ? &measured : nullptr)) {
        logf("inference refused: %s", err.c_str());
        fail(ST_NOT_OWNED);
        reply(ST_NOT_OWNED, {});
        return;
    }
    if (measure) {
        timing->add_phase(RpcTimingPhase::WorkerQuantPrepare, measured.prepare);
        timing->add_phase(RpcTimingPhase::WorkerPool, measured.pool);
        timing->add_phase(RpcTimingPhase::WorkerWeightedReduce, measured.reduce);
    }
    last_sequence = r.sequence;
    have_sequence = true;
    if (opt.verbose > 0)
        logf("infer session %llx seq %llu layer %u T %u K %u: ok", (unsigned long long) session,
             (unsigned long long) r.sequence, r.layer, r.T, r.K);
    rep.session = session;
    rep.sequence = r.sequence;
    const uint64_t encode_start = measure ? mono_ns() : 0;
    auto& payload = scratch.payload;
    encode_infer_ack(rep, payload);
    reply(ST_OK, payload, encode_start);
}

void ExpertWorkerService::Impl::on_status(sock_fd fd) {
    std::vector<uint8_t> reply;
    put64(reply, state == State::Empty ? 0 : session);
    put64(reply, counters->setup_weight_bytes);
    put64(reply, counters->infer_in_bytes);
    put64(reply, counters->infer_out_bytes);
    put64(reply, counters->requests);
    put64(reply, counters->failures);
    put64(reply, counters->cpu_service_ns);
    reply.push_back((uint8_t) (counters->last_error & 0xff));
    reply.push_back((uint8_t) (counters->last_error >> 8));
    reply.push_back(state == State::Ready ? 1 : 0);
    send_frame(fd, MSG_STATUS_ACK, ST_OK, reply);
}

void ExpertWorkerService::Impl::on_cache_status(sock_fd fd) {
    std::vector<uint8_t> reply;
    put32(reply, 1); // cache status extension version
    put32(reply, (uint32_t)state);
    put64(reply, session);
    put64(reply, arena_bytes);
    const auto elapsed = resident_complete ? (mono_ns() - idle_since_ns) / 1000000 : 0;
    put64(reply, resident_complete && opt.residency_forever ? kResidencyNoExpiry :
                 resident_complete && elapsed < (uint64_t)opt.residency_ms ? opt.residency_ms - elapsed : 0);
    for (auto v : {counters->cache_hits, counters->reused_bytes, counters->loaded_bytes,
                   counters->verified_bytes, counters->peak_arena_bytes, counters->arena_allocations,
                   counters->cache_expirations, counters->cache_unloads}) put64(reply, v);
    put32(reply, resident_complete ? 1 : 0);
    send_frame(fd, MSG_CACHE_STATUS_ACK, ST_OK, reply);
}

void ExpertWorkerService::Impl::serve_connection(sock_fd fd) {
    WireFrame f;
    while (!stopping.load(std::memory_order_acquire)) {
        bool malformed = false;
        if (!recv_frame(fd, f, malformed)) {
            if (malformed) {
                logf("connection dropped: malformed frame");
                fail(ST_PROTOCOL);
            } else {
                logf("connection closed (eof or deadline)");
            }
            break;
        }
        expire_resident();
        switch (f.msg) {
        case MSG_LOCAL_SETUP:
            on_setup_begin(fd, f.payload.data(), f.payload.size(), true);
            break;
        case MSG_LOCAL_LOAD:
            on_local_load(fd, f.payload.data(), f.payload.size());
            break;
        case MSG_SETUP_BEGIN:
            on_setup_begin(fd, f.payload.data(), f.payload.size());
            break;
        case MSG_UPLOAD:
            on_upload(fd, f.payload.data(), f.payload.size());
            break;
        case MSG_READY:
            on_ready(fd, f.payload.data(), f.payload.size());
            break;
        case MSG_INFER:
            counters->infer_in_bytes += kHeaderBytes + f.payload.size();
            on_infer(fd, f.payload.data(), f.payload.size(), timing->enabled() ? mono_ns() : 0);
            break;
        case MSG_CACHE_STATUS:
            on_cache_status(fd);
            break;
        case MSG_UNLOAD:
            void_session();
            unload_weights("explicit");
            ++counters->cache_unloads;
            send_frame(fd, MSG_UNLOAD_ACK, ST_OK, {});
            break;
        case MSG_STATUS:
            on_status(fd);
            break;
        case MSG_SHUTDOWN:
            send_frame(fd, MSG_STATUS_ACK, ST_OK, {});
            sock::close_fd(fd);
            stopping.store(true, std::memory_order_release);
            void_session();
            return; // run() sees stopping and returns
        default:
            fail(ST_PROTOCOL);
            send_frame(fd, MSG_STATUS_ACK, ST_PROTOCOL, {});
            break;
        }
        // Inference receive storage survives the next request; large setup/upload staging does not.
        if (f.payload.capacity() > kInferPayloadMax)
            std::vector<uint8_t>().swap(f.payload);
    }
    sock::close_fd(fd);
    expire_resident();
    void_session();   // a disconnect voids the session: the next connection restarts with SETUP_BEGIN
}

ExpertWorkerService::ExpertWorkerService(WorkerOptions options)
    : impl_(std::make_unique<Impl>(options, &counters_, &rpc_timing_)) {
    rpc_timing_.set_enabled(options.rpc_timing);
}
ExpertWorkerService::~ExpertWorkerService() {
    stop();
    sock::close_fd(impl_->listener);
}

bool ExpertWorkerService::listen(uint16_t& port, std::string& err) {
    if (!sock::listener_policy(impl_->opt.bind_ip, impl_->opt.allow_peer_ips, impl_->allowed_peers, err))
        return false;
    if ((impl_->opt.bind_ip != "127.0.0.1" || !impl_->opt.allow_peer_ips.empty()) && impl_->opt.deadline_ms <= 0) {
        err = "explicit bind/peer policy requires a positive --deadline-ms";
        return false;
    }
    if (impl_->opt.residency_ms < 0 || impl_->opt.residency_ms > kMaxResidencyMs ||
        (impl_->opt.residency_forever && impl_->opt.residency_ms != 0) ||
        ((impl_->opt.residency_ms || impl_->opt.residency_forever) && impl_->opt.deadline_ms <= 0)) {
        err = "residency must be 0..86400000 ms or explicit forever, with a positive I/O deadline";
        return false;
    }
    if (impl_->opt.upload_chunk_max == 0 || impl_->opt.upload_chunk_max > kUploadChunkMax) {
        err = "setup chunk cap must be between 1 and " + std::to_string(kUploadChunkMax);
        return false;
    }
    if (impl_->opt.local_root.empty() != impl_->opt.local_manifest.empty() ||
        impl_->opt.local_setup_ms <= 0) {
        err = "local root and manifest must both be configured, with a positive setup budget";
        return false;
    }
    if (!impl_->opt.local_root.empty()) {
        if (impl_->opt.deadline_ms <= 0) {
            err = "local setup needs a positive per-batch deadline";
            return false;
        }
        impl_->local = std::make_unique<LocalSource>();
        if (!impl_->local->open(impl_->opt.local_root, impl_->opt.local_manifest, err))
            return false;
    }
    return impl_->opt.bind_ip == "127.0.0.1" && impl_->allowed_peers.empty()
        ? sock::listen_loopback(port, impl_->listener, err)
        : sock::listen_ipv4(impl_->opt.bind_ip, port, impl_->listener, err);
}

void ExpertWorkerService::run() {
    while (!impl_->stopping.load(std::memory_order_acquire)) {
        impl_->expire_resident();
        sock_fd fd = kBadFd;
        if (impl_->listener == kBadFd) break;
        const bool accepted = impl_->allowed_peers.empty()
            ? sock::accept_one(impl_->listener, fd, 100, impl_->opt.accepted_tcp_nodelay)
            : sock::accept_allowed(impl_->listener, fd, 100, impl_->opt.accepted_tcp_nodelay, impl_->allowed_peers);
        if (!accepted) continue;
        impl_->serve_connection(fd);
    }
    impl_->void_session();
    impl_->unload_weights("shutdown");
    sock::close_fd(impl_->listener);
    impl_->logf("shutdown: arena released; joining CPU pool");
    impl_->pool.reset();
    impl_->logf("shutdown: CPU pool joined; resident=0 B");
}

void ExpertWorkerService::stop() {
    impl_->stopping.store(true, std::memory_order_release);
    // accept polls every 100 ms; controlled active I/O observes stop every 50 ms.
    // Only run() closes sockets, so a stop cannot race descriptor reuse.
}

}  // namespace strata::worker
