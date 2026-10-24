// tests/worker/expert_worker_test.cpp - inferswarm #348: the resident CPU-expert worker's focused checks.
//
// CPU-only, no GPU and no model file: Q8_0 packed fixtures are generated in-process with ggml's own
// quantizer, the service runs on a real loopback socket, and the transported results are compared BIT
// FOR BIT with ExpertWorkerService::direct_native_partial() - the equivalent direct native CPU call under
// the same-kernel/ordered-reduction contract in expert_wire.hpp.  The rest of the file is the rejection
// battery: setup, ownership, digest, session, sequence, nonfinite, restart, disconnect, interrupted
// upload, malformed frames and deadlines.
#include "strata/worker/expert_client.hpp"
#include "strata/worker/expert_worker.hpp"
#include "strata/worker/expert_wire.hpp"

#include "wire_sock.hpp"

#include "ggml.h"

#include <algorithm>
#include <array>
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

namespace strata::worker {
// Capacity inspection only, on the synchronous client's owning thread. No allocation instrumentation.
struct RpcWorkspaceTestAccess {
    static std::array<size_t, 3> capacities(const ExpertWorkerClient& c) {
        return {c.infer_payload_.capacity(), c.infer_frame_.capacity(), c.infer_reply_.payload.capacity()};
    }
};
}

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}

using strata::worker::Digest128;
using strata::worker::InferReply;
using strata::worker::InferRequest;
using strata::worker::SetupManifest;
using strata::worker::Status;
using strata::kernels::cpu::NativeFmt;

constexpr uint16_t kGu = 8, kD = 8;   // GGML_TYPE_Q8_0
constexpr uint16_t kLayers = 3, kExperts = 16, kEmbd = 2560, kFf = 640;

uint64_t mono_ms() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ---- deterministic Q8_0 packed fixtures (an intended native format, not a float fake) -------------

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 1u) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return (float) ((int32_t) (s >> 9)) / (float) (1 << 23) - 1.f;   // [-1, 1)
    }
    void fill(std::vector<float>& v) {
        for (size_t i = 0; i < v.size(); ++i) v[i] = next();
    }
};

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

std::pair<SetupManifest, BlobMap> make_world(uint64_t model_id, int delay_ms = 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    NativeFmt f;
    std::string err;
    check(strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, f, err), "Q8_0/Q8_0 is a native layout");
    SetupManifest m;
    m.nonce = 0x1234abcd;
    m.model_id = model_id;
    std::snprintf(m.model_name, sizeof m.model_name, "fixture-model");
    std::snprintf(m.quant_name, sizeof m.quant_name, "Q8_0");
    m.gu_type = kGu;
    m.d_type = kD;
    m.n_layers = kLayers;
    m.n_expert = kExperts;
    m.n_embd = kEmbd;
    m.n_ff = kFf;
    const size_t stride = ((size_t) kExperts + 7) / 8;
    m.partition.assign((size_t) kLayers * stride, 0);
    const std::pair<int, int> owned_list[] = {{0, 2}, {0, 5}, {0, 9}, {1, 1}, {1, 2}, {1, 3}, {2, 7}};
    BlobMap blobs;
    uint32_t seed = 7;
    for (const auto& [l, e] : owned_list) {
        m.partition[(size_t) l * stride + (size_t) e / 8] |= (uint8_t) (1u << (e % 8));
        blobs[{l, e}] = make_blob(seed++, f);
    }
    m.partition_id = strata::worker::blob_digest(m.partition.data(), m.partition.size()).a;
    return {m, blobs};
}

// ---- the service harness ------------------------------------------------------------------------

struct Svc {
    strata::worker::WorkerOptions opt;
    std::unique_ptr<strata::worker::ExpertWorkerService> svc;
    uint16_t port = 0;
    std::thread th;
    explicit Svc(strata::worker::WorkerOptions o = {}) : opt(o) {
        if (opt.workers == 0) opt.workers = 2;
        svc = std::make_unique<strata::worker::ExpertWorkerService>(opt);
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

// ---- a raw frame client, for the wire-level controls ----------------------------------------------

struct Raw {
    strata::worker::sock_fd fd = strata::worker::kBadFd;
    bool conn(uint16_t port) {
        std::string err;
        return strata::worker::sock::connect_host("127.0.0.1", port, 2000, fd, err);
    }
    bool send(uint16_t msg, uint16_t status, const void* payload, size_t n) {
        std::vector<uint8_t> buf;
        strata::worker::encode_frame(buf, msg, status, payload, n);
        return strata::worker::sock::send_all(fd, buf.data(), buf.size(), 2000);
    }
    bool recv_reply(uint16_t& msg, uint16_t& status, std::vector<uint8_t>& payload, int deadline = 2000) {
        payload.clear();
        uint8_t head[16];
        if (!strata::worker::sock::recv_all(fd, head, 16, deadline)) return false;
        if (head[0] != 'S' || head[1] != 'E' || head[2] != 'W' || head[3] != '1') return false;
        msg = (uint16_t) (head[4] | head[5] << 8);
        status = (uint16_t) (head[6] | head[7] << 8);
        uint64_t len = 0;
        for (int b = 0; b < 8; ++b) len |= (uint64_t) head[8 + b] << (8 * b);
        if (len > strata::worker::kMaxFrameBytes) return false;
        payload.resize((size_t) len);
        return len == 0 || strata::worker::sock::recv_all(fd, payload.data(), payload.size(), deadline);
    }
    ~Raw() { strata::worker::sock::close_fd(fd); }
};

// an UPLOAD frame: session | layer | expert | total | off | len | digest(16) | bytes
std::vector<uint8_t> upload_frame(uint64_t sid, int layer, int expert, uint32_t total, uint32_t off,
                                  uint32_t len, const Digest128& d, const uint8_t* data) {
    std::vector<uint8_t> up(40 + len);   // 8 session + 2 layer + 2 expert + 4 total + 4 off + 4 len + 16 digest
    auto p64 = [&](size_t at, uint64_t v) { for (int b = 0; b < 8; ++b) up[at + b] = (uint8_t) (v >> (8 * b)); };
    auto p32 = [&](size_t at, uint32_t v) { for (int b = 0; b < 4; ++b) up[at + b] = (uint8_t) (v >> (8 * b)); };
    p64(0, sid);
    up[8] = (uint8_t) layer;
    up[9] = (uint8_t) (layer >> 8);
    up[10] = (uint8_t) expert;
    up[11] = (uint8_t) (expert >> 8);
    p32(12, total);
    p32(16, off);
    p32(20, len);
    p64(24, d.a);
    p64(32, d.b);
    if (len) std::memcpy(up.data() + 40, data, len);
    return up;
}

bool payload_u64(const std::vector<uint8_t>& p, size_t at, uint64_t& v) {
    v = 0;
    if (at > p.size() || p.size() - at < 8) return false;
    for (int b = 0; b < 8; ++b) v |= (uint64_t) p[at + b] << (8 * b);
    return true;
}

bool setup_ack(Raw& raw, uint64_t& sid) {
    uint16_t msg = 0, st = 0;
    std::vector<uint8_t> payload;
    const bool ok = raw.recv_reply(msg, st, payload) && msg == strata::worker::MSG_SETUP_ACK &&
                    st == strata::worker::ST_OK && payload.size() == 20 && payload_u64(payload, 0, sid) && sid != 0;
    check(ok, "a complete successful SETUP_ACK carries a session");
    return ok;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Frozen scalar little-endian reference, independent of the production codec and its wrappers.
void reference_word(std::vector<uint8_t>& out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out.push_back((uint8_t) (value >> (i * 8)));
}
void reference_float(std::vector<uint8_t>& out, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    reference_word(out, bits, 4);
}
std::vector<uint8_t> reference_request(const InferRequest& r) {
    std::vector<uint8_t> out;
    for (auto value : {r.session, r.model_id, r.partition_id, r.sequence}) reference_word(out, value, 8);
    reference_word(out, r.layer, 2);
    out.push_back(r.T); out.push_back(r.K);
    for (auto value : r.mask) out.push_back(value != 0 ? 1 : 0);
    for (auto value : r.ids) reference_word(out, value, 2);
    for (auto value : r.weights) reference_float(out, value);
    for (auto value : r.activations) reference_float(out, value);
    return out;
}

void codec_workspace_reuse() {
    using namespace strata::worker;
    std::vector<uint8_t> payload, frame, ack;
    InferRequest decoded;
    InferReply reply, decoded_reply;
    std::array<size_t, 8> warm{};
    const int tokens[] = {8, 1, 2, 1};
    const int ranks[] = {10, 1, 3, 2};
    for (int pass = 0; pass < 3; ++pass) for (int step = 0; step < 4; ++step) {
        InferRequest r;
        r.session = 0x8877665544332211ull; r.model_id = 29; r.partition_id = 31;
        r.sequence = 100 + pass * 4 + step; r.layer = 1;
        r.T = tokens[step]; r.K = ranks[step];
        const size_t count = (size_t) r.T * r.K;
        r.mask.resize(count); r.ids.resize(count); r.weights.resize(count);
        for (size_t i = 0; i < count; ++i) {
            r.mask[i] = step == 1 ? 0 : (i % 3 ? 7 : 0);
            r.ids[i] = i % kExperts;
            r.weights[i] = i % 2 ? -0.f : 0.125f * (float) i;
        }
        r.activations.assign((size_t) r.T * kEmbd, (float) (step + pass) * -0.25f);
        check(encode_infer(r, kEmbd, payload), "reuse request encodes");
        const auto expected = reference_request(r);
        check(payload == expected, "reused request is byte-identical to the scalar reference");
        encode_frame(frame, MSG_INFER, ST_OK, payload.data(), payload.size());
        std::vector<uint8_t> expected_frame{'S', 'E', 'W', '1', MSG_INFER, 0, 0, 0};
        reference_word(expected_frame, expected.size(), 8);
        expected_frame.insert(expected_frame.end(), expected.begin(), expected.end());
        check(frame == expected_frame, "reused contiguous frame preserves exact bytes");
        check(decode_infer(payload.data(), payload.size(), kLayers, kExperts, kEmbd, decoded),
              "reuse request decodes");
        check(reference_request(decoded) == expected, "reused decoded fields contain no stale tail");
        reply.session = r.session; reply.sequence = r.sequence;
        reply.partials.assign((size_t) r.T * kEmbd, step == 1 ? 0.f : -0.375f);
        encode_infer_ack(reply, ack);
        std::vector<uint8_t> expected_ack;
        reference_word(expected_ack, reply.session, 8); reference_word(expected_ack, reply.sequence, 8);
        for (float value : reply.partials) reference_float(expected_ack, value);
        check(ack == expected_ack && encode_infer_ack(reply) == expected_ack,
              "reused reply and compatibility wrapper preserve scalar reference bytes");
        check(decode_infer_ack(ack.data(), ack.size(), kEmbd, r.T, decoded_reply) &&
                  same_bits(reply.partials, decoded_reply.partials), "reused reply has exact bits and shape");
        const std::array<size_t, 8> capacities{payload.capacity(), frame.capacity(), ack.capacity(),
            decoded.mask.capacity(), decoded.ids.capacity(), decoded.weights.capacity(),
            decoded.activations.capacity(), decoded_reply.partials.capacity()};
        if (pass == 0 && step == 0) warm = capacities;
        check(capacities == warm, "codec capacities remain at the warm high-water mark");
        check(!decode_infer(payload.data(), payload.size() - 1, kLayers, kExperts, kEmbd, decoded),
              "truncated request is refused between valid reused decodes");
    }
    InferRequest small;
    small.T = small.K = 1; small.mask = {1}; small.ids = {2}; small.weights = {1.f};
    small.activations.assign(kEmbd, 0.f);
    std::vector<uint8_t> small_payload;
    check(encode_infer(small, kEmbd, small_payload) && small_payload.capacity() < kInferPayloadMax,
          "a fresh T1 encoder does not reserve the T8 maximum");
}

// one request, transported and direct, compared bitwise
void parity_case(strata::worker::ExpertWorkerClient& cl, const SetupManifest& m, const BlobMap& blobs,
                 const NativeFmt& f, uint64_t seq, uint16_t layer, int T, int K, const int* ids_in,
                 const int* mask_in, const float* w_in, Rng& act_rng, const char* what) {
    InferRequest r;
    r.session = cl.session();
    r.model_id = m.model_id;
    r.partition_id = m.partition_id;
    r.sequence = seq;
    r.layer = layer;
    r.T = (uint8_t) T;
    r.K = (uint8_t) K;
    const size_t n = (size_t) T * K;
    r.mask.resize(n);
    for (size_t i = 0; i < n; ++i) r.mask[i] = mask_in[i] != 0 ? 1 : 0;
    r.ids.resize(n);
    r.weights.assign(w_in, w_in + n);
    for (size_t i = 0; i < n; ++i) r.ids[i] = (uint16_t) ids_in[i];
    r.activations.resize((size_t) T * kEmbd);
    act_rng.fill(r.activations);

    InferReply got;
    std::string err;
    check(cl.infer(r, got, err),
          (std::string(what) + ": transported request served (" + (err.empty() ? "ok" : err) + ")").c_str());
    check(got.sequence == seq && got.session == cl.session(), (std::string(what) + ": reply identity").c_str());

    std::vector<const uint8_t*> per_entry(n, nullptr);
    for (size_t i = 0; i < n; ++i)
        if (r.mask[i]) {
            auto it = blobs.find({layer, (int) r.ids[i]});
            if (it == blobs.end()) {
                check(false, (std::string(what) + ": owned blob exists").c_str());
                return;   // no reference to compute against: nothing to compare
            }
            per_entry[i] = it->second.data();
        }
    std::vector<float> want;
    check(strata::worker::ExpertWorkerService::direct_native_partial(f, T, K, r.mask.data(),
                                                                     r.weights.data(),
                                                                     r.activations.data(), per_entry.data(),
                                                                     want, err),
          (std::string(what) + ": reference computed").c_str());
    check(same_bits(got.partials, want), (std::string(what) + ": bitwise equality").c_str());
}

#include "setup_chunk_checks.hpp"

// ---- sections ------------------------------------------------------------------------------------

void codec_tests() {
    uint64_t value = 99;
    for (size_t n = 0; n < 8; ++n)
        check(!payload_u64(std::vector<uint8_t>(n), 0, value), "a short raw reply never decodes a session");
    check(!payload_u64({}, std::numeric_limits<size_t>::max(), value), "an overflowing raw offset is refused");
    check(payload_u64(std::vector<uint8_t>(8, 1), 0, value) && value == 0x0101010101010101ull,
          "a complete raw session decodes");
    check(strata::worker::kInferFrameMax < 96u * 1024u, "the inference frame cap is under 96 KiB");
    check(strata::worker::kInferPayloadMax == strata::worker::kInferFixed + (size_t) 8 * 10 +
                                                  (size_t) 8 * 10 * 2 + (size_t) 8 * 10 * 4 +
                                                  (size_t) 8 * 2560 * 4,
          "the inference cap is the schema's own arithmetic");
    auto [m, blobs] = make_world(1);
    std::vector<uint8_t> wire;
    check(strata::worker::encode_setup_begin(m, wire), "manifest encodes");
    SetupManifest back;
    check(strata::worker::decode_setup_begin(wire.data(), wire.size(), back), "manifest decodes");
    check(back.n_layers == kLayers && back.n_embd == kEmbd && back.gu_type == kGu && back.owns(0, 2) &&
              !back.owns(0, 3) && back.owned_count() == blobs.size(),
          "the roundtrip keeps geometry and partition");
    check(!strata::worker::decode_setup_begin(wire.data(), wire.size() - 1, back), "a cut manifest is refused");
    SetupManifest big = m;
    big.n_embd = 4096;
    check(!strata::worker::encode_setup_begin(big, wire), "geometry past the compiled bound does not encode");

    InferRequest r;
    r.T = 1;
    r.K = 3;
    r.mask = {1, 1, 0};
    r.ids = {2, 9, 11};
    r.weights = {0.5f, 0.3f, 0.2f};
    r.activations.resize(kEmbd);
    check(strata::worker::encode_infer(r, kEmbd, wire) && wire.size() <= strata::worker::kInferPayloadMax,
          "an inference request encodes inside the cap");
    InferRequest rb;
    check(strata::worker::decode_infer(wire.data(), wire.size(), kLayers, kExperts, kEmbd, rb),
          "an inference request decodes");
    check(rb.T == 1 && rb.K == 3 && rb.ids[1] == 9 && rb.mask[2] == 0, "the roundtrip keeps the request");
    check(!strata::worker::decode_infer(wire.data(), wire.size(), kLayers, 5, kEmbd, rb),
          "an id past n_expert is refused at decode");
    InferReply rep;
    rep.session = 7;
    rep.sequence = 9;
    rep.partials.assign(kEmbd, 0.25f);
    const std::vector<uint8_t> ack = strata::worker::encode_infer_ack(rep);
    InferReply rep2;
    check(strata::worker::decode_infer_ack(ack.data(), ack.size(), kEmbd, 1, rep2) && rep2.session == 7 &&
              rep2.partials[0] == 0.25f,
          "the reply roundtrips");
    r.T = 9;
    check(!strata::worker::encode_infer(r, kEmbd, wire), "T past the bound does not encode");
    r.T = 1;
    r.K = 11;
    check(!strata::worker::encode_infer(r, kEmbd, wire), "K past the bound does not encode");

    // an inference frame cannot carry weight bytes: the whole frame is smaller than one expert blob
    // and its schema-derived cap leaves room only for identity, mask, ids, weights and activations
    check(wire.size() < blobs.at({0, 2}).size(), "an inference frame is smaller than one expert blob");

    Digest128 d1 = strata::worker::blob_digest(blobs.at({0, 2}).data(), blobs.at({0, 2}).size());
    Digest128 d2 = strata::worker::blob_digest(blobs.at({0, 2}).data(), blobs.at({0, 2}).size());
    Digest128 d3 = strata::worker::blob_digest(blobs.at({0, 2}).data(), blobs.at({0, 2}).size() - 1);
    Digest128 d4 = strata::worker::blob_digest(blobs.at({0, 5}).data(), blobs.at({0, 5}).size());
    check(d1 == d2, "the digest is stable");
    check(d1 != d3, "a truncated blob digests differently");
    check(d1 != d4, "another blob digests differently");
}

void happy_path_and_parity() {
    Svc svc(strata::worker::WorkerOptions{.memory_budget = 64ull << 20, .workers = 2, .deadline_ms = 4000});
    check(svc.start(), "the service listens on loopback");
    auto [m, blobs] = make_world(42);
    NativeFmt f;
    std::string nerr;
    strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, f, nerr);

    strata::worker::ExpertWorkerClient cl;
    std::string err;
    check(cl.connect_host("127.0.0.1", svc.port, 4000, err), "the client connects");
    check(cl.setup(m, err) && cl.session() != 0, "setup starts a session");
    check(cl.blob_bytes() == f.bytes, "the ack reports the layout's blob size");
    for (const auto& [key, blob] : blobs)
        check(cl.upload(key.first, key.second, blob.data(), (uint32_t) blob.size(), err),
              "an owned blob uploads in bounded chunks");
    check(cl.mark_ready(err), "readiness after the full assigned set");
    Rng rng(991);

    // T=1, mixed ownership: entry 2 is GPU-side (mask 0) and must not be computed here
    {
        const int ids[] = {2, 9, 11};
        const int mask[] = {1, 1, 0};
        const float w[] = {0.5f, 0.3f, 0.2f};
        parity_case(cl, m, blobs, f, 100, 0, 1, 3, ids, mask, w, rng, "T=1 mixed ownership");
    }
    // T=4 with a repeated expert id for token 0 and one GPU-side entry per token
    {
        const int ids[] = {2, 2, 5, 11, 9, 2, 5, 11, 2, 9, 5, 11, 5, 2, 9, 11};
        const int mask[] = {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};
        const float w[16] = {0.4f, 0.2f, 0.1f, 0.3f, 0.25f, 0.25f, 0.1f, 0.4f,
                             0.3f, 0.2f, 0.1f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};
        parity_case(cl, m, blobs, f, 101, 0, 4, 4, ids, mask, w, rng, "T=4 repeated ids");
    }
    // T=8 (the compiled bound) with K=10, layer 1: only experts 1, 2, 3 are CPU-owned.  Unmasked
    // entries still carry valid ids - a frame always carries the router's full top-K.
    {
        std::vector<int> ids, mask;
        std::vector<float> w;
        Rng kr(5);
        for (int t = 0; t < 8; ++t)
            for (int j = 0; j < 10; ++j) {
                const int e = 4 + (int) ((kr.next() + 1.f) * 4.f);   // 4..11: none of them owned
                ids.push_back(e);
                mask.push_back(0);
                w.push_back(0.05f);
            }
        // route a few entries to the owned experts, including a duplicate for token 3
        ids[0] = 1; mask[0] = 1;
        ids[3] = 3; mask[3] = 1;
        ids[8] = 2; mask[8] = 1;
        ids[13] = 2; mask[13] = 1;
        ids[20] = 1; mask[20] = 1;
        ids[79] = 3; mask[79] = 1;
        parity_case(cl, m, blobs, f, 102, 1, 8, 10, ids.data(), mask.data(), w.data(), rng,
                    "T=8 boundary mixed");
    }
    // zero CPU work: every entry GPU-side; the reply is zeros, and the reference agrees
    {
        const int ids[] = {4, 6};
        const int mask[] = {0, 0};
        const float w[] = {0.5f, 0.5f};
        parity_case(cl, m, blobs, f, 103, 0, 1, 2, ids, mask, w, rng, "zero CPU work");
    }
    // a wrong-token control: two tokens with clearly different activations - a reply whose token 0 was
    // computed from token 1's activation (the transposition a wrong mapping would produce) is caught
    {
        const int ids[] = {2, 5};
        const int mask[] = {1, 1};
        const float w[] = {0.5f, 0.5f};
        InferRequest r;
        r.session = cl.session();
        r.model_id = m.model_id;
        r.partition_id = m.partition_id;
        r.sequence = 104;
        r.layer = 0;
        r.T = 2;
        r.K = 1;
        r.mask.assign(mask, mask + 2);
        r.ids.assign(ids, ids + 2);
        r.weights.assign(w, w + 2);
        r.activations.resize((size_t) 2 * kEmbd);
        for (size_t i = 0; i < r.activations.size(); ++i)
            r.activations[i] = (i < kEmbd ? 0.25f : -0.75f) * (1.f + 0.01f * (float) (i % 97));
        InferReply got;
        std::string err;
        check(cl.infer(r, got, err), "the two-token request serves");
        const uint8_t* blobs2[2] = {blobs.at({0, 2}).data(), blobs.at({0, 5}).data()};
        std::vector<float> want;
        check(strata::worker::ExpertWorkerService::direct_native_partial(f, 2, 1, r.mask.data(),
                                                                         r.weights.data(),
                                                                         r.activations.data(), blobs2, want,
                                                                         err),
              "the two-token reference computes");
        const float* got0 = got.partials.data();
        const float* want0 = want.data();
        const float* want1 = want.data() + kEmbd;
        check(std::memcmp(got0, want0, kEmbd * sizeof(float)) == 0, "token 0's rows are token 0's");
        check(std::memcmp(got0, want1, kEmbd * sizeof(float)) != 0,
              "token 0's rows are not token 1's (a transposed reply is caught)");
        check(std::memcmp(got.partials.data() + kEmbd, want1, kEmbd * sizeof(float)) == 0,
              "token 1's rows are token 1's");
    }
    // consecutive changed activations: the second answer must be the new one, not the cached first
    {
        const int ids[] = {7};
        const int mask[] = {1};
        const float w[] = {1.f};
        Rng a(1234), b(4321);
        parity_case(cl, m, blobs, f, 105, 2, 1, 1, ids, mask, w, a, "changed activations (a)");
        parity_case(cl, m, blobs, f, 106, 2, 1, 1, ids, mask, w, b, "changed activations (b)");
    }

    strata::worker::WorkerStatusInfo st;
    check(cl.status(st, err) && st.ready == 1 && st.requests >= 6 && st.setup_weight_bytes > 0 &&
              st.infer_in_bytes > 0 && st.infer_out_bytes > 0 && st.cpu_service_ns > 0,
          "the counters saw the whole session");
    cl.close();
}

void inference_workspace_reuse() {
    using namespace strata::worker;
    Svc svc(WorkerOptions{.memory_budget = 64ull << 20, .workers = 2, .deadline_ms = 4000});
    if (!svc.start()) { check(false, "reuse service starts"); return; }
    auto [m, blobs] = make_world(818);
    NativeFmt fmt;
    std::string err;
    check(strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, fmt, err), "reuse native format");
    ExpertWorkerClient cl;
    if (!cl.connect_host("127.0.0.1", svc.port, 4000, err) || !cl.setup(m, err)) {
        check(false, "reuse setup"); return;
    }
    for (const auto& [key, blob] : blobs)
        check(cl.upload(key.first, key.second, blob.data(), (uint32_t) blob.size(), err), "reuse upload");
    check(cl.mark_ready(err), "reuse ready");
    std::array<size_t, 3> warm{};
    const int tokens[] = {8, 1, 2, 1}, ranks[] = {3, 10, 2, 1};
    uint64_t sequence = 1;
    InferReply got;
    size_t reply_capacity = 0;
    for (int pass = 0; pass < 3; ++pass) for (int step = 0; step < 4; ++step) {
        InferRequest r;
        r.session = cl.session(); r.model_id = m.model_id; r.partition_id = m.partition_id;
        r.sequence = sequence++; r.layer = 0; r.T = tokens[step]; r.K = ranks[step];
        const size_t count = (size_t) r.T * r.K;
        r.mask.assign(count, 0); r.ids.assign(count, 11); r.weights.resize(count);
        const uint16_t owned[] = {2, 5, 9};
        std::vector<const uint8_t*> per_entry(count, nullptr);
        for (size_t i = 0; i < count; ++i) {
            r.weights[i] = (i % 2 ? -0.25f : 0.5f) + pass * 0.125f;
            if (step == 0 || step == 3 || (step == 2 && i == 3)) {
                r.mask[i] = 1; r.ids[i] = owned[i % 3];
                per_entry[i] = blobs.at({0, r.ids[i]}).data();
            }
        }
        r.activations.resize((size_t) r.T * kEmbd);
        Rng rng(41 + pass * 4 + step); rng.fill(r.activations);
        if (step == 2) {
            auto refused = r;
            refused.T = 1; refused.K = 10;
            refused.mask.assign(10, 1); refused.ids.assign(10, 2); refused.weights.assign(10, 1.f);
            refused.activations.resize(kEmbd);
            check(!cl.infer(refused, got, err) && cl.last_status() == ST_BAD_REQUEST,
                  "too many expert uses is refused after warm dense and zero work");
            // A refused request must not consume this sequence or leave stale validation scratch.
        }
        check(cl.infer(r, got, err), "alternating reuse request serves");
        std::vector<float> want;
        check(ExpertWorkerService::direct_native_partial(fmt, r.T, r.K, r.mask.data(), r.weights.data(),
                  r.activations.data(), per_entry.data(), want, err), "alternating reuse reference");
        check(got.sequence == r.sequence && same_bits(got.partials, want),
              "T8/T1/T2/T1 dense/zero/sparse partials are bitwise the fresh reference");
        const auto capacities = RpcWorkspaceTestAccess::capacities(cl);
        if (pass == 0 && step == 0) { warm = capacities; reply_capacity = got.partials.capacity(); }
        check(capacities == warm && got.partials.capacity() == reply_capacity,
              "client payload/frame/receive and caller reply retain warm capacities");
    }
    // New setup on the same connection, then reconnect to another geometry. Neither may use old rows.
    for (int embd : {128, 256}) {
        if (embd == 256) {
            cl.close();
            check(RpcWorkspaceTestAccess::capacities(cl) == std::array<size_t, 3>{},
                  "close releases inference buffers before reconnect");
            check(cl.connect_host("127.0.0.1", svc.port, 4000, err), "reuse geometry reconnect");
        }
        SetupManifest changed = m;
        changed.model_id += embd; changed.n_embd = embd; changed.n_ff = 64;
        changed.n_layers = 1; changed.n_expert = 4; changed.partition = {2};
        NativeFmt small;
        check(strata::kernels::cpu::native_fmt(kGu, kD, embd, 64, small, err), "changed geometry format");
        const auto blob = make_blob(embd, small);
        check(cl.setup(changed, err) && cl.upload(0, 1, blob.data(), (uint32_t) blob.size(), err) &&
                  cl.mark_ready(err), "changed geometry setup and ready");
        InferRequest r;
        r.session = cl.session(); r.model_id = changed.model_id; r.partition_id = changed.partition_id;
        r.sequence = 1; r.T = 1; r.K = 1; r.mask = {1}; r.ids = {1}; r.weights = {-0.5f};
        r.activations.assign(embd, -0.125f);
        const uint8_t* ptr = blob.data();
        std::vector<float> want;
        check(cl.infer(r, got, err) && ExpertWorkerService::direct_native_partial(small, 1, 1,
                  r.mask.data(), r.weights.data(), r.activations.data(), &ptr, want, err) &&
                  same_bits(got.partials, want), "geometry changes produce exact fresh-sized partials");
    }
    check(cl.unload(err), "reuse session unloads normally");
    cl.close();
}

void inference_rejections() {
    Svc svc(strata::worker::WorkerOptions{.memory_budget = 64ull << 20, .workers = 2, .deadline_ms = 4000});
    check(svc.start(), "the rejection service listens");
    auto [m, blobs] = make_world(7);
    NativeFmt f;
    std::string nerr;
    strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, f, nerr);
    strata::worker::ExpertWorkerClient cl;
    std::string err;
    check(cl.connect_host("127.0.0.1", svc.port, 4000, err) && cl.setup(m, err), "setup");
    for (const auto& [key, blob] : blobs) check(cl.upload(key.first, key.second, blob.data(), (uint32_t) blob.size(), err), "upload");
    check(cl.mark_ready(err), "ready");

    uint64_t seq = 200;
    auto expect_refusal = [&](InferRequest& r, Status want, const char* what) {
        InferReply rep;
        std::string e;
        // the client's encode step must not reject a structurally valid but semantically wrong frame
        check(!cl.infer(r, rep, e) && cl.last_status() == want,
              (std::string(what) + " is refused with its typed status").c_str());
    };
    InferRequest base;
    base.session = cl.session();
    base.model_id = m.model_id;
    base.partition_id = m.partition_id;
    base.sequence = seq++;
    base.layer = 0;
    base.T = 1;
    base.K = 2;
    base.mask = {1, 0};
    base.ids = {2, 11};
    base.weights = {0.6f, 0.4f};
    base.activations.assign(kEmbd, 0.1f);

    InferRequest r = base;
    r.sequence = seq++;
    {
        InferReply rep;
        check(cl.infer(r, rep, err), "a valid request still passes first");
    }
    // stale session
    r = base;
    r.sequence = seq++;
    r.session = cl.session() ^ 1;
    expect_refusal(r, strata::worker::ST_STALE_SESSION, "a wrong session");
    // wrong model identity
    r = base;
    r.sequence = seq++;
    r.model_id = 123456;
    expect_refusal(r, strata::worker::ST_STALE_SESSION, "a wrong model identity");
    // wrong partition identity
    r = base;
    r.sequence = seq++;
    r.partition_id = 99;
    expect_refusal(r, strata::worker::ST_STALE_SESSION, "a wrong partition identity");
    // replayed sequence
    r = base;
    r.sequence = base.sequence;
    expect_refusal(r, strata::worker::ST_BAD_REQUEST, "a replayed sequence");
    // an older sequence
    r = base;
    r.sequence = base.sequence - 1;
    expect_refusal(r, strata::worker::ST_BAD_REQUEST, "an older sequence");
    // an entry claims a slot the GPU side owns
    r = base;
    r.sequence = seq++;
    r.mask = {1, 1};
    r.ids = {2, 11};
    expect_refusal(r, strata::worker::ST_NOT_OWNED, "a masked entry the worker does not own");
    // an id past the model's expert count
    r = base;
    r.sequence = seq++;
    r.ids = {(uint16_t) (kExperts + 3), 11};
    expect_refusal(r, strata::worker::ST_BAD_REQUEST, "an expert id out of range");
    // a layer past the model's depth
    r = base;
    r.sequence = seq++;
    r.layer = kLayers;
    expect_refusal(r, strata::worker::ST_BAD_REQUEST, "a layer out of range");
    // nonfinite activation
    r = base;
    r.sequence = seq++;
    r.activations[10] = std::nanf("");
    expect_refusal(r, strata::worker::ST_NONFINITE, "a nonfinite activation");
    // nonfinite routing weight
    r = base;
    r.sequence = seq++;
    r.activations[10] = 0.1f;
    r.weights[0] = std::numeric_limits<float>::infinity();
    expect_refusal(r, strata::worker::ST_NONFINITE, "a nonfinite weight");
    // the typed rejections above must not have consumed the sequence: a fresh valid request still works
    InferRequest ok = base;
    ok.sequence = seq++;
    InferReply rep;
    check(cl.infer(ok, rep, err), "the session still serves after refusals");
    cl.close();
}

void setup_rejections() {
    Svc svc(strata::worker::WorkerOptions{.memory_budget = 8ull << 20, .workers = 1, .deadline_ms = 4000});
    check(svc.start(), "the setup-rejection service listens");
    auto [m, blobs] = make_world(5);
    NativeFmt f;
    std::string nerr;
    strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, f, nerr);
    strata::worker::ExpertWorkerClient cl;
    std::string err;
    check(cl.connect_host("127.0.0.1", svc.port, 4000, err), "connect");

    // over budget: 7 blobs of ~5 MiB against 8 MiB
    check(!cl.setup(m, err) && cl.last_status() == strata::worker::ST_OVER_BUDGET,
          "an over-budget assigned set is refused");
    // unsupported layout: F32 experts
    SetupManifest bad = m;
    bad.gu_type = 0;
    bad.d_type = 0;
    check(!cl.setup(bad, err) && cl.last_status() == strata::worker::ST_UNSUPPORTED_LAYOUT,
          "an unsupported layout is refused");
    // geometry past the compiled bound is refused at encode and decode alike
    SetupManifest huge = m;
    huge.n_embd = 4096;
    std::vector<uint8_t> wire;
    check(!strata::worker::encode_setup_begin(huge, wire), "geometry past the compiled bound does not encode");
    cl.close();

    // a big enough budget takes the same world in
    Svc big(strata::worker::WorkerOptions{.memory_budget = 64ull << 20, .workers = 2, .deadline_ms = 4000});
    check(big.start(), "the second service listens");
    check(cl.connect_host("127.0.0.1", big.port, 4000, err) && cl.setup(m, err), "setup on the big budget");
    // not-owned upload: slot (0,3) is outside the partition
    {
        const auto& blob = blobs.at({0, 2});
        check(!cl.upload(0, 3, blob.data(), (uint32_t) blob.size(), err) &&
                  cl.last_status() == strata::worker::ST_NOT_OWNED,
              "an upload outside the partition is refused");
    }
    // inference before readiness
    {
        InferRequest r;
        r.session = cl.session();
        r.model_id = m.model_id;
        r.partition_id = m.partition_id;
        r.sequence = 1;
        r.layer = 0;
        r.T = 1;
        r.K = 1;
        r.mask = {1};
        r.ids = {2};
        r.weights = {1.f};
        r.activations.assign(kEmbd, 0.f);
        InferReply rep;
        check(!cl.infer(r, rep, err) && cl.last_status() == strata::worker::ST_NOT_READY,
              "inference before readiness is refused");
    }
    cl.close();   // the service serves one connection at a time: free it for the raw frames below

    // ---- chunk-level controls, one raw session -----------------------------------------------
    Raw raw;
    check(raw.conn(big.port), "raw connect");
    uint16_t msg = 0, st = 0;
    std::vector<uint8_t> payload;
    if (!strata::worker::encode_setup_begin(m, wire) ||
        !raw.send(strata::worker::MSG_SETUP_BEGIN, 0, wire.data(), wire.size())) {
        check(false, "the raw setup sends"); return;
    }
    uint64_t sid = 0;
    if (!setup_ack(raw, sid)) return;
    auto upload_chunk = [&](int layer, int expert, uint32_t off, uint32_t len, const Digest128& d,
                            const uint8_t* data, Status want, const char* what) {
        const std::vector<uint8_t> up =
            upload_frame(sid, layer, expert, (uint32_t) f.bytes, off, len, d, data);
        if (!raw.send(strata::worker::MSG_UPLOAD, 0, up.data(), up.size())) { check(false, what); return; }
        check(raw.recv_reply(msg, st, payload) && st == want, what);
    };
    // a digest that does not match the bytes: the final chunk's verification refuses it
    {
        const std::vector<uint8_t>& blob = blobs.at({1, 2});
        std::vector<uint8_t> corrupt = blob;
        corrupt[12345] ^= 0xff;
        const Digest128 honest = strata::worker::blob_digest(blob.data(), blob.size());
        const uint32_t full = (uint32_t) blob.size(), chunk = 0x10000;
        uint32_t off = 0;
        for (; off + chunk < full; off += chunk)
            upload_chunk(1, 2, off, chunk, honest, corrupt.data() + off, strata::worker::ST_OK,
                         "the corrupted chunks land");
        upload_chunk(1, 2, off, full - off, honest, corrupt.data() + off,
                     strata::worker::ST_DIGEST_MISMATCH, "a mismatched digest is refused");
        // the bad slot is sticky: any further chunk for it names the same failure
        upload_chunk(1, 2, 0, chunk, honest, corrupt.data(), strata::worker::ST_DIGEST_MISMATCH,
                     "a failed slot is sticky");
    }
    // an out-of-order chunk: its offset is not the slot's fill point
    {
        const std::vector<uint8_t>& blob = blobs.at({2, 7});
        const Digest128 d = strata::worker::blob_digest(blob.data(), blob.size());
        upload_chunk(2, 7, 4096, 4096, d, blob.data() + 4096, strata::worker::ST_BAD_REQUEST,
                     "an out-of-order chunk is refused");
    }
    // a duplicate: a fully-uploaded slot cannot be uploaded again
    {
        const std::vector<uint8_t>& blob = blobs.at({0, 5});
        const Digest128 d = strata::worker::blob_digest(blob.data(), blob.size());
        const uint32_t full = (uint32_t) blob.size(), chunk = 0x10000;
        for (uint32_t off = 0; off < full; off += chunk)
            upload_chunk(0, 5, off, std::min(chunk, full - off), d, blob.data() + off,
                         strata::worker::ST_OK, "a clean slot uploads");
        upload_chunk(0, 5, 0, chunk, d, blob.data(), strata::worker::ST_DUPLICATE,
                     "a completed slot cannot be re-uploaded");
    }
    // readiness with slots still missing
    {
        std::vector<uint8_t> p2;
        for (int b = 0; b < 8; ++b) p2.push_back((uint8_t) (sid >> (8 * b)));
        check(raw.send(strata::worker::MSG_READY, 0, p2.data(), p2.size()) &&
                  raw.recv_reply(msg, st, payload) && st == strata::worker::ST_INCOMPLETE,
              "readiness is refused while slots are missing or bad");
    }
}

void restart_and_disconnect() {
    Svc svc(strata::worker::WorkerOptions{.memory_budget = 64ull << 20, .workers = 2, .deadline_ms = 4000});
    check(svc.start(), "the restart service listens");
    auto [m, blobs] = make_world(31);
    strata::worker::ExpertWorkerClient cl;
    std::string err;
    check(cl.connect_host("127.0.0.1", svc.port, 4000, err) && cl.setup(m, err), "first setup");
    const uint64_t old_session = cl.session();
    for (const auto& [key, blob] : blobs) cl.upload(key.first, key.second, blob.data(), (uint32_t) blob.size(), err);
    check(cl.mark_ready(err), "first ready");

    // a second SETUP_BEGIN restarts: a new session, and the old one cannot continue
    check(cl.setup(m, err) && cl.session() != old_session, "a restart mints a new session");
    InferRequest r;
    r.session = old_session;
    r.model_id = m.model_id;
    r.partition_id = m.partition_id;
    r.sequence = 1;
    r.layer = 0;
    r.T = 1;
    r.K = 1;
    r.mask = {1};
    r.ids = {2};
    r.weights = {1.f};
    r.activations.assign(kEmbd, 0.f);
    InferReply rep;
    check(!cl.infer(r, rep, err) && cl.last_status() == strata::worker::ST_STALE_SESSION,
          "the old session's requests cannot continue");
    cl.close();   // free the service for the raw connection below

    // an interrupted upload voids the session on disconnect
    {
        Raw raw;
        check(raw.conn(svc.port), "raw connect");
        std::vector<uint8_t> begin;
        strata::worker::encode_setup_begin(m, begin);
        raw.send(strata::worker::MSG_SETUP_BEGIN, 0, begin.data(), begin.size());
        uint16_t msg = 0, st = 0;
        std::vector<uint8_t> payload;
        uint64_t sid = 0;
        if (!setup_ack(raw, sid)) return;
        const std::vector<uint8_t>& blob = blobs.at({0, 2});
        const Digest128 d = strata::worker::blob_digest(blob.data(), blob.size());
        const std::vector<uint8_t> up = upload_frame(sid, 0, 2, (uint32_t) blob.size(), 0, 4096, d,
                                                     blob.data());
        check(raw.send(strata::worker::MSG_UPLOAD, 0, up.data(), up.size()), "one chunk lands");
        check(raw.recv_reply(msg, st, payload) && st == strata::worker::ST_OK, "the chunk is taken");
    }   // raw closes here: an interrupted upload
    // the void session refuses inference on the next connection (raw frame: a client without its own
    // session has no manifest yet, and the refusal must not depend on client-side state)
    {
        Raw raw2;
        check(raw2.conn(svc.port), "reconnect");
        InferRequest stale = r;
        std::vector<uint8_t> frame;
        check(strata::worker::encode_infer(stale, kEmbd, frame), "the stale frame encodes");
        check(raw2.send(strata::worker::MSG_INFER, 0, frame.data(), frame.size()), "the stale frame is sent");
        uint16_t msg = 0, st = 0;
        std::vector<uint8_t> payload;
        check(raw2.recv_reply(msg, st, payload) && st == strata::worker::ST_WRONG_STATE,
              "a voided session is not carried into the next connection");
    }
    strata::worker::ExpertWorkerClient cl2;
    InferReply rep2;
    check(cl2.connect_host("127.0.0.1", svc.port, 4000, err), "reconnect");
    // ...and a full re-setup on the new connection works end to end
    check(cl2.setup(m, err), "a fresh setup after the interruption");
    for (const auto& [key, blob] : blobs)
        check(cl2.upload(key.first, key.second, blob.data(), (uint32_t) blob.size(), err), "re-upload");
    check(cl2.mark_ready(err), "ready again");
    InferRequest fresh = r;
    fresh.session = cl2.session();
    fresh.sequence = 5;
    check(cl2.infer(fresh, rep2, err), "the restarted session serves");
    cl2.close();
}

void malformed_and_deadline() {
    {   // an empty INFER reaches the real handler, is refused, and remains attributable as T=0
        Svc empty(strata::worker::WorkerOptions{.memory_budget = 64ull << 20, .workers = 1,
                                                .deadline_ms = 250, .rpc_timing = true});
        check(empty.start(), "the empty-inference timing service listens");
        {
            Raw raw;
            check(raw.conn(empty.port), "raw connect for empty inference");
            uint16_t msg = 0, st = 0;
            std::vector<uint8_t> payload;
            check(raw.send(strata::worker::MSG_INFER, 0, nullptr, 0) &&
                      raw.recv_reply(msg, st, payload) && msg == strata::worker::MSG_INFER_ACK &&
                      st == strata::worker::ST_WRONG_STATE,
                  "an empty inference is refused in the empty state");
        }
        empty.stop(); // join before reading the service-owned diagnostic
        check(empty.svc->rpc_timing().calls() == 1 && empty.svc->rpc_timing().failures() == 1 &&
                  empty.svc->rpc_timing().token_shapes()[0] == 1,
              "worker timing attributes the real empty/refused request once");
    }
    Svc svc(strata::worker::WorkerOptions{.memory_budget = 64ull << 20, .workers = 1, .deadline_ms = 250});
    check(svc.start(), "the wire-abuse service listens");
    {   // bad magic: the connection closes with no reply
        Raw raw;
        check(raw.conn(svc.port), "raw connect");
        uint8_t junk[16] = {};
        junk[0] = 'X';
        check(strata::worker::sock::send_all(raw.fd, junk, 16, 1000), "junk header sent");
        uint8_t head[16];
        check(!strata::worker::sock::recv_all(raw.fd, head, 16, 3000), "a bad magic closes the connection");
    }
    {   // an oversized length claim is refused without reading it
        Raw raw;
        check(raw.conn(svc.port), "raw connect 2");
        uint8_t head[16] = {'S', 'E', 'W', '1', 0x07, 0, 0, 0};
        const uint64_t huge = 512ull * 1024 * 1024;
        for (int b = 0; b < 8; ++b) head[8 + b] = (uint8_t) (huge >> (8 * b));
        check(strata::worker::sock::send_all(raw.fd, head, 16, 1000), "huge length sent");
        uint8_t back[16];
        check(!strata::worker::sock::recv_all(raw.fd, back, 16, 3000), "an oversized frame closes");
    }
    {   // no header at all: the accepted connection's idle budget expires, then the next client works.
        Raw idle;
        check(idle.conn(svc.port), "idle client connects");
        const uint64_t start = mono_ms();
        uint8_t head[16];
        check(!strata::worker::sock::recv_all(idle.fd, head, sizeof head, 3000), "idle header deadline closes");
        const uint64_t elapsed = mono_ms() - start;
        check(elapsed >= 150 && elapsed < 2000, "idle close follows the 250 ms header budget");
    }
    {   // a header that promises more than it sends: the deadline closes it, no hang
        // Slow fixture preparation belongs outside an active connection's 250 ms idle budget.
        // This fixed 400 ms pause reproduces the slower-host scheduling pattern on fast machines too.
        auto [m, blobs] = make_world(3, 400);
        std::vector<uint8_t> begin;
        strata::worker::encode_setup_begin(m, begin);
        Raw raw;
        check(raw.conn(svc.port), "raw connect 3");
        check(raw.send(strata::worker::MSG_SETUP_BEGIN, 0, begin.data(), begin.size()), "setup sent");
        uint16_t msg = 0, st = 0;
        std::vector<uint8_t> payload;
        uint64_t sid = 0;
        if (!setup_ack(raw, sid)) return;
        InferRequest r;
        r.session = sid;
        r.model_id = m.model_id;
        r.partition_id = m.partition_id;
        r.sequence = 1;
        r.layer = 0;
        r.T = 1;
        r.K = 1;
        r.mask = {1};
        r.ids = {2};
        r.weights = {1.f};
        r.activations.assign(kEmbd, 0.05f);
        std::vector<uint8_t> wire;
        check(strata::worker::encode_infer(r, kEmbd, wire), "infer encodes");
        // a header that promises the whole frame, then only its first 100 payload bytes, then silence
        std::vector<uint8_t> cut(16 + 100);
        cut[0] = 'S';
        cut[1] = 'E';
        cut[2] = 'W';
        cut[3] = '1';
        cut[4] = (uint8_t) strata::worker::MSG_INFER;
        for (int b = 0; b < 8; ++b) cut[8 + b] = (uint8_t) (wire.size() >> (8 * b));
        std::memcpy(cut.data() + 16, wire.data(), 100);
        check(strata::worker::sock::send_all(raw.fd, cut.data(), cut.size(), 1000), "a cut frame's first bytes");
        // now silence: the worker's deadline must close the connection
        const uint64_t t0 = mono_ms();
        uint8_t head[16];
        check(!strata::worker::sock::recv_all(raw.fd, head, 16, 10000), "the deadline closes the frame");
        const uint64_t elapsed = mono_ms() - t0;
        check(elapsed >= 150 && elapsed < 2000, "the partial-payload close follows the 250 ms deadline");
        // A closed connection voids its session. A fresh connection must reject READY from that session.
        Raw next;
        if (!next.conn(svc.port)) { check(false, "reconnect after the payload deadline"); return; }
        uint8_t old_sid[8];
        for (int b = 0; b < 8; ++b) old_sid[b] = (uint8_t) (sid >> (8 * b));
        check(next.send(strata::worker::MSG_READY, 0, old_sid, sizeof old_sid) &&
              next.recv_reply(msg, st, payload) && msg == strata::worker::MSG_READY_ACK &&
              st == strata::worker::ST_WRONG_STATE, "the deadline invalidated the old session");
        check(next.send(strata::worker::MSG_SETUP_BEGIN, 0, begin.data(), begin.size()), "fresh setup sent");
        uint64_t fresh_sid = 0;
        if (!setup_ack(next, fresh_sid)) return;
        check(fresh_sid != sid, "reconnect setup creates a new generation");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--deadline-only") == 0) {
        malformed_and_deadline();
        return fails == 0 ? 0 : 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--setup-bench") == 0) {
        for (int delay_ms : {0, 1})
            for (uint32_t cap : {65536u, 512u*1024, 1024u*1024}) setup_cap_parity(cap, delay_ms, true);
        return fails == 0 ? 0 : 1;
    }
    setup_ack_controls();
    setup_upload_controls();
    for (uint32_t cap : {65536u, 512u*1024, 1024u*1024}) setup_cap_parity(cap, 0, false);
    if (argc == 2 && std::strcmp(argv[1], "--setup-only") == 0) return fails == 0 ? 0 : 1;
    codec_tests();
    codec_workspace_reuse();
    happy_path_and_parity();
    inference_workspace_reuse();
    inference_rejections();
    setup_rejections();
    restart_and_disconnect();
    malformed_and_deadline();
    if (fails == 0) std::printf("expert_worker_test: all checks passed\n");
    else std::printf("expert_worker_test: %d checks failed\n", fails);
    return fails == 0 ? 0 : 1;
}
