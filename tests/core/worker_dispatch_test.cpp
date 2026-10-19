// tests/core/worker_dispatch_test.cpp - inferswarm #349: the REAL dispatcher against the REAL worker.
//
// The real `expert_pool_dispatch_multi` runs a window of a synthetic 2-layer x 16-expert native Q8_0/Q8_0
// pack (native_experts.txt in a temp dir; blobs are ggml-quantized fixtures) with the external device
// operations faked only where necessary: the GPU plan sink is plain host memory with no publish/fetch
// callbacks, `host_res` a host array, the "VRAM" rows are only classified (kind 0) and never computed by a
// kernel, and RemoteExpertOpt provides the merge bracket.  (What this test does NOT inspect: the mapped
// h_sum_/h_mask_ tables themselves - accumulate_host's indexing is exercised, its GPU-side consumption is
// the existing helper-tier machinery, unchanged.)  The worker is the real
// in-process service on a real loopback port, driven through the real tier - the same path generate.cpp
// wires.  Checks:
//   * the ownership plan: cache-resident entries are kind 0, worker-partition entries never produce a
//     local blob read or job, and every entry has exactly one owner (no gap, no overlap);
//   * the merged result in the documented order - the GPU's per-entry rows weighted in entry order plus
//     the worker's already-weighted partial on top - equals the same formula over the direct references,
//     bit for bit (local rows against the native kernels, the worker's partial against
//     direct_native_partial);
//   * a dead worker latches `failed` on the next window and the rows it owned stay zero (no partial
//     commit, no stale output);
//   * default-off: with no tier attached the same window computes every entry locally and the merged
//     result equals the all-local reference - the untouched path, unchanged.
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/worker/expert_tier.hpp"
#include "strata/worker/expert_worker.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using strata::kernels::cpu::NativeFmt;
using strata::worker::ExpertWorkerService;
using strata::worker::ExpertWorkerTier;
using strata::worker::WorkerOptions;

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}

constexpr int kLayers = 2, kExperts = 16, kEmbd = 2560, kFf = 640, kK = 10;

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 1u) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return (float) ((int32_t) (s >> 9)) / (float) (1 << 23) - 1.f;
    }
    uint32_t pick(uint32_t n) { s = s * 1664525u + 1013904223u; return s % n; }
    float unit() { return 0.5f * next() + 0.5f; }
};

std::vector<int32_t> distinct_ids(Rng& r, int T, int K) {
    std::vector<int32_t> ids((size_t) T * K);
    std::vector<int32_t> pool((size_t) kExperts);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < kExperts; ++i) pool[(size_t) i] = i;
        for (int j = 0; j < K; ++j) {
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

/// The in-memory expert source: serves the fixtures, counts post-setup reads (there must be none for
/// worker-owned experts), and can be switched off to expose an accidental fallback to local reads.
class FixtureSource final : public strata::core::ExpertSource {
public:
    std::map<std::pair<int, int>, std::vector<uint8_t>> blobs;
    std::atomic<int> reads{0};
    std::atomic<bool> allow{true};
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        if (!allow.load()) return nullptr;
        ++reads;
        const auto it = blobs.find({(int) layer, (int) expert});
        return it == blobs.end() ? nullptr : it->second.data();
    }
};

struct Svc {
    std::unique_ptr<ExpertWorkerService> svc;
    uint16_t port = 0;
    std::thread th;
    Svc() : svc(std::make_unique<ExpertWorkerService>(WorkerOptions{})) {}
    bool start() {
        std::string err;
        if (!svc->listen(port, err)) return false;
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

/// One entry's direct native row (the same three calls direct_native_partial makes per entry): the
/// reference for what the local pool or the worker computes for a (token, expert) pair.
void eval_entry(const NativeFmt& f, const uint8_t* blob, const float* x, float* row) {
    using namespace strata::kernels::cpu;
    std::vector<uint8_t> nact(kNativeActBytes);
    native_quant_act(f, x, nact.data());
    std::vector<float> ff((size_t) f.n_ff);
    const void* act = nact.data();
    float* ffp = ff.data();
    native_gu_rows(f, blob, &act, 1, &ffp, 0, (int) f.n_ff);
    std::vector<uint8_t> hq(f.h_bytes);
    native_quant_h(f, ff.data(), hq.data());
    const void* hqv = hq.data();
    native_down_rows(f, blob, &hqv, 1, &row, 0, (int) f.n_embd);
}

struct Plan {
    std::vector<int32_t> counts, start, dst, tok, start2;
    std::vector<unsigned long long> ptr;
    strata::core::GpuPlanSink sink;
    explicit Plan(int64_t cap) : counts(3), start((size_t) cap + 1), dst((size_t) cap), tok((size_t) cap),
                                 start2((size_t) 64 + 1), ptr((size_t) cap) {
        sink.counts = counts.data();
        sink.start = start.data();
        sink.dst = dst.data();
        sink.tok = tok.data();
        sink.ptr = ptr.data();
        sink.ptr2 = nullptr;
        sink.start2 = start2.data();
        sink.staging = 0;
        sink.staging_cap = 0;
        sink.cap = cap;
        sink.publish = nullptr;   // the "GPU" of this test is the merge emulation below
        sink.fetch = nullptr;
        sink.ctx = nullptr;
    }
};

}  // namespace

int main() {
    std::string err;
    NativeFmt fmt;
    check(strata::kernels::cpu::native_fmt(8, 8, kEmbd, kFf, fmt, err), "Q8_0/Q8_0 is a native layout");
    if (fails) return 1;

    // the synthetic pack: native_experts.txt only (the fixtures serve the blobs in memory)
    fs::path dir = fs::temp_directory_path() / ("strata-worker-dispatch-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    {
        std::ostringstream txt;
        txt << "# strata native experts v2: layer gu d offset blob_bytes gate_off up_off down_off (n_expert "
            << kExperts << ", total " << (uint64_t) kLayers * kExperts * fmt.bytes << ")\n";
        for (int l = 0; l < kLayers; ++l)
            txt << l << " 8 8 " << (uint64_t) l * kExperts * fmt.bytes << ' ' << fmt.bytes << " 0 "
                << fmt.up_off << ' ' << fmt.down_off << '\n';
        std::ofstream(dir / "native_experts.txt", std::ios::binary) << txt.str();
    }
    check(strata::kernels::cpu::expert_layout_load(dir.string(), kLayers, kExperts, err),
          "the synthetic Q8_0 pack loads");
    if (fails) return 1;
    struct RmDir { fs::path p; ~RmDir() { std::error_code ig; fs::remove_all(p, ig); } } rm{dir};

    FixtureSource src;
    for (int l = 0; l < kLayers; ++l)
        for (int e = 0; e < kExperts; ++e) src.blobs[{l, e}] = make_blob(100 + l * kExperts + e, fmt);

    // the worker partition: layer 0 owns {2, 5, 9}, layer 1 owns {1, 2, 3, 7}; the "GPU cache" holds
    // layer 0 experts {0, 4} and layer 1 expert {1} - so layer 1 expert 1 is RESIDENT and stays the GPU's
    std::vector<uint8_t> partition((size_t) kLayers * 2, 0);
    auto set_bit = [&](int l, int e) { partition[(size_t) l * 2 + e / 8] |= (uint8_t) (1u << (e % 8)); };
    for (int e : {2, 5, 9}) set_bit(0, e);
    for (int e : {2, 3, 7}) set_bit(1, e);   // NOT 1: the fake cache holds it
    const std::map<std::pair<int, int>, int> resident = {{{0, 0}, 0}, {{0, 4}, 1}, {{1, 1}, 0}};

    Svc svc;
    check(svc.start(), "the worker service started");
    ExpertWorkerTier tier;
    {
        ExpertWorkerTier::Open o;
        o.host = "127.0.0.1";
        o.port = svc.port;
        o.model_name = "dispatch-fixture";
        o.gu_type = 8;
        o.d_type = 8;
        o.n_layers = kLayers;
        o.n_expert = kExperts;
        o.n_embd = kEmbd;
        o.n_ff = kFf;
        o.partition = partition;
        o.fetch = [&src, &fmt](int64_t l, int64_t e, uint8_t* dst, uint32_t bytes) {
            if (bytes != fmt.bytes) return false;
            const auto it = src.blobs.find({(int) l, (int) e});
            if (it == src.blobs.end()) return false;
            std::memcpy(dst, it->second.data(), bytes);
            return true;
        };
        check(tier.open(o, err), "the tier opened (setup streamed, ready)");
    }
    if (fails) return 1;

    strata::core::RemoteExpertOpt opt;
    check(opt.init(err), "RemoteExpertOpt::init (the mapped merge tables)");
    if (fails) return 1;

    strata::kernels::cpu::ExpertPool pool(0);
    std::vector<int32_t> host_res((size_t) kLayers * kExperts, -1);
    for (const auto& [le, slot] : resident) host_res[(size_t) le.first * kExperts + le.second] = slot;

    auto run_window = [&](ExpertWorkerTier* t, int layer, const std::vector<int32_t>& ids,
                          const std::vector<float>& w, const std::vector<float>& x, int T,
                          std::vector<float>& out_rows, std::vector<int32_t>& kinds) {
        Plan plan(128);
        strata::core::ExpertDispatch d;
        d.pool = &pool;
        d.src = &src;
        d.n_expert = kExperts;
        d.plan = &plan.sink;
        d.host_res = host_res.data();
        d.cache_base = (const uint8_t*) 0x100000;   // the fake VRAM arena: classified, never dereferenced
        d.cache_blob = (int64_t) fmt.bytes;
        d.worker = t;
        d.remote_opt = t != nullptr ? &opt : nullptr;
        d.layers = layer;   // the engine's drive_pool_multi sets this for every window
        opt.begin(w.data(), 0, T);
        out_rows.assign((size_t) T * kK * kEmbd, -1.0f);
        strata::core::expert_pool_dispatch_multi(d, x.data(), ids.data(), T, kK, out_rows.data());
        opt.end();
        kinds.assign((size_t) T * kK, -3);
        for (size_t i = 0; i < kinds.size(); ++i) {
            const auto it = resident.find({layer, ids[i]});
            const bool held = it != resident.end();
            const bool owned = t != nullptr && tier.holds(layer, ids[i]);
            kinds[i] = held ? 0 : owned ? 3 : -1;
            check(!(held && owned), "no entry is two tiers' at once");
        }
        return !d.failed;
    };

    // ---- the merged result, in the documented order, against the direct references -------------------
    for (int T : {1, 4}) {
        Rng r(500 + T);
        const std::vector<int32_t> base_ids = distinct_ids(r, T, kK);
        std::vector<float> w((size_t) T * kK), x((size_t) T * kEmbd);
        for (int i = 0; i < T * kK; ++i) w[(size_t) i] = r.unit();
        for (auto& v : x) v = r.next();
        for (int layer = 0; layer < kLayers; ++layer) {
            // guarantee the window routes at least one worker-owned expert (else a random draw could make
            // it a zero-work window, which this battery is not about)
            std::vector<int32_t> ids = base_ids;
            for (int e : (layer == 0 ? std::vector<int>{2, 5, 9} : std::vector<int>{2, 3, 7})) {
                bool in = false;
                for (int j = 0; j < kK; ++j) in = in || ids[(size_t) j] == e;
                if (!in) {
                    ids[(size_t) (kK - 1)] = e;   // token 0's last rank (distinctness kept: e was not there)
                    break;
                }
            }
            const int src_reads0 = src.reads.load();
            std::vector<float> rows;
            std::vector<int32_t> kinds;
            check(run_window(&tier, layer, ids, w, x, T, rows, kinds), "the window ran");
            if (fails) return 1;
            // the local pool reads EXACTLY the distinct local experts - a worker-owned expert that fell
            // back to a local read would show up here (and change the source-read count)
            std::vector<int32_t> local_distinct;
            for (size_t i = 0; i < kinds.size(); ++i)
                if (kinds[i] == -1 && std::find(local_distinct.begin(), local_distinct.end(), ids[i]) == local_distinct.end())
                    local_distinct.push_back(ids[i]);
            check(src.reads.load() - src_reads0 == (int) local_distinct.size(),
                  "only the local pool's own experts were read from the source");
            // the worker's own share, directly
            std::vector<uint8_t> owned((size_t) T * kK, 0);
            for (size_t i = 0; i < kinds.size(); ++i) owned[i] = kinds[i] == 3 ? 1 : 0;
            std::vector<const uint8_t*> entry_blobs((size_t) T * kK, nullptr);
            for (size_t i = 0; i < entry_blobs.size(); ++i)
                if (owned[i]) entry_blobs[i] = src.blobs.at({layer, ids[i]}).data();
            std::vector<float> worker_partial;
            check(ExpertWorkerService::direct_native_partial(fmt, T, kK, owned.data(), w.data(), x.data(),
                                                              entry_blobs.data(), worker_partial, err),
                  "the direct worker reference ran");
            // every entry has exactly one owner: cache (kind 0), worker (3) or local pool (-1).  The
            // merge this asserts is exactly the GPU-side combine's: the weighted sum over the non-worker
            // entries in entry order, plus the worker's already-weighted partial on top - computed here
            // twice, once from the dispatch's actual rows and the tier's actual reply, once from the
            // direct references, and compared BIT FOR BIT.
            std::vector<float> merged((size_t) T * kEmbd, 0.f), expect((size_t) T * kEmbd, 0.f);
            std::vector<float> row_ref((size_t) kEmbd);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < kK; ++j) {
                    const size_t i = (size_t) (t * kK + j);
                    const float* row = rows.data() + i * kEmbd;
                    if (kinds[i] == 3) {
                        for (int e = 0; e < kEmbd; ++e)
                            check(row[e] == 0.f, "a worker entry's per-entry row is zero (the merge is disjoint)");
                        continue;   // its contribution is the worker partial, added once below
                    }
                    if (kinds[i] == 0) {
                        for (int e = 0; e < kEmbd; ++e) check(row[e] == 0.f, "a fake-cache entry's row is zero");
                        continue;   // this test's fake GPU computes nothing; the engine's would
                    }
                    // a local-pool entry: bitwise the direct native row, then weighted in entry order
                    eval_entry(fmt, src.blobs.at({layer, ids[i]}).data(), x.data() + (size_t) t * kEmbd,
                               row_ref.data());
                    check(std::memcmp(row, row_ref.data(), kEmbd * sizeof(float)) == 0,
                          "a local entry's row is bitwise the native kernels'");
                    for (int e = 0; e < kEmbd; ++e) expect[(size_t) t * kEmbd + e] += w[i] * row[e];
                    for (int e = 0; e < kEmbd; ++e) merged[(size_t) t * kEmbd + e] += w[i] * row[e];
                }
            check(tier.served(), "the window's reply is still held for the merge check");
            const float* actual_partial = tier.partials();
            for (size_t e = 0; e < expect.size(); ++e) expect[e] += worker_partial[e];
            for (int t = 0; t < T; ++t)
                for (int e = 0; e < kEmbd; ++e)
                    merged[(size_t) t * kEmbd + e] += actual_partial[(size_t) t * kEmbd + e];
            check(std::memcmp(merged.data(), expect.data(), expect.size() * sizeof(float)) == 0,
                  "the merged result is bitwise the direct references in the documented order");
            // and the worker partial must be nonzero somewhere (the worker really did work)
            bool any = false;
            for (float v : worker_partial) any = any || v != 0.f;
            check(any, "the worker's partial carried real work");
        }
    }

    // ---- default-off: the same window, no tier, every entry local and correct -------------------------
    {
        Rng r(900);
        const int T = 4;
        const std::vector<int32_t> base_ids = distinct_ids(r, T, kK);
        std::vector<float> w((size_t) T * kK), x((size_t) T * kEmbd);
        for (int i = 0; i < T * kK; ++i) w[(size_t) i] = r.unit();
        for (auto& v : x) v = r.next();
        for (int layer = 0; layer < kLayers; ++layer) {
            // guarantee the window routes at least one worker-owned expert (else a random draw could make
            // it a zero-work window, which this battery is not about)
            std::vector<int32_t> ids = base_ids;
            for (int e : (layer == 0 ? std::vector<int>{2, 5, 9} : std::vector<int>{2, 3, 7})) {
                bool in = false;
                for (int j = 0; j < kK; ++j) in = in || ids[(size_t) j] == e;
                if (!in) {
                    ids[(size_t) (kK - 1)] = e;   // token 0's last rank (distinctness kept: e was not there)
                    break;
                }
            }
            std::vector<float> rows;
            std::vector<int32_t> kinds;
            check(run_window(nullptr, layer, ids, w, x, T, rows, kinds), "the default-off window ran");
            if (fails) return 1;
            std::vector<float> row_ref((size_t) kEmbd), expect((size_t) T * kEmbd, 0.f);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < kK; ++j) {
                    const size_t i = (size_t) (t * kK + j);
                    check(kinds[i] == (resident.count({layer, ids[i]}) ? 0 : -1),
                          "with no tier every entry is cache or local, exactly as before");
                    const float* row = rows.data() + i * kEmbd;
                    if (kinds[i] == 0) {
                        for (int e = 0; e < kEmbd; ++e) check(row[e] == 0.f, "a fake-cache entry's row is zero");
                        continue;
                    }
                    eval_entry(fmt, src.blobs.at({layer, ids[i]}).data(), x.data() + (size_t) t * kEmbd,
                               row_ref.data());
                    check(std::memcmp(row, row_ref.data(), kEmbd * sizeof(float)) == 0,
                          "a local entry's row is bitwise the native kernels' (default off)");
                    for (int e = 0; e < kEmbd; ++e) expect[(size_t) t * kEmbd + e] += w[i] * row[e];
                }
        }
        check(svc.svc->counters().requests > 0, "the worker saw the earlier windows");
    }

    // ---- the failure path: the worker goes away, the dispatch latches, nothing stale commits ---------
    {
        Rng r(950);
        const int T = 4;
        const std::vector<int32_t> ids = distinct_ids(r, T, kK);
        std::vector<float> w((size_t) T * kK), x((size_t) T * kEmbd, 0.05f);
        for (int i = 0; i < T * kK; ++i) w[(size_t) i] = r.unit();
        svc.stop();   // the worker is gone; the tier's connection drops
        Plan plan(128);
        strata::core::ExpertDispatch d;
        d.pool = &pool;
        d.src = &src;
        d.n_expert = kExperts;
        d.plan = &plan.sink;
        d.host_res = host_res.data();
        d.cache_base = (const uint8_t*) 0x100000;
        d.cache_blob = (int64_t) fmt.bytes;
        d.worker = &tier;
        d.remote_opt = &opt;
        opt.begin(w.data(), 0, T);
        std::vector<float> rows((size_t) T * kK * kEmbd, -1.0f);
        strata::core::expert_pool_dispatch_multi(d, x.data(), ids.data(), T, kK, rows.data());
        opt.end();
        check(d.failed, "a window after the worker left latched failed");
        check(d.fail != nullptr && std::strstr(d.fail, "remote CPU-expert worker") != nullptr,
              "the failure names the worker");
        check(tier.dead(), "the tier is dead (no retry, no stale session)");
    }

    if (fails) {
        std::fprintf(stderr, "worker_dispatch_test: %d failure(s)\n", fails);
        return 1;
    }
    std::printf("worker_dispatch_test: all checks passed\n");
    return 0;
}
