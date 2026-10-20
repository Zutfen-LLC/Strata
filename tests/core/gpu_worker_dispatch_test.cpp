// Static-authored fixture source; never run on Geekom. Execute only in the Hermes CUDA gate.
// Real CPU service + real network CUDA service + real dispatcher and mapped merge. The primary
// GPU's resident rows are emulated to keep the fixture bounded; network CUDA execution is real.
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/worker/expert_tier.hpp"
#include "strata/worker/expert_worker.hpp"
#include "strata/worker/gpu_client.hpp"
#include "strata/worker/gpu_worker.hpp"
#include "strata/worker/setup_keepalive.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>

namespace {
using namespace strata::worker;
namespace cpu = strata::kernels::cpu;
namespace fs = std::filesystem;
constexpr int H = 2560, FF = 640, E = 12, K = 10;
int failures = 0;
bool check(bool ok, const char* what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
    return ok;
}
bool gpu(cudaError_t rc) { return check(rc == cudaSuccess, cudaGetErrorString(rc)); }
struct Allocation {
    void* p = nullptr;
    explicit Allocation(size_t bytes) { gpu(cudaMalloc(&p, bytes)); }
    ~Allocation() { if (p) cudaFree(p); }
};
std::vector<uint8_t> blob(const cpu::NativeFmt& f, int expert) {
    std::vector<uint8_t> b(f.bytes);
    std::vector<float> row(H);
    auto role = [&](size_t off, int rows, int cols, size_t stride, int salt) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c)
                row[c] = 0.003f * (float)(((r * 17 + c * 13 + expert * 7 + salt) % 23) - 11);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, row.data(), b.data() + off + r * stride, 0, 1, cols, nullptr);
        }
    };
    role(0, FF, H, f.gu_row, 1); role(f.up_off, FF, H, f.gu_row, 3);
    role(f.down_off, H, FF, f.d_row, 5);
    return b;
}
void row_reference(const cpu::NativeFmt& f, const uint8_t* b, const float* x, float* out) {
    std::vector<uint8_t> act(cpu::kNativeActBytes), hq(f.h_bytes);
    std::vector<float> ff(FF);
    cpu::native_quant_act(f, x, act.data());
    const void* a = act.data(); float* h = ff.data();
    cpu::native_gu_rows(f, b, &a, 1, &h, 0, FF);
    cpu::native_quant_h(f, h, hq.data());
    const void* q = hq.data(); cpu::native_down_rows(f, b, &q, 1, &out, 0, H);
}
class Source final : public strata::core::ExpertSource {
public:
    std::vector<std::vector<uint8_t>> blobs;
    int reads = 0;
    const uint8_t* blob(int64_t, int64_t e) override { ++reads; return blobs.at((size_t)e).data(); }
};
struct Services {
    std::unique_ptr<ExpertWorkerService> cpu;
    std::unique_ptr<GpuWorkerService> cuda;
    std::thread ct, gt;
    ~Services() {
        if (cuda) cuda->stop(); if (gt.joinable()) gt.join();
        if (cpu) cpu->stop(); if (ct.joinable()) ct.join();
    }
    void stop_gpu() { cuda->stop(); if (gt.joinable()) gt.join(); }
};
struct Plan {
    int32_t counts[3]{}, start[129]{}, dst[128]{}, tok[128]{}, start2[65]{};
    unsigned long long ptr[128]{};
    strata::core::GpuPlanSink sink;
    Plan() {
        sink.counts = counts; sink.start = start; sink.dst = dst; sink.tok = tok; sink.ptr = ptr;
        sink.start2 = start2; sink.cap = 128; sink.staging_cap = 0;
    }
};

bool delayed_acquisition_keeps_cpu_ready(const fs::path& root, const cpu::NativeFmt& f,
                                        Source& source, const GpuPlan& assigned) {
    const auto cache = root / "keepalive-cache";
    fs::create_directory(cache);
    Services services;
    WorkerOptions co;
    co.workers = 1; co.memory_budget = 16ull << 20; co.deadline_ms = 200;
    services.cpu = std::make_unique<ExpertWorkerService>(co);
    GpuWorkerOptions go;
    go.plan = assigned; go.cache_root = cache.string(); go.cache_budget = 32ull << 20;
    services.cuda = std::make_unique<GpuWorkerService>(go, create_cuda_gpu_executor(0));
    std::string err;
    uint16_t cp = 0, gp = 0;
    if (!check(services.cpu->listen(cp, err) && services.cuda->listen(gp, err), "keepalive fixture listeners")) return false;
    services.ct = std::thread([&] { services.cpu->run(); });
    services.gt = std::thread([&] { services.cuda->run(); });
    auto fetch = [&](int64_t, int64_t e, uint8_t* dst, uint32_t n) {
        if (e < 0 || e >= E || n != f.bytes) return false;
        std::memcpy(dst, source.blobs[(size_t)e].data(), n); return true;
    };
    ExpertWorkerTier cpu_tier;
    ExpertWorkerTier::Open cop;
    cop.port = cp; cop.n_layers = 1; cop.n_expert = E; cop.n_embd = H; cop.n_ff = FF;
    cop.partition = {0x0c, 0}; cop.fetch = fetch;
    if (!check(cpu_tier.open(cop, err), "short-idle CPU READY")) return false;
    const uint64_t session = cpu_tier.stats().session;
    CpuSetupKeepalive keepalive(cpu_tier, 25);
    GpuWorkerTier gpu_tier;
    GpuWorkerTier::Open gop;
    gop.port = gp; gop.plan = assigned;
    gop.cancelled = [&] { return keepalive.failed(); };
    gop.fetch = [&](int64_t l, int64_t e, uint8_t* dst, uint32_t n) {
        // Each assigned fetch exceeds the CPU service's complete idle receive deadline.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        return fetch(l, e, dst, n);
    };
    const bool gpu_ready = gpu_tier.open(gop, err);
    std::string keepalive_err;
    const bool still_ready = keepalive.finish(keepalive_err); // joins before any further CPU socket access
    WorkerStatusInfo status;
    return check(gpu_ready && still_ready, "STATUS preserves CPU connection through delayed GPU acquisition") &&
           check(cpu_tier.status(status, err) && status.ready == 1 && status.session == session && status.requests == 0,
                 "same CPU session remains READY without any inference or residency renewal");
}
}
int main() {
    std::string err;
    cpu::NativeFmt f;
    if (!check(cpu::native_fmt(8, 8, H, FF, f, err), "bounded Q8_0 layout")) return 1;
    const auto root = fs::temp_directory_path() / ("strata-network-route-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root / "cache");
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove_all(p, ec); } } cleanup{root};
    {
        std::ofstream index(root / "native_experts.txt");
        index << "# strata native experts v2: layer gu d offset blob_bytes gate_off up_off down_off (n_expert "
              << E << ", total " << E * f.bytes << ")\n0 8 8 0 " << f.bytes << " 0 " << f.up_off << ' ' << f.down_off << '\n';
    }
    if (!check(cpu::expert_layout_load(root.string(), 1, E, err), "fixture index")) return 1;
    Source source;
    for (int e = 0; e < E; ++e) source.blobs.push_back(blob(f, e));
    LocalManifest metadata;
    metadata.n_layers = 1; metadata.n_expert = E; metadata.n_embd = H; metadata.n_ff = FF;
    metadata.gu_type = metadata.d_type = 8; metadata.identity = {1729, 1733}; metadata.index_digest = {19, 23};
    metadata.layers.resize(1);
    for (const auto& b : source.blobs) metadata.digests.push_back(blob_digest(b));
    GpuPlan assigned;
    if (!check(make_gpu_plan(metadata, {{0, 2}, {0, 3}}, assigned, err), "explicit GPU subset")) return 1;
    if (!delayed_acquisition_keeps_cpu_ready(root, f, source, assigned)) return 1;
    Services services;
    WorkerOptions co; co.workers = 1; co.memory_budget = 64ull << 20;
    services.cpu = std::make_unique<ExpertWorkerService>(co);
    GpuWorkerOptions go; go.plan = assigned; go.cache_root = (root / "cache").string(); go.cache_budget = 32ull << 20;
    services.cuda = std::make_unique<GpuWorkerService>(go, create_cuda_gpu_executor(0));
    uint16_t cp = 0, gp = 0;
    if (!check(services.cpu->listen(cp, err), "CPU listen") || !check(services.cuda->listen(gp, err), "CUDA listen")) return 1;
    services.ct = std::thread([&] { services.cpu->run(); }); services.gt = std::thread([&] { services.cuda->run(); });
    ExpertWorkerTier cpu_tier;
    ExpertWorkerTier::Open cop;
    cop.port = cp; cop.n_layers = 1; cop.n_expert = E; cop.n_embd = H; cop.n_ff = FF;
    cop.partition = {0xfc, 0x0f}; // unchanged full CPU complement, INCLUDING CUDA04's duplicate 2,3 weights
    auto fetch = [&](int64_t, int64_t e, uint8_t* dst, uint32_t n) {
        if (e < 0 || e >= E || n != f.bytes) return false;
        std::memcpy(dst, source.blobs[(size_t)e].data(), n); return true;
    };
    cop.fetch = fetch;
    GpuWorkerTier gpu_tier;
    GpuWorkerTier::Open gop; gop.port = gp; gop.plan = assigned; gop.fetch = fetch;
    if (!check(cpu_tier.open(cop, err), "CPU READY") || !check(gpu_tier.open(gop, err), "CUDA resident READY")) return 1;
    check(cpu_tier.holds(0, 2) && gpu_tier.holds(0, 2), "CPU duplicate weights remain resident");
    strata::core::RemoteExpertOpt merge;
    if (!check(merge.init(err), "real merge tables")) return 1;
    cpu::ExpertPool pool(1);
    int32_t resident[E]; std::fill_n(resident, E, -1); resident[0] = 0; resident[1] = 1;
    const int T = 2;
    const int32_t ids[T * K] = {0,2,4,6,8,1,3,5,7,9, 3,1,5,7,9,2,0,4,6,8};
    std::vector<float> weights(T * K), x(T * H);
    for (int i = 0; i < T * K; ++i) weights[i] = 0.025f * (i % K + 1);
    for (int i = 0; i < T * H; ++i) x[i] = 0.02f * (float)((i * 11 % 37) - 18);
    auto dispatch = [&](std::vector<float>& rows, Plan& plan) {
        strata::core::ExpertDispatch d;
        d.pool = &pool; d.src = &source; d.n_expert = E; d.plan = &plan.sink; d.host_res = resident;
        d.cache_base = (const uint8_t*)0x100000; d.cache_blob = f.bytes;
        d.worker = &cpu_tier; d.gpu_worker = &gpu_tier; d.remote_opt = &merge;
        rows.assign(T * K * H, -11.5f);
        merge.begin(weights.data(), 0, T);
        strata::core::expert_pool_dispatch_multi(d, x.data(), ids, T, K, rows.data());
        merge.end();
        if (!d.failed) check(d.gpu_worker_entries == 4 && d.worker_entries == 12,
            "each routed occurrence belongs to exactly one network executor");
        return d.failed;
    };
    std::vector<float> rows; Plan plan;
    if (!check(!dispatch(rows, plan), "mixed local GPU/CUDA04/CPU05 dispatch succeeds")) {
        std::fprintf(stderr, "detail: %s\n", err.c_str()); return 1;
    }
    check(source.reads == 0, "no local CPU expert fallback or source reads");
    check(plan.counts[0] == 2 && plan.counts[1] == 4 && plan.counts[2] == 0, "only primary resident experts in GPU plan");
    std::vector<uint8_t> cpu_mask(T * K);
    std::vector<const uint8_t*> entry_blobs(T * K, nullptr);
    for (int i = 0; i < T * K; ++i) {
        cpu_mask[i] = ids[i] >= 4;
        if (cpu_mask[i]) entry_blobs[i] = source.blobs[(size_t)ids[i]].data();
        if (ids[i] >= 2) check(std::all_of(rows.data()+i*H, rows.data()+(i+1)*H,
            [](float v) { return v == 0; }), "all network rows zero before weighted merge");
    }
    std::vector<float> cpu_ref;
    if (!check(ExpertWorkerService::direct_native_partial(f, T, K, cpu_mask.data(), weights.data(), x.data(),
        entry_blobs.data(), cpu_ref, err), "CPU uncovered direct reference")) return 1;
    check(std::memcmp(cpu_ref.data(), cpu_tier.partials(), cpu_ref.size()*sizeof(float)) == 0,
          "CPU05 receives ONLY uncovered experts, never CUDA04 duplicates");
    std::vector<float> base(T * H), expected(T * H), row(H);
    for (int i = 0; i < T * K; ++i) if (ids[i] < 2) {
        row_reference(f, source.blobs[(size_t)ids[i]].data(), x.data() + (i / K) * H, row.data());
        for (int h = 0; h < H; ++h) base[(i/K)*H+h] += weights[i]*row[h];
    }
    bool nonzero = false;
    for (size_t i = 0; i < expected.size(); ++i) {
        // accumulate_host adds GPU04 then CPU05 to zero; combine adds that sum to primary output.
        expected[i] = base[i] + (gpu_tier.partials()[i] + cpu_ref[i]);
        nonzero = nonzero || gpu_tier.partials()[i] != 0;
    }
    check(nonzero, "CUDA04 contributed nonzero executed output");
    Allocation merged(base.size()*sizeof(float));
    if (!gpu(cudaMemcpy(merged.p, base.data(), base.size()*sizeof(float), cudaMemcpyHostToDevice))) return 1;
    merge.combine((float*)merged.p, nullptr, 0, T, nullptr, 0, nullptr);
    if (!gpu(cudaMemcpy(base.data(), merged.p, base.size()*sizeof(float), cudaMemcpyDeviceToHost))) return 1;
    check(std::memcmp(base.data(), expected.data(), base.size()*sizeof(float)) == 0,
          "actual merge includes local/GPU04/CPU05 exactly once in declared order");
    // Failure must not advance CPU serving or commit an earlier partial. The sync GPU post detects
    // the lost peer before CPU post; the engine's failed latch rejects the affected request.
    WorkerStatusInfo before_loss, after_loss;
    if (!check(cpu_tier.status(before_loss, err), "CPU counters before GPU loss")) return 1;
    services.stop_gpu();
    Plan lost_plan;
    check(dispatch(rows, lost_plan), "GPU loss fails the affected request");
    check(gpu_tier.dead(), "GPU loss latches dead; no invisible CPU retry");
    check(cpu_tier.status(after_loss, err) && after_loss.requests == before_loss.requests,
          "CPU05 never receives GPU04 fallback work");
    check(source.reads == 0, "GPU loss never falls back to local source");
    std::printf("gpu_worker_dispatch_test: failures=%d (primary rows emulated; network CUDA and merge real)\n", failures);
    return failures ? 1 : 0;
}
