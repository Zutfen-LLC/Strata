// Manual bounded numerical discriminator. Real Q8 services/pools/group; no model, GPU or external port.
#include "local_fixture.hpp"
#include "strata/worker/cpu_group.hpp"
#include "diagnostic_child.hpp"
#include "cpu_capture_micro.hpp"
#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <poll.h>
#include <cerrno>

namespace {
constexpr int H = 2560, K = 10, Experts = 32, CpuExperts = 20;
using Vec = std::vector<float>;
using Mask = std::vector<uint8_t>;
uint64_t ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

using Child = strata::worker::diagnostic::Child;
strata::worker::diagnostic::ChildSpec launch(const char* binary,const Fixture& fixture) {
    strata::worker::diagnostic::ChildSpec spec;spec.binary=binary;spec.root=fixture.root.string();
    spec.manifest=(fixture.root/"local.manifest").string();return spec;
}

struct Input {
    std::string name;
    int layer = 0;
    std::vector<int32_t> ids;
    Vec weights, x;
    Mask mask(int peer = -1) const {
        Mask out(K);
        for (int i = 0; i < K; ++i) out[i] = ids[i] < CpuExperts && (peer < 0 || ids[i] % 2 == peer);
        return out;
    }
};
std::vector<Input> inputs() {
    std::vector<Input> out;
    for (int c = 0; c < 7; ++c) {
        Input in; in.layer = c % 2; in.x.resize(H); in.ids.resize(K); in.weights.resize(K);
        const char* names[] = {"dense", "shifted", "sparse", "peer0-only", "peer1-only", "zero-cpu", "cancellation"};
        in.name = names[c];
        uint32_t seed = 123 + c;
        for (auto& x : in.x) { seed = seed * 1664525u + 1013904223u; x = (int(seed >> 9) - (1 << 22)) * 0x1p-23f; }
        for (int i = 0; i < K; ++i) {
            in.ids[i] = c == 1 ? 19 - i : c == 2 ? (i % 2 ? 20 + i : i / 2) :
                c == 3 ? 2 * i : c == 4 ? 2 * i + 1 : c == 5 ? 20 + i : c == 6 ? 10 + i : i;
            in.weights[i] = (float)(i + 1) / 55.f;
        }
        if (c == 6) {
            // Adjacent experts have identical Q8 bytes. Entry-order contributions mostly cancel;
            // the final pair leaves a small residual. Signed weights are never renormalized.
            std::fill(in.weights.begin(), in.weights.end(), 1.f);
            for (int i = 1; i < K; i += 2) in.weights[i] = -1.f;
            in.weights.back() += 0x1p-20f;
        }
        auto distinct = in.ids; std::sort(distinct.begin(), distinct.end());
        require(distinct.front() >= 0 && distinct.back() < Experts &&
                std::adjacent_find(distinct.begin(), distinct.end()) == distinct.end(), "distinct bounded K10 ids");
        out.push_back(std::move(in));
    }
    return out;
}

enum class Gate { Bits, Envelope };
void compare(const std::string& tag, const Vec& got, const Vec& expected, Gate gate,
             const std::vector<double>& magnitude = {}) {
    require(got.size() == expected.size(), "numeric shape mismatch");
    double maximum = 0, relative = 0, delta2 = 0, norm2 = 0;
    size_t first = got.size(); bool envelope = true;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]) && std::isfinite(expected[i]), "nonfinite diagnostic result");
        if (first == got.size() && std::memcmp(&got[i], &expected[i], 4)) first = i;
        const double delta = std::abs((double)got[i] - expected[i]);
        maximum = std::max(maximum, delta);
        relative = std::max(relative, delta / std::max(1e-12, std::abs((double)expected[i])));
        delta2 += delta * delta; norm2 += (double)expected[i] * expected[i];
        if (gate == Gate::Envelope) envelope = envelope && delta <= 1e-4 + 4e-6 * magnitude.at(i);
    }
    const bool bits = first == got.size();
    std::printf("NUMERIC stage=%s bits=%d max_abs=%.9g max_rel=%.9g rel_l2=%.9g rmse=%.9g first=%lld",
        tag.c_str(), bits, maximum, relative, std::sqrt(delta2 / std::max(1e-30, norm2)),
        std::sqrt(delta2 / got.size()), bits ? -1ll : (long long)first);
    if (!bits) std::printf(" got=%a expected=%a", (double)got[first], (double)expected[first]);
    std::printf(" gate=%s pass=%d\n", gate == Gate::Bits ? "bits" : "1e-4+4e-6*sum_abs",
        gate == Gate::Bits ? bits : envelope);
    if (!(gate == Gate::Bits ? bits : envelope)) ++failures;
}
Vec reduce(const Input& in, const Mask& mask, const Vec& rows) {
    Vec out(H); ExpertWorkerService::accumulate_partial(1, K, mask.data(), in.weights.data(), rows.data(), H, out.data());
    return out;
}
Vec ordered(const Input& in, const Vec& a, const Vec& b) {
    const auto m = in.mask(0); const bool first = std::find(m.begin(), m.end(), 1) != m.end();
    Vec out = first ? a : b;
    const auto n = in.mask(1);
    if (first && std::find(n.begin(), n.end(), 1) != n.end())
        for (int i = 0; i < H; ++i) out[i] = out[i] + b[i];
    return out;
}
Vec rows(Fixture& f, strata::kernels::cpu::ExpertPool& pool, const Input& in, const Mask& mask) {
    Vec out(K * H, 0.f); std::vector<uint8_t> act(strata::kernels::cpu::kNativeActBytes);
    strata::kernels::cpu::native_quant_act(f.fmt, in.x.data(), act.data());
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
    for (int i = 0; i < K; ++i) if (mask[i]) {
        strata::kernels::cpu::ExpertJobMulti job{};
        job.blob = f.blobs[in.layer * Experts + in.ids[i]].data(); job.nt = 1; job.fmt = &f.fmt;
        job.nact[0] = act.data(); job.out[0] = out.data() + i * H; jobs.push_back(job);
    }
    pool.run_split_multi_native(f.fmt, jobs.data(), (int)jobs.size());
    return out;
}
struct Reference { Vec full, peer[2]; std::vector<double> magnitude; };
Vec independent_row(Fixture& f, const Input& in, int entry) {
    using namespace strata::kernels::cpu;
    const auto* blob = f.blobs[in.layer * Experts + in.ids[entry]].data();
    std::vector<uint8_t> act(kNativeActBytes), hq(f.fmt.h_bytes); Vec ff(f.fmt.n_ff), out(H);
    native_quant_act(f.fmt, in.x.data(), act.data());
    const void* a = act.data(); float* hidden = ff.data();
    native_gu_rows(f.fmt, blob, &a, 1, &hidden, 0, (int)f.fmt.n_ff);
    native_quant_h(f.fmt, ff.data(), hq.data());
    const void* h = hq.data(); float* result = out.data();
    native_down_rows(f.fmt, blob, &h, 1, &result, 0, H);
    return out;
}
std::vector<Reference> references(Fixture& f, const std::vector<Input>& cases) {
    strata::kernels::cpu::ExpertPool pool(12); std::vector<Reference> refs;
    for (const auto& in : cases) {
        auto common = rows(f, pool, in, in.mask()); Reference ref; ref.magnitude.assign(H, 0);
        Vec independent(K * H, 0.f);
        for (int i = 0; i < K; ++i) if (in.mask()[i]) {
            auto result = independent_row(f, in, i);
            std::copy(result.begin(), result.end(), independent.begin() + i * H);
            for (int j = 0; j < H; ++j) ref.magnitude[j] += std::abs((double)in.weights[i] * result[j]);
        }
        compare(in.name + "/common-pool-vs-independent-rows", common, independent, Gate::Bits);
        ref.full = reduce(in, in.mask(), independent);
        for (int p = 0; p < 2; ++p) {
            const auto mask = in.mask(p); auto split = rows(f, pool, in, mask); Vec selected(K * H, 0.f);
            for (int i = 0; i < K; ++i) if (mask[i])
                std::copy_n(common.data() + i * H, H, selected.data() + i * H);
            compare(in.name + "/peer" + std::to_string(p) + "-batch-vs-common-rows", split, selected, Gate::Bits);
            ref.peer[p] = reduce(in, mask, independent);
        }
        compare(in.name + "/reference-grouping-vs-unsplit", ordered(in, ref.peer[0], ref.peer[1]),
                ref.full, Gate::Envelope, ref.magnitude);
        refs.push_back(std::move(ref));
    }
    return refs;
}
ExpertWorkerTier::Open option(const Fixture& f, const Child& child, int peer = -1) {
    ExpertWorkerTier::Open o; o.port = child.port; o.local = &f.m; o.resident = true;
    o.n_layers = 2; o.n_expert = Experts; o.n_embd = H; o.n_ff = 640; o.gu_type = o.d_type = 8;
    o.partition.assign(2 * (Experts / 8), 0); o.setup_ms = 30000; o.deadline_ms = 5000;
    for (int l = 0; l < 2; ++l) for (int e = 0; e < CpuExperts; ++e) if (peer < 0 || e % 2 == peer)
        o.partition[l * (Experts / 8) + e / 8] |= (uint8_t)(1u << (e % 8));
    return o;
}
Vec result(ExpertWorkerTier& tier) {
    return tier.served() ? Vec(tier.partials(), tier.partials() + H) : Vec(H, 0.f);
}
std::vector<Vec> full(Fixture& f, Child& child, const char* label, const std::vector<Input>& cases,
          const std::vector<Reference>& refs) {
    ExpertWorkerTier tier; std::string err; require(tier.open(option(f, child), err), err); std::vector<Vec> actual;
    for (size_t c = 0; c < cases.size(); ++c) for (int p : {-1, 0, 1}) {
        const auto& in = cases[c]; auto mask = in.mask(p);
        require(tier.post(in.layer, in.x.data(), in.ids.data(), in.weights.data(), mask.data(), 1, K, err), err);
        require(tier.finish(err), err);
        const auto got = result(tier); if (p < 0) actual.push_back(got);
        compare(in.name + "/" + label + (p < 0 ? "-full" : "-masked" + std::to_string(p)), got,
                p < 0 ? refs[c].full : refs[c].peer[p], Gate::Bits);
        child.pump();
    }
    return actual;
}
void exercise(const char* a_path, const char* b_path) {
    Fixture f(Experts, 10); const auto cases = inputs(); const auto refs = references(f, cases);
    std::printf("FIXTURE experts=%d cpu_experts=%d layers=2 K=10 T=1 workers=12 loaders=4 full_arena_bytes=%llu paired_from=10\n",
        Experts, CpuExperts, (unsigned long long)(2 * CpuExperts * f.fmt.bytes));
    Child a(launch(a_path,f)), b(launch(b_path,f));
    const auto full_a = full(f, a, "binaryA", cases, refs), full_b = full(f, b, "binaryB", cases, refs);
    for (size_t c = 0; c < cases.size(); ++c)
        compare(cases[c].name + "/binaryA-vs-binaryB-full", full_a[c], full_b[c], Gate::Bits);
    a.unload(); b.unload(); // only these fresh ephemeral fixture arenas; never a fleet endpoint
    const auto oa = option(f, a, 0), ob = option(f, b, 1);
    std::vector<Vec> manual;
    {
        ExpertWorkerTier ta, tb; std::string err; require(ta.open(oa, err) && tb.open(ob, err), err);
        for (size_t c = 0; c < cases.size(); ++c) {
            const auto& in = cases[c]; auto ma = in.mask(0), mb = in.mask(1);
            require(ta.post(in.layer, in.x.data(), in.ids.data(), in.weights.data(), ma.data(), 1, K, err), err);
            require(tb.post(in.layer, in.x.data(), in.ids.data(), in.weights.data(), mb.data(), 1, K, err), err);
            const bool good_a = ta.finish(err), good_b = tb.finish(err); require(good_a && good_b, err);
            auto pa = result(ta), pb = result(tb);
            compare(in.name + "/subsetA-vs-common-batch-mask", pa, refs[c].peer[0], Gate::Bits);
            compare(in.name + "/subsetB-vs-common-batch-mask", pb, refs[c].peer[1], Gate::Bits);
            manual.push_back(ordered(in, pa, pb)); a.pump(); b.pump();
        }
    }
    {
        ExpertWorkerGroup group; std::string err; require(group.open({oa, ob}, option(f, a).partition, err), err);
        check(group.peer_stats(0).cache_hit && group.peer_stats(1).cache_hit, "matching synthetic subsets reused");
        for (size_t c = 0; c < cases.size(); ++c) {
            auto pending = cases[c]; auto owned = pending.mask();
            const auto before_a = group.peer_stats(0).requests, before_b = group.peer_stats(1).requests;
            require(group.post(pending.layer, pending.x.data(), pending.ids.data(), pending.weights.data(), owned.data(), 1, K, err), err);
            pending.x.assign(H, 99.f); pending.weights.assign(K, 99.f); pending.ids.assign(K, Experts - 1);
            require(group.finish(err), err);
            for (int p = 0; p < 2; ++p) {
                const auto mask = cases[c].mask(p); const bool active = std::find(mask.begin(), mask.end(), 1) != mask.end();
                check(group.peer_stats(p).requests == (p ? before_b : before_a) + (active ? 1 : 0),
                      "only active peers issue INFER");
            }
            Vec got = group.served() ? Vec(group.partials(), group.partials() + H) : Vec(H, 0.f);
            if (cases[c].name == "zero-cpu") check(!group.served() && !group.partials(), "zero CPU share exposes no stale partial");
            compare(cases[c].name + "/group-vs-manual-peers", got, manual[c], Gate::Bits);
            compare(cases[c].name + "/group-vs-grouped-common-reference", got,
                ordered(cases[c], refs[c].peer[0], refs[c].peer[1]), Gate::Bits);
            compare(cases[c].name + "/group-vs-unsplit", got, refs[c].full, Gate::Envelope, refs[c].magnitude);
            compare(cases[c].name + "/group-vs-actual-fullA", got, full_a[c], Gate::Envelope, refs[c].magnitude);
            a.pump(); b.pump();
        }
    }
    {
        auto cancelled = oa; cancelled.cancelled = [] { return true; }; ExpertWorkerGroup group; std::string err;
        check(!group.open({cancelled, ob}, option(f, a).partition, err), "pre-connect cancellation refused");
    }
    {
        ExpertWorkerGroup group; std::string err; require(group.open({oa, ob}, option(f, a).partition, err), err);
        check(group.peer_stats(0).cache_hit && group.peer_stats(1).cache_hit, "cancelled open preserved complete subsets");
        const auto& in = cases.front(); auto mask = in.mask();
        require(group.post(in.layer, in.x.data(), in.ids.data(), in.weights.data(), mask.data(), 1, K, err) &&
                group.finish(err), err);
        compare("after-cancel/group-vs-manual", Vec(group.partials(), group.partials() + H), manual.front(), Gate::Bits);
    }
    a.stop(); b.stop(); std::puts("CLEANUP fixture_children_reaped=2 model_loads=0");
}
}
int main(int argc, char** argv) {
    if((argc==10 || (argc==11 && std::string(argv[10])=="--readiness")) && std::string(argv[1])=="--worker-a" && std::string(argv[3])=="--worker-b" &&
       std::string(argv[5])=="--capture-record0"){
        CpuCaptureFile f;LocalManifest local;std::string err;strata::worker::diagnostic::MicroReport result;
        const bool decoded=std::filesystem::path(argv[6]).is_absolute() && std::filesystem::path(argv[7]).is_absolute() &&
            read_cpu_capture(argv[6],f,err) && read_local_manifest(argv[7],local,err);
        const bool ok=decoded && strata::worker::diagnostic::run_capture_micro(f,local,argv[7],argv[8],argv[9],argv[2],argv[4],result,err,{},argc==11);
        if(!ok)std::fprintf(stderr,"MICRO failure: %s\n",err.c_str());
        std::printf("CPU_CAPTURE_MICRO ok=%d reference=%d peer0=%d peer1=%d group=%d payload=%llu owned_reaped=%d peak_owned_rss_bound=%llu min_available=%llu GEN=0 full_endpoint_connections=0\n",
            ok,result.reference,result.peer[0],result.peer[1],result.group,(unsigned long long)result.payload,result.reaped,
            (unsigned long long)result.peak_rss,(unsigned long long)result.min_available);return ok?0:1;
    }
    if (argc != 5 || std::string(argv[1]) != "--worker-a" || std::string(argv[3]) != "--worker-b") {
        std::fprintf(stderr, "usage: cpu_group_numeric_diagnostic --worker-a /absolute/worker --worker-b /absolute/worker\n");
        std::fprintf(stderr, "  optional: --capture-record0 /absolute/capture /absolute/manifest /absolute/local-root /absolute/new-output-root [--readiness]\n");
        return 2;
    }
    try { exercise(argv[2], argv[4]); }
    catch (const std::exception& error) { std::fprintf(stderr, "FAIL %s\n", error.what()); ++failures; }
    std::printf("CPU_GROUP_NUMERIC_DIAGNOSTIC failures=%d\n", failures);
    return failures ? 1 : 0;
}
