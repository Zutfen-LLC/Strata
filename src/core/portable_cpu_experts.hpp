#pragma once
// Private provider seam, also exercised by the real native-row differential
// test.
#include "portable_graph.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/portable_model.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/worker/local_source.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <set>
namespace strata::core::portable_detail {
inline constexpr size_t H = 2560, FF = 640, NE = 512, K = 10, L = 48;
inline void finite(const std::vector<float> &x, const char *what) {
    for (float v : x)
        require(std::isfinite(v), std::string("portable model: nonfinite ") + what);
}
inline PortableTensorView view(const GgufModel &m, const std::string &name) {
    size_t shard = 0;
    const auto *t = m.find(name, &shard);
    require(t && m.in_bounds(*t, shard), "portable model: missing/out-of-bounds " + name);
    PortableTensorView v{name, {}, t->type, m.shard(shard).tensor_data(*t), size_t(tensor_payload_bytes(*t))};
    for (uint64_t d : t->shape) {
        require(d <= INT64_MAX, "portable model: dimension overflow");
        v.shape.push_back(int64_t(d));
    }
    return v;
}
// Standard NativeFmt is deliberately distinct from the canonical Q2_0 packed
// engine layout. Verify packed role order once, then use native mapped views.
// The session owns the GgufModel for the complete lifetime of every view.
class CpuExperts {
    const GgufModel &model;
    worker::LocalManifest manifest;
    kernels::cpu::NativeFmt fmt;
    std::array<std::array<PortableTensorView, 3>, L> weights;
    std::vector<uint8_t> staging;
    std::unique_ptr<kernels::cpu::ExpertPool> pool;
    std::array<std::array<bool, NE>, L> verified{};
    std::array<kernels::cpu::NativeFmt, L> mapped_fmt{};
    std::vector<std::filesystem::file_time_type> source_times;
    PortableExpertWork counters;
    bool compare_staging;

    void check_sources() {
        auto times = source_times;
        bool changed = false;
        for (size_t s = 0; s < model.size(); ++s) {
            const auto &file = model.shard(s);
            require(std::filesystem::file_size(file.path()) == file.file_size(),
                    "portable model: source size changed; reopen the model");
            const auto time = std::filesystem::last_write_time(file.path());
            changed |= time != source_times[s];
            times[s] = time;
        }
        source_times.swap(times);
        if (changed) {
            verified = {};
            ++counters.source_invalidations;
        }
    }

  public:
    CpuExperts(const GgufModel &m, const std::string &path, const PortableModelOptions &opts)
        : model(m), compare_staging(opts.expert_dispatch_oracle) {
        std::string error;
        require(worker::read_local_manifest(path, manifest, error), error);
        require(manifest.n_layers == L && manifest.n_expert == NE && manifest.n_embd == H &&
                    manifest.n_ff == FF && manifest.gu_type == 8 && manifest.d_type == 8,
                "portable model: expert manifest geometry/type mismatch");
        require(kernels::cpu::native_fmt(8, 8, H, FF, fmt, error), error);
        require(fmt.gu_row == H / 32 * 34 && fmt.d_row == FF / 32 * 34 && fmt.up_off == FF * fmt.gu_row &&
                    fmt.down_off == 2 * fmt.up_off && fmt.bytes == fmt.down_off + H * fmt.d_row &&
                    !fmt.tail_off && !fmt.had_gu && !fmt.had_d,
                "portable model: unsupported native Q8 expert layout");
        require(K * fmt.bytes <= opts.expert_staging_cap, "portable model: expert staging cap too small");
        const char *roles[]{"gate", "up", "down"};
        for (size_t l = 0; l < L; ++l)
            for (size_t r = 0; r < 3; ++r) {
                const auto name = "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight";
                size_t shard = 0;
                const auto *t = model.find(name, &shard);
                require(t && model.in_bounds(*t, shard), "portable model: missing expert tensor " + name);
                const auto &range = manifest.layers[l][r];
                const auto &file = manifest.files[range.file];
                const size_t per = r < 2 ? fmt.up_off : fmt.bytes - fmt.down_off;
                require(
                    t->type == 8 && t->shape == std::vector<uint64_t>{r < 2 ? H : FF, r < 2 ? FF : H, NE} &&
                        tensor_payload_bytes(*t) == per * NE && range.bytes == per &&
                        range.offset == model.shard(shard).data_start() + t->offset &&
                        file.name == std::filesystem::path(model.shard(shard).path()).filename().string() &&
                        file.bytes == model.shard(shard).file_size(),
                    "portable model: expert manifest/header/range mismatch: " + name);
                weights[l][r] = view(model, name);
            }
        for (size_t l = 0; l < L; ++l) {
            mapped_fmt[l] = fmt;
            // Same modular offsets as FileExpertSource's retained native views.
            const auto base = reinterpret_cast<uintptr_t>(weights[l][0].data);
            mapped_fmt[l].up_off = reinterpret_cast<uintptr_t>(weights[l][1].data) - base;
            mapped_fmt[l].down_off = reinterpret_cast<uintptr_t>(weights[l][2].data) - base;
        }
        for (size_t s = 0; s < model.size(); ++s)
            source_times.push_back(std::filesystem::last_write_time(model.shard(s).path()));
        staging.resize(K * fmt.bytes);
        require(opts.expert_workers >= 1 && opts.expert_workers <= 256,
                "portable model: expert worker count outside [1,256]");
        // Explicit count, no affinity changes to unrelated processes/services.
        pool = std::make_unique<kernels::cpu::ExpertPool>(opts.expert_workers, false);
    }
    PortableExpertWork work() const { return counters; }
    size_t bytes() const { return staging.size(); }
    const worker::LocalManifest &identity() const { return manifest; }
    bool load_blob(int64_t layer, int32_t expert, std::vector<uint8_t>& blob, std::string& error) {
        try {
            require(layer >= 0 && layer < int64_t(L) && expert >= 0 && expert < int32_t(NE), "invalid local asset");
            check_sources();
            blob.resize(fmt.bytes);
            const size_t offsets[]{0, fmt.up_off, fmt.down_off};
            const size_t sizes[]{fmt.up_off, fmt.up_off, fmt.bytes-fmt.down_off};
            for (size_t r=0;r<3;++r)
                std::memcpy(blob.data()+offsets[r], static_cast<const uint8_t*>(weights[size_t(layer)][r].data)+size_t(expert)*sizes[r], sizes[r]);
            require(worker::blob_digest(blob) == manifest.digests[size_t(layer)*NE+size_t(expert)], "local expert digest mismatch");
            return true;
        } catch (const std::exception& e) { blob.clear(); error=e.what(); return false; }
    }
    bool run(int64_t layer, const PortableRoute &route, std::vector<float> &rows, std::string &error,
             const std::array<bool, K>* active = nullptr) {
        error.clear();
        rows.clear();
        try {
            require(layer >= 0 && layer < int64_t(L) && route.ids.size() == K && route.activation.size() == H,
                    "portable model: invalid expert request");
            check_sources();
            finite(route.activation, "expert activation");
            alignas(64) std::array<uint8_t, kernels::cpu::kNativeActBytes> activation{};
            kernels::cpu::native_quant_act(fmt, route.activation.data(), activation.data());
            rows.assign(K * H, 0);
            std::array<kernels::cpu::ExpertJobMulti, K> jobs{};
            const size_t offsets[]{0, fmt.up_off, fmt.down_off};
            const size_t sizes[]{fmt.up_off, fmt.up_off, fmt.bytes - fmt.down_off};
            std::set<int32_t> seen;
            size_t job_count = 0;
            for (size_t slot = 0; slot < K; ++slot) {
                const int32_t id = route.ids[slot];
                require(id >= 0 && id < int32_t(NE) && seen.insert(id).second,
                        "portable model: expert ID duplicate/out of range");
                if (active && !(*active)[slot]) continue;
                auto *blob = staging.data() + slot * fmt.bytes;
                const size_t l = size_t(layer), e = size_t(id);
                if (compare_staging || !verified[l][e]) {
                    for (size_t r = 0; r < 3; ++r)
                        std::memcpy(blob + offsets[r],
                                    static_cast<const uint8_t *>(weights[size_t(layer)][r].data) +
                                        size_t(id) * sizes[r],
                                    sizes[r]);
                    counters.copied_bytes += fmt.bytes;
                    counters.digest_bytes += fmt.bytes;
                    ++counters.verifications;
                    const auto digest = worker::blob_digest(blob, fmt.bytes);
                    const auto expected = manifest.digests[size_t(layer) * NE + size_t(id)];
                    require(digest.a == expected.a && digest.b == expected.b,
                            "portable model: selected expert digest mismatch at layer " +
                                std::to_string(layer) + " expert " + std::to_string(id));
                    verified[l][e] = true; // publish only after all three roles passed
                } else {
                    ++counters.hits;
                }
                jobs[job_count].blob =
                    compare_staging ? blob : static_cast<const uint8_t *>(weights[l][0].data) + e * sizes[0];
                jobs[job_count].fmt = compare_staging ? nullptr : &mapped_fmt[l];
                jobs[job_count].nt = 1;
                jobs[job_count].nact[0] = activation.data();
                jobs[job_count].out[0] = rows.data() + slot * H;
                ++job_count;
            }
            if (job_count) {
                ++counters.native_batches;
                pool->run_split_multi_native(fmt, jobs.data(), int(job_count));
            }
            finite(rows, "expert rows"); // deliberately unweighted, in route order
            return true;
        } catch (const std::exception &e) {
            rows.clear();
            error = e.what();
            return false;
        }
    }
};
} // namespace strata::core::portable_detail
