// Selective Q8_0 network CUDA assets. Operator metadata and content digests are trusted inputs;
// Digest128 is the existing noncryptographic integrity check, never authentication.
#pragma once

#include "strata/worker/local_source.hpp"
#include <utility>

namespace strata::worker {
inline constexpr size_t kGpuPlanMax = 4u << 20;
inline constexpr uint64_t kGpuBlobMax = 8u << 20;

struct GpuAsset {
    uint16_t layer = 0, expert = 0;
    uint64_t bytes = 0;
    Digest128 digest{};
};
struct GpuPlan {
    uint64_t model_id = 0;
    uint16_t n_layers = 0, n_expert = 0, n_embd = 0, n_ff = 0, gu_type = 8, d_type = 8;
    Digest128 layout_id{}, placement_id{};
    std::vector<GpuAsset> assets; // strictly increasing (layer, expert); each appears once
    bool owns(int layer, int expert) const;
    const GpuAsset *find(int layer, int expert) const;
    uint64_t assigned_bytes() const;
};
bool validate_gpu_plan(const GpuPlan &plan, std::string &err);
bool encode_gpu_plan(const GpuPlan &plan, std::vector<uint8_t> &out, std::string &err);
bool decode_gpu_plan(const uint8_t *data, size_t bytes, GpuPlan &out, std::string &err);
bool read_gpu_plan(const std::string &path, GpuPlan &out, std::string &err);
bool write_gpu_plan(const std::string &path, const GpuPlan &plan, std::string &err);
bool make_gpu_plan(const LocalManifest &source, const std::vector<std::pair<uint16_t, uint16_t>> &assigned,
                   GpuPlan &out, std::string &err);

struct GpuAssetStats {
    uint64_t assigned_bytes = 0, cached_bytes = 0, transferred_bytes = 0;
    uint64_t source_read_bytes = 0, persisted_bytes = 0, retained_disk_bytes = 0;
    uint64_t host_staging_peak = 0, verified_bytes = 0, reused_bytes = 0;
};

// Uses existing per-expert GGUF ranges: gate || up || down, with three bounded pread calls.
// No complete layer, shard, or model is staged. Construct this only on the source/engine host.
class GpuAssetSource {
public:
    bool open(const std::string &root, const LocalManifest &source, const GpuPlan &plan, std::string &err);
    bool read(const GpuAsset &asset, std::vector<uint8_t> &out, std::string &err);
    bool read_into(const GpuAsset &asset, uint8_t *dst, size_t bytes, std::string &err);
    const GpuAssetStats &stats() const { return stats_; }
private:
    LocalSource source_;
    GpuPlan plan_;
    GpuAssetStats stats_;
};

// Dedicated operator-created cache directory. POSIX only. An exclusive process lock prevents two
// workers from publishing at once. All regular files count against budget, including stale temp files
// and unrelated files; nothing is automatically evicted. prepare() reserves space for every missing
// blob while retaining previous bytes, so changed plans cannot grow disk without a bound.
class GpuCache {
public:
    GpuCache() = default;
    ~GpuCache();
    GpuCache(const GpuCache &) = delete;
    GpuCache &operator=(const GpuCache &) = delete;
    bool open(const std::string &root, uint64_t budget, std::string &err);
    bool prepare(const GpuPlan &plan, std::vector<uint8_t> &missing, std::string &err);
    bool read(const GpuAsset &asset, std::vector<uint8_t> &out, std::string &err);
    bool publish(const GpuAsset &asset, const uint8_t *data, size_t bytes, std::string &err);
    // Explicit operator cleanup only: removes this format's unassigned assets and abandoned temps.
    // Call before prepare(), while this process has exclusive ownership. Other files are preserved.
    bool cleanup_unassigned(const GpuPlan &keep, uint64_t &removed_bytes, std::string &err);
    void close();
    const GpuAssetStats &stats() const { return stats_; }
    std::string key(const GpuAsset &asset) const;
private:
    bool inventory(std::string &err);
    bool load(const GpuAsset &asset, std::vector<uint8_t> &out, bool &hit, std::string &err);
    const GpuAsset *bound(const GpuAsset &asset) const;
    int root_ = -1, lock_ = -1;
    uint64_t budget_ = 0, reserved_ = 0, serial_ = 0;
    GpuPlan plan_;
    std::vector<uint8_t> missing_;
    GpuAssetStats stats_;
};
} // namespace strata::worker
