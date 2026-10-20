// Focused selective GGUF acquisition/cache fixture. Added under a no-execution authoring restriction;
// this source is not runtime evidence. Uses four synthetic Q8_0 blobs, never an external model.
#include "local_fixture.hpp"
#include "strata/worker/gpu_assets.hpp"

int main() {
    try {
        Fixture f;
        std::string err;
        const auto cache_root = f.root / "gpu-cache";
        fs::create_directory(cache_root);
        GpuPlan one, two;
        require(make_gpu_plan(f.m, {{0, 0}}, one, err), err);
        require(make_gpu_plan(f.m, {{0, 0}, {1, 1}}, two, err), err);
        require(write_gpu_plan((f.root / "gpu-one.plan").string(), one, err), err);
        require(write_gpu_plan((f.root / "gpu-two.plan").string(), two, err), err);
        std::ofstream(f.root / "gpu-assigned.txt") << "0 0\n1 1\n";
        check(!write_gpu_plan((f.root / "gpu-one.plan").string(), two, err), "plan publication preserves existing metadata");
        GpuPlan read;
        require(read_gpu_plan((f.root / "gpu-two.plan").string(), read, err), err);
        check(read.placement_id == two.placement_id && read.assets.size() == 2, "plan round trip preserves frozen identity");
        std::vector<uint8_t> encoded;
        require(encode_gpu_plan(two, encoded, err), err);
        encoded.push_back(0);
        check(!decode_gpu_plan(encoded.data(), encoded.size(), read, err), "plan refuses trailing bytes");
        auto stale = two;
        stale.assets[0].expert = 1;
        check(!validate_gpu_plan(stale, err), "changed assignment refuses stale placement identity");
        check(!make_gpu_plan(f.m, {{0, 0}, {0, 0}}, read, err), "duplicate assignment refused");
        check(!make_gpu_plan(f.m, {{2, 0}}, read, err), "out-of-range assignment refused");
        check(!make_gpu_plan(f.m, {}, read, err), "empty GPU assignment refused");

        GpuAssetSource source;
        require(source.open(f.root.string(), f.m, two, err), err);
        GpuCache cache;
        const auto budget = 6 * f.fmt.bytes + 4096;
        require(cache.open(cache_root.string(), budget, err), err);
        GpuCache contender;
        check(!contender.open(cache_root.string(), budget, err), "second cache writer refused by exclusive lock");
        std::vector<uint8_t> missing, staging;
        require(cache.prepare(one, missing, err), err);
        check(missing == std::vector<uint8_t>{1}, "cold setup requests assigned expert only");
        require(source.read(one.assets[0], staging, err), err);
        check(staging == f.blobs[0] && source.stats().source_read_bytes == f.fmt.bytes,
              "selective reader reads only three ranges of one expert, no layer/shard");
        require(cache.publish(one.assets[0], staging.data(), staging.size(), err), err);
        check(!cache.publish(one.assets[0], staging.data(), staging.size(), err), "duplicate publication refused");
        const auto first_key = cache.key(one.assets[0]);
        check(cache.stats().persisted_bytes == f.fmt.bytes && cache.stats().cached_bytes == f.fmt.bytes,
              "cache reports actual assigned and persisted payload separately");
        cache.close();

        require(cache.open(cache_root.string(), budget, err), err);
        require(cache.prepare(one, missing, err), err);
        check(missing == std::vector<uint8_t>{0} && cache.stats().reused_bytes == f.fmt.bytes &&
                  cache.stats().transferred_bytes == 0, "restart rehashes and reuses verified assigned blob without transfer");
        require(cache.prepare(two, missing, err), err);
        check(missing == std::vector<uint8_t>({0, 1}) && cache.key(two.assets[0]) == first_key,
              "assignment addition keeps existing asset key and fetches only missing addition");
        const auto before = source.stats().source_read_bytes;
        require(source.read(two.assets[1], staging, err), err);
        require(cache.publish(two.assets[1], staging.data(), staging.size(), err), err);
        check(source.stats().source_read_bytes - before == f.fmt.bytes && staging == f.blobs[3],
              "delta acquisition reads one extra expert only");
        const auto second_key = cache.key(two.assets[1]);
        cache.close();

        // Incomplete final files and orphaned temporary files are never cache hits. They stay charged.
        std::ofstream(cache_root / first_key, std::ios::binary | std::ios::trunc) << "partial";
        std::ofstream(cache_root / ".sgpu1-tmp-123-9", std::ios::binary) << "abandoned";
        std::ofstream(cache_root / "operator-note", std::ios::binary) << "preserve me";
        require(cache.open(cache_root.string(), budget, err), err);
        require(cache.prepare(two, missing, err), err);
        check(missing == std::vector<uint8_t>({1, 0}), "incomplete cache asset is missing while unrelated verified expert remains reusable");
        require(source.read(two.assets[0], staging, err), err);
        check(!cache.publish(two.assets[0], staging.data(), staging.size() - 1, err), "short upload refused before publication");
        staging[0] ^= 1;
        check(!cache.publish(two.assets[0], staging.data(), staging.size(), err), "wrong digest refused before publication");
        staging[0] ^= 1;
        require(cache.publish(two.assets[0], staging.data(), staging.size(), err), err);
        cache.close();

        // A complete but corrupt blob is also a miss; repair is atomic and does not delete other assets.
        {
            std::fstream corrupt(cache_root / first_key, std::ios::binary | std::ios::in | std::ios::out);
            char c = 0; corrupt.get(c); corrupt.seekp(0); corrupt.put(c ^ 1);
        }
        require(cache.open(cache_root.string(), budget, err), err);
        require(cache.prepare(two, missing, err), err);
        check(missing == std::vector<uint8_t>({1, 0}), "complete corrupt cache asset is not accepted");
        require(source.read_into(two.assets[0], staging.data(), staging.size(), err), err);
        require(cache.publish(two.assets[0], staging.data(), staging.size(), err), err);
        require(cache.read(two.assets[0], staging, err), err);
        check(staging == f.blobs[0] && fs::exists(cache_root / second_key), "atomic repair preserves other cached experts");
        check(cache.stats().host_staging_peak <= f.fmt.bytes && source.stats().host_staging_peak <= f.fmt.bytes,
              "each acquisition stage remains bounded by one expert blob");
        cache.close();

        // Existing bytes count independently of VRAM and cannot be evicted implicitly for a new plan.
        GpuPlan other;
        require(make_gpu_plan(f.m, {{0, 1}}, other, err), err);
        const auto occupied = fs::file_size(cache_root / first_key) + fs::file_size(cache_root / second_key) +
                              fs::file_size(cache_root / ".sgpu1-tmp-123-9") + fs::file_size(cache_root / "operator-note");
        require(cache.open(cache_root.string(), occupied + f.fmt.bytes - 1, err), err);
        check(!cache.prepare(other, missing, err), "cache additions refuse before transfer when retained bytes consume disk budget");
        check(fs::exists(cache_root / first_key) && fs::exists(cache_root / second_key), "capacity refusal preserves retained assets");
        cache.close();

        // Cleanup is explicit, format-specific, and requires inactive cache ownership.
        require(cache.open(cache_root.string(), budget, err), err);
        uint64_t removed = 0;
        require(cache.cleanup_unassigned(one, removed, err), err);
        check(removed == f.fmt.bytes + 9 && fs::exists(cache_root / first_key) &&
                  !fs::exists(cache_root / second_key) && fs::exists(cache_root / "operator-note"),
              "explicit cleanup removes unassigned own assets/temps and preserves assigned/operator files");
        require(cache.prepare(one, missing, err), err);
        check(!cache.cleanup_unassigned(one, removed, err), "cleanup refused for an active frozen cache binding");
        cache.close();

        // Symlinks never become inputs or cache destinations; paths are generated from identities.
        fs::remove(cache_root / first_key);
        fs::create_symlink(f.root / "local.manifest", cache_root / first_key);
        check(!cache.open(cache_root.string(), budget, err), "cache refuses symlink containment violation");
        fs::remove(cache_root / first_key);
        const auto before_unassigned = source.stats().source_read_bytes;
        GpuAsset unassigned{0, 1, f.fmt.bytes, blob_digest(f.blobs[1])};
        check(!source.read(unassigned, staging, err) && source.stats().source_read_bytes == before_unassigned,
              "unassigned source read is refused without reading weights");

        if (std::getenv("STRATA_WORKER_FIXTURE_KEEP"))
            std::fprintf(stderr, "GPU selective fixtures: manifest=%s plan=%s assignment=%s cache=%s\n",
                         (f.root / "local.manifest").c_str(), (f.root / "gpu-two.plan").c_str(),
                         (f.root / "gpu-assigned.txt").c_str(), cache_root.c_str());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL gpu_assets_test: %s\n", e.what()); return 1;
    }
    return failures ? 1 : 0;
}
