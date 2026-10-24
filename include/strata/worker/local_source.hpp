// Operator-owned GGUF slice manifest. No filesystem path comes from the wire.
#pragma once
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/worker/expert_wire.hpp"
#include <array>
#include <functional>
namespace strata::worker {
inline constexpr uint32_t kLocalVersion = 1, kLocalBatchMax = 32;
inline constexpr int kMaxLocalLoadThreads = 4;
inline constexpr size_t kLocalManifestMax = 8u << 20;
struct LocalFile {
    std::string name;
    uint64_t bytes = 0;
};
struct LocalRange {
    uint32_t file = 0;
    uint64_t offset = 0, bytes = 0;
};
struct LocalManifest {
    uint16_t n_layers = 0, n_expert = 0, n_embd = 0, n_ff = 0, gu_type = 0, d_type = 0;
    Digest128 index_digest{}, identity{};
    std::vector<LocalFile> files;
    std::vector<std::array<LocalRange, 3>> layers;
    std::vector<Digest128> digests;                     // layer-major; digest of gate || up || down
    std::vector<std::array<Digest128, 3>> role_digests; // verifies present roles when another role is absent
    bool matches(const SetupManifest &setup) const;
};
bool read_local_manifest(const std::string &path, LocalManifest &out, std::string &err);
bool write_local_manifest(const std::string &path, const LocalManifest &m, std::string &err);
// Check the native index and file inventory without reading expert payloads.
bool match_local_index(const LocalManifest &m, const std::string &pack, const std::string &shard,
                       const strata::kernels::cpu::ExpertLayout &layout, std::string &err);
// One-time preparation uses one blob of staging and validates GGUF tensor headers.
bool prepare_local_manifest(const std::string &pack, const std::string &shard, uint16_t layers,
                            uint16_t experts, LocalManifest &out, std::string &err,
                            const std::function<bool()> &cancelled = {});
// Only regular basename files beneath the configured root can be opened.
// ENOENT alone is a miss. Other open, size and read errors are fatal.
class LocalSource {
  public:
    ~LocalSource();
    bool open(const std::string &root, const std::string &manifest, std::string &err);
    bool open(const std::string &root, const LocalManifest &manifest, std::string &err);
    const LocalManifest &manifest() const { return manifest_; }
    int file(uint32_t index, std::string &err); // >=0 descriptor, -1 absent, -2 invalid
    bool read(uint32_t file, uint64_t offset, void *dst, size_t bytes, std::string &err);
    // Parallel batch reads: file() must open/check every source on the serving thread first.
    // No file(), open() or close_files() may run until all readers join. Tables/stamps stay immutable.
    bool read_opened(uint32_t file, uint64_t offset, void *dst, size_t bytes, std::string &err) const;
    void close_files();

  private:
    LocalManifest manifest_;
    int root_ = -1;
    std::vector<int> files_;
    std::vector<std::array<uint64_t, 6>> stamps_;
};
} // namespace strata::worker
