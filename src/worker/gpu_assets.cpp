#include "strata/worker/gpu_assets.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::worker {
namespace {
void put(std::vector<uint8_t> &p, uint64_t v, size_t n) {
    for (size_t i = 0; i < n; ++i) p.push_back((uint8_t)(v >> (i * 8)));
}
void digest_put(std::vector<uint8_t> &p, Digest128 d) { put(p, d.a, 8); put(p, d.b, 8); }
struct Cursor {
    const uint8_t *p;
    size_t n, i = 0;
    bool ok = true;
    uint64_t get(size_t count) {
        if (!ok || count > n - i) { ok = false; return 0; }
        uint64_t v = 0;
        for (size_t j = 0; j < count; ++j) v |= (uint64_t)p[i++] << (j * 8);
        return v;
    }
    Digest128 digest() { const auto a = get(8); const auto b = get(8); return {a, b}; }
};
bool same(const GpuAsset &a, const GpuAsset &b) {
    return a.layer == b.layer && a.expert == b.expert && a.bytes == b.bytes && a.digest == b.digest;
}
void plan_bytes(const GpuPlan &p, std::vector<uint8_t> &out) {
    out = {'S','G','P','U','P','L','N','1'};
    put(out, p.model_id, 8);
    for (auto v : {p.n_layers, p.n_expert, p.n_embd, p.n_ff, p.gu_type, p.d_type}) put(out, v, 2);
    digest_put(out, p.layout_id);
    put(out, p.assets.size(), 4);
    for (const auto &a : p.assets) {
        put(out, a.layer, 2); put(out, a.expert, 2); put(out, a.bytes, 8); digest_put(out, a.digest);
    }
}
bool geometry(const GpuPlan &p, strata::kernels::cpu::NativeFmt &f, std::string &err) {
    if (!p.model_id || !p.layout_id.a || !p.n_layers || p.n_layers > kMaxLayers || !p.n_expert ||
        p.n_expert > kMaxExperts || !p.n_embd || p.n_embd > kMaxNEmbd || !p.n_ff || p.n_ff > kMaxNFf ||
        p.gu_type != 8 || p.d_type != 8 || p.assets.empty() ||
        p.assets.size() > (size_t)p.n_layers * p.n_expert) {
        err = "CUDA asset plan requires bounded plain Q8_0 geometry and a nonempty assigned set";
        return false;
    }
    if (!strata::kernels::cpu::native_fmt(p.gu_type, p.d_type, p.n_embd, p.n_ff, f, err)) return false;
    if (!f.bytes || f.bytes > kGpuBlobMax || f.tail_off || f.had_block) {
        err = "CUDA asset blob layout exceeds the bounded plain layout";
        return false;
    }
    return true;
}
std::string hex(uint64_t v, int width) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
    return std::string(b + 16 - width, (size_t)width);
}
std::string asset_key(const GpuPlan &p, const GpuAsset &a) {
    return "sgpu1-" + hex(p.model_id, 16) + "-" + hex(p.layout_id.a, 16) + hex(p.layout_id.b, 16) +
           "-" + hex(a.layer, 4) + "-" + hex(a.expert, 4) + "-" + hex(a.bytes, 16) + "-" +
           hex(a.digest.a, 16) + hex(a.digest.b, 16) + ".blob";
}
bool owned_asset_name(const std::string &name) {
    // Exactly the generated fixed-width basename; cleanup never follows a prefix alone.
    if (name.size() != 120 || name.compare(0, 6, "sgpu1-") || name.compare(115, 5, ".blob")) return false;
    const std::set<size_t> separators{22, 55, 60, 65, 82};
    for (size_t i = 6; i < 115; ++i) {
        if (separators.count(i)) { if (name[i] != '-') return false; }
        else if (!((name[i] >= '0' && name[i] <= '9') || (name[i] >= 'a' && name[i] <= 'f'))) return false;
    }
    return true;
}
bool temp_name(const std::string &name) {
    constexpr const char *prefix = ".sgpu1-tmp-";
    if (name.compare(0, 11, prefix) || name.size() <= 13) return false;
    bool separator = false;
    for (size_t i = 11; i < name.size(); ++i) {
        if (name[i] == '-' && !separator && i > 11 && i + 1 < name.size()) separator = true;
        else if (name[i] < '0' || name[i] > '9') return false;
    }
    return separator;
}
#ifndef _WIN32
bool read_exact(int fd, uint8_t *p, size_t n) {
    size_t done = 0;
    while (done < n) {
        const auto got = ::pread(fd, p + done, n - done, (off_t)done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        done += (size_t)got;
    }
    return true;
}
bool write_exact(int fd, const uint8_t *p, size_t n) {
    size_t done = 0;
    while (done < n) {
        const auto got = ::write(fd, p + done, n - done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        done += (size_t)got;
    }
    return true;
}
bool regular(const struct stat &st) { return S_ISREG(st.st_mode) && st.st_size >= 0 && st.st_nlink == 1; }
#endif
} // namespace

const GpuAsset *GpuPlan::find(int l, int e) const {
    if (l < 0 || e < 0 || l >= n_layers || e >= n_expert) return nullptr;
    const auto it = std::lower_bound(assets.begin(), assets.end(), std::make_pair(l, e),
        [](const GpuAsset &a, const std::pair<int, int> &b) { return std::make_pair((int)a.layer, (int)a.expert) < b; });
    return it != assets.end() && it->layer == l && it->expert == e ? &*it : nullptr;
}
bool GpuPlan::owns(int l, int e) const { return find(l, e) != nullptr; }
uint64_t GpuPlan::assigned_bytes() const {
    uint64_t bytes = 0;
    for (const auto &a : assets) {
        if (a.bytes > UINT64_MAX - bytes) return UINT64_MAX;
        bytes += a.bytes;
    }
    return bytes;
}
bool validate_gpu_plan(const GpuPlan &p, std::string &err) {
    strata::kernels::cpu::NativeFmt f;
    if (!geometry(p, f, err)) return false;
    int previous = -1;
    for (const auto &a : p.assets) {
        const int slot = (int)a.layer * p.n_expert + a.expert;
        if (a.layer >= p.n_layers || a.expert >= p.n_expert || a.bytes != f.bytes || slot <= previous) {
            err = "CUDA asset plan has duplicate, unordered, out-of-range or wrong-length assets";
            return false;
        }
        previous = slot;
    }
    std::vector<uint8_t> bytes;
    plan_bytes(p, bytes);
    if (bytes.size() + 16 > kGpuPlanMax || blob_digest(bytes) != p.placement_id) {
        err = "CUDA placement identity does not match its immutable asset list";
        return false;
    }
    return true;
}
bool encode_gpu_plan(const GpuPlan &p, std::vector<uint8_t> &out, std::string &err) {
    if (!validate_gpu_plan(p, err)) return false;
    plan_bytes(p, out); digest_put(out, p.placement_id); return true;
}
bool decode_gpu_plan(const uint8_t *data, size_t bytes, GpuPlan &out, std::string &err) {
    if (!data || bytes < 64 || bytes > kGpuPlanMax || std::memcmp(data, "SGPUPLN1", 8)) {
        err = "unsupported or overlong CUDA placement metadata"; return false;
    }
    Cursor c{data, bytes}; c.i = 8;
    GpuPlan p;
    p.model_id = c.get(8);
    p.n_layers = c.get(2); p.n_expert = c.get(2); p.n_embd = c.get(2); p.n_ff = c.get(2);
    p.gu_type = c.get(2); p.d_type = c.get(2); p.layout_id = c.digest();
    const auto count = c.get(4);
    if (!c.ok || !count || count > (uint64_t)kMaxLayers * kMaxExperts ||
        count > (bytes - c.i) / 28 || bytes - c.i != count * 28 + 16) {
        err = "malformed CUDA asset count or metadata length"; return false;
    }
    p.assets.resize((size_t)count);
    for (auto &a : p.assets) { a.layer = c.get(2); a.expert = c.get(2); a.bytes = c.get(8); a.digest = c.digest(); }
    p.placement_id = c.digest();
    if (!c.ok || c.i != bytes || !validate_gpu_plan(p, err)) return false;
    out = std::move(p); return true;
}
bool read_gpu_plan(const std::string &path, GpuPlan &out, std::string &err) {
#ifdef _WIN32
    err = "selective CUDA asset metadata requires POSIX"; return false;
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    struct stat st{};
    if (fd < 0 || ::fstat(fd, &st) || !regular(st) || (uint64_t)st.st_size > kGpuPlanMax) {
        if (fd >= 0) ::close(fd);
        err = "cannot read bounded regular CUDA placement metadata"; return false;
    }
    std::vector<uint8_t> bytes((size_t)st.st_size);
    const bool ok = read_exact(fd, bytes.data(), bytes.size());
    ::close(fd);
    if (!ok) { err = "short CUDA placement metadata"; return false; }
    return decode_gpu_plan(bytes.data(), bytes.size(), out, err);
#endif
}
bool write_gpu_plan(const std::string &path, const GpuPlan &p, std::string &err) {
    std::vector<uint8_t> bytes;
    if (!encode_gpu_plan(p, bytes, err)) return false;
#ifdef _WIN32
    err = "selective CUDA asset metadata requires POSIX"; return false;
#else
    const std::filesystem::path file(path);
    const auto parent = file.parent_path().empty() ? std::string(".") : file.parent_path().string();
    const auto name = file.filename().string();
    if (name.empty() || name == "." || name == "..") { err = "invalid CUDA plan output name"; return false; }
    const int dir = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (dir < 0) { err = "cannot open CUDA plan output directory"; return false; }
    const auto temp = ".sgpu1-tmp-" + std::to_string((unsigned long long)::getpid()) + "-0";
    const int fd = ::openat(dir, temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { ::close(dir); err = "CUDA plan temporary file already exists or cannot be created"; return false; }
    bool ok = write_exact(fd, bytes.data(), bytes.size()) && ::fsync(fd) == 0;
    ::close(fd);
    // linkat refuses an existing destination; preserve previous plans and operator files.
    if (ok) ok = ::linkat(dir, temp.c_str(), dir, name.c_str(), 0) == 0;
    ::unlinkat(dir, temp.c_str(), 0);
    if (ok) ok = ::fsync(dir) == 0;
    ::close(dir);
    if (!ok) err = "cannot publish CUDA plan without replacing an existing file";
    return ok;
#endif
}
bool make_gpu_plan(const LocalManifest &source, const std::vector<std::pair<uint16_t, uint16_t>> &assigned,
                   GpuPlan &out, std::string &err) {
    GpuPlan p;
    p.model_id = source.identity.a; p.n_layers = source.n_layers; p.n_expert = source.n_expert;
    p.n_embd = source.n_embd; p.n_ff = source.n_ff; p.gu_type = source.gu_type; p.d_type = source.d_type;
    std::vector<uint8_t> layout{'Q','8','_','0','-','G','U','D','-','1'};
    digest_put(layout, source.identity); digest_put(layout, source.index_digest);
    for (auto v : {p.n_layers, p.n_expert, p.n_embd, p.n_ff, p.gu_type, p.d_type}) put(layout, v, 2);
    p.layout_id = blob_digest(layout);
    strata::kernels::cpu::NativeFmt f;
    // Include one temporary entry because geometry() also bounds the assigned count.
    p.assets.resize(1);
    if (!geometry(p, f, err) || source.layers.size() != p.n_layers ||
        source.digests.size() != (size_t)p.n_layers * p.n_expert) {
        err = "source metadata does not describe the supported selective Q8_0 layout"; return false;
    }
    p.assets.clear();
    auto ordered = assigned;
    std::sort(ordered.begin(), ordered.end());
    for (const auto &s : ordered) {
        if (s.first >= p.n_layers || s.second >= p.n_expert ||
            (!p.assets.empty() && p.assets.back().layer == s.first && p.assets.back().expert == s.second)) {
            err = "selective assignment has duplicate or out-of-range experts"; return false;
        }
        p.assets.push_back({s.first, s.second, f.bytes, source.digests[(size_t)s.first * p.n_expert + s.second]});
    }
    std::vector<uint8_t> content;
    plan_bytes(p, content); p.placement_id = blob_digest(content);
    if (!validate_gpu_plan(p, err)) return false;
    out = std::move(p); return true;
}

bool GpuAssetSource::open(const std::string &root, const LocalManifest &m, const GpuPlan &p, std::string &err) {
    if (!validate_gpu_plan(p, err)) return false;
    std::vector<std::pair<uint16_t, uint16_t>> assigned;
    for (const auto &a : p.assets) assigned.emplace_back(a.layer, a.expert);
    GpuPlan expected;
    if (!make_gpu_plan(m, assigned, expected, err) || expected.placement_id != p.placement_id ||
        !source_.open(root, m, err)) {
        if (err.empty()) err = "selective source does not match the frozen GPU plan";
        return false;
    }
    plan_ = p; stats_ = {}; stats_.assigned_bytes = p.assigned_bytes(); return true;
}
bool GpuAssetSource::read(const GpuAsset &a, std::vector<uint8_t> &out, std::string &err) {
    if (a.bytes > kGpuBlobMax) { err = "source blob exceeds bounded staging"; return false; }
    out.resize((size_t)a.bytes);
    return read_into(a, out.data(), out.size(), err);
}
bool GpuAssetSource::read_into(const GpuAsset &a, uint8_t *out, size_t bytes, std::string &err) {
    const auto *bound = plan_.find(a.layer, a.expert);
    if (!bound || !same(a, *bound) || !out || bytes != a.bytes) {
        err = "source read outside the frozen GPU assignment or wrong destination length"; return false;
    }
    stats_.host_staging_peak = std::max<uint64_t>(stats_.host_staging_peak, a.bytes);
    const auto &m = source_.manifest();
    size_t dst = 0;
    for (const auto &range : m.layers[a.layer]) {
        if (dst > bytes || range.bytes > bytes - dst ||
            !source_.read(range.file, range.offset + (uint64_t)a.expert * range.bytes,
                          out + dst, (size_t)range.bytes, err)) return false;
        stats_.source_read_bytes += range.bytes;
        dst += (size_t)range.bytes;
    }
    if (dst != bytes || blob_digest(out, bytes) != a.digest) {
        err = "selective source expert failed expected length/digest"; return false;
    }
    stats_.verified_bytes += a.bytes; return true;
}

GpuCache::~GpuCache() { close(); }
void GpuCache::close() {
#ifndef _WIN32
    if (lock_ >= 0) ::close(lock_);
    if (root_ >= 0) ::close(root_);
#endif
    root_ = lock_ = -1; budget_ = reserved_ = serial_ = 0; plan_ = {}; missing_.clear(); stats_ = {};
}
bool GpuCache::inventory(std::string &err) {
#ifdef _WIN32
    err = "selective GPU cache requires POSIX"; return false;
#else
    // A fresh directory descriptor avoids sharing the scan offset with earlier scans.
    const int scanfd = ::openat(root_, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    DIR *dir = scanfd < 0 ? nullptr : ::fdopendir(scanfd);
    if (!dir) { if (scanfd >= 0) ::close(scanfd); err = "cannot scan CUDA cache"; return false; }
    uint64_t total = 0;
    bool ok = true;
    errno = 0;
    while (const auto *entry = ::readdir(dir)) {
        const std::string name(entry->d_name);
        if (name == "." || name == "..") continue;
        struct stat st{};
        if (::fstatat(root_, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) || !regular(st) ||
            (uint64_t)st.st_size > budget_ - std::min(budget_, total)) { ok = false; break; }
        total += (uint64_t)st.st_size;
    }
    if (errno) ok = false;
    ::closedir(dir);
    if (!ok || total > budget_) { err = "CUDA cache contains unsafe entries or exceeds its independent disk budget"; return false; }
    stats_.retained_disk_bytes = total; return true;
#endif
}
bool GpuCache::open(const std::string &root, uint64_t budget, std::string &err) {
    close();
#ifdef _WIN32
    err = "selective GPU cache requires POSIX"; return false;
#else
    if (!budget || budget > INT64_MAX) { err = "CUDA cache needs a positive bounded disk budget"; return false; }
    root_ = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (root_ < 0) { err = "CUDA cache root must already be an operator-created directory"; return false; }
    lock_ = ::openat(root_, ".sgpu1-lock", O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK, 0600);
    struct stat st{};
    if (lock_ < 0 || ::fstat(lock_, &st) || !regular(st) || st.st_size != 0 ||
        ::flock(lock_, LOCK_EX | LOCK_NB)) {
        err = "CUDA cache is unsafe or owned by another process"; close(); return false;
    }
    budget_ = budget;
    if (!inventory(err)) { close(); return false; }
    return true;
#endif
}
const GpuAsset *GpuCache::bound(const GpuAsset &a) const {
    const auto *b = plan_.find(a.layer, a.expert);
    return b && same(a, *b) ? b : nullptr;
}
std::string GpuCache::key(const GpuAsset &a) const { return bound(a) ? asset_key(plan_, a) : std::string(); }
bool GpuCache::load(const GpuAsset &a, std::vector<uint8_t> &out, bool &hit, std::string &err) {
    hit = false;
#ifdef _WIN32
    err = "selective GPU cache requires POSIX"; return false;
#else
    const auto name = key(a);
    if (root_ < 0 || name.empty()) { err = "cache access outside the frozen GPU assignment"; return false; }
    const int fd = ::openat(root_, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) {
        if (errno == ENOENT) return true;
        err = "cannot open regular assigned CUDA cache blob"; return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) || !regular(st)) {
        ::close(fd); err = "unsafe assigned CUDA cache entry"; return false;
    }
    if ((uint64_t)st.st_size != a.bytes) { ::close(fd); return true; }
    out.resize((size_t)a.bytes);
    stats_.host_staging_peak = std::max(stats_.host_staging_peak, a.bytes);
    const bool ok = read_exact(fd, out.data(), out.size());
    ::close(fd);
    if (!ok) { err = "failed assigned CUDA cache read"; return false; }
    stats_.verified_bytes += a.bytes;
    hit = blob_digest(out) == a.digest;
    return true;
#endif
}
bool GpuCache::prepare(const GpuPlan &p, std::vector<uint8_t> &missing, std::string &err) {
    if (root_ < 0) { err = "CUDA cache is not open"; return false; }
    if (!validate_gpu_plan(p, err) || !inventory(err)) return false;
    plan_ = p; missing_.assign(p.assets.size(), 1); reserved_ = 0;
    stats_.assigned_bytes = p.assigned_bytes(); stats_.cached_bytes = stats_.reused_bytes = 0;
    std::vector<uint8_t> staging;
    for (size_t i = 0; i < p.assets.size(); ++i) {
        bool hit = false;
        if (!load(p.assets[i], staging, hit, err)) return false;
        missing_[i] = hit ? 0 : 1;
        if (hit) { stats_.cached_bytes += p.assets[i].bytes; stats_.reused_bytes += p.assets[i].bytes; }
        else {
            if (p.assets[i].bytes > budget_ - stats_.retained_disk_bytes - reserved_) {
                err = "assigned CUDA cache additions exceed disk budget; explicit cleanup is required";
                return false;
            }
            reserved_ += p.assets[i].bytes;
        }
    }
    missing = missing_; return true;
}
bool GpuCache::read(const GpuAsset &a, std::vector<uint8_t> &out, std::string &err) {
    bool hit = false;
    if (!bound(a)) { err = "cache read outside the frozen GPU assignment"; return false; }
    if (!load(a, out, hit, err)) return false;
    if (!hit) { err = "assigned CUDA cache blob is absent, incomplete or corrupt"; return false; }
    return true;
}
bool GpuCache::publish(const GpuAsset &a, const uint8_t *data, size_t bytes, std::string &err) {
#ifdef _WIN32
    err = "selective GPU cache requires POSIX"; return false;
#else
    const auto *b = bound(a);
    if (!b || root_ < 0 || !data || bytes != a.bytes || blob_digest(data, bytes) != a.digest) {
        err = "CUDA cache publication rejected length, digest or assignment"; return false;
    }
    const size_t index = (size_t)(b - plan_.assets.data());
    if (!missing_[index]) { err = "duplicate CUDA cache publication"; return false; }
    if (!inventory(err) || reserved_ > budget_ - stats_.retained_disk_bytes) {
        err = "CUDA cache changed or exceeded reserved disk budget"; return false;
    }
    const auto name = key(a);
    struct stat old{};
    uint64_t old_bytes = 0;
    if (::fstatat(root_, name.c_str(), &old, AT_SYMLINK_NOFOLLOW) == 0) {
        if (!regular(old)) { err = "unsafe CUDA cache replacement"; return false; }
        old_bytes = (uint64_t)old.st_size;
    } else if (errno != ENOENT) { err = "cannot inspect CUDA cache replacement"; return false; }
    const auto temp = ".sgpu1-tmp-" + std::to_string((unsigned long long)::getpid()) + "-" + std::to_string(++serial_);
    const int fd = ::openat(root_, temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { err = "cannot create exclusive CUDA cache staging file"; return false; }
    bool ok = write_exact(fd, data, bytes) && ::fsync(fd) == 0;
    ::close(fd);
    if (ok) ok = ::renameat(root_, temp.c_str(), root_, name.c_str()) == 0;
    if (!ok) {
        ::unlinkat(root_, temp.c_str(), 0);
        err = "cannot atomically publish CUDA cache blob"; return false;
    }
    // A failure here is an uncertain durability result; the next setup must reverify the final blob.
    if (::fsync(root_)) { err = "CUDA cache publication durability could not be confirmed"; return false; }
    missing_[index] = 0; reserved_ -= a.bytes;
    stats_.cached_bytes += a.bytes; stats_.persisted_bytes += a.bytes; stats_.transferred_bytes += a.bytes;
    stats_.verified_bytes += a.bytes;
    stats_.retained_disk_bytes += a.bytes;
    stats_.retained_disk_bytes -= old_bytes;
    stats_.host_staging_peak = std::max(stats_.host_staging_peak, a.bytes);
    return true;
#endif
}
bool GpuCache::cleanup_unassigned(const GpuPlan &keep, uint64_t &removed_bytes, std::string &err) {
    removed_bytes = 0;
#ifdef _WIN32
    err = "selective GPU cache requires POSIX"; return false;
#else
    if (root_ < 0 || !plan_.assets.empty() || !validate_gpu_plan(keep, err)) {
        err = "explicit CUDA cleanup requires an open, inactive cache and a valid keep plan"; return false;
    }
    std::set<std::string> retained;
    for (const auto &a : keep.assets) retained.insert(asset_key(keep, a));
    const int scanfd = ::openat(root_, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    DIR *dir = scanfd < 0 ? nullptr : ::fdopendir(scanfd);
    if (!dir) { if (scanfd >= 0) ::close(scanfd); err = "cannot scan CUDA cache for explicit cleanup"; return false; }
    std::vector<std::pair<std::string, uint64_t>> remove;
    bool ok = true;
    errno = 0;
    while (const auto *entry = ::readdir(dir)) {
        const std::string name(entry->d_name);
        if ((!owned_asset_name(name) && !temp_name(name)) || retained.count(name)) continue;
        struct stat st{};
        if (::fstatat(root_, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) || !regular(st)) { ok = false; break; }
        remove.emplace_back(name, (uint64_t)st.st_size);
    }
    if (errno) ok = false;
    ::closedir(dir);
    if (ok) for (const auto &item : remove) {
        if (::unlinkat(root_, item.first.c_str(), 0)) { ok = false; break; }
        removed_bytes += item.second;
    }
    if (!ok || ::fsync(root_)) { err = "explicit CUDA cache cleanup stopped on an error"; return false; }
    return inventory(err);
#endif
}
} // namespace strata::worker
