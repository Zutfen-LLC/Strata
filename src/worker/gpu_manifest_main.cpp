#include "strata/worker/gpu_assets.hpp"
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using namespace strata::worker;
bool assignments(const std::string &path, std::vector<std::pair<uint16_t, uint16_t>> &out, std::string &err) {
#ifdef _WIN32
    err = "selective GPU plans require POSIX"; return false;
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    struct stat st{};
    if (fd < 0 || ::fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0 || st.st_size > (1 << 20)) {
        if (fd >= 0) ::close(fd);
        err = "assignments must be a regular file of at most 1 MiB"; return false;
    }
    std::string text((size_t)st.st_size, '\0');
    size_t done = 0;
    while (done < text.size()) {
        const auto got = ::read(fd, &text[done], text.size() - done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) break;
        done += (size_t)got;
    }
    ::close(fd);
    if (done != text.size()) { err = "short assignment file"; return false; }
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream fields(line);
        std::string l, e, extra;
        if (!(fields >> l)) continue;
        if (!(fields >> e) || (fields >> extra)) { err = "assignment line requires LAYER EXPERT"; return false; }
        auto number = [](const std::string &value, uint16_t cap, uint16_t &out_value) {
            if (value.empty() || value.size() > 5) return false;
            unsigned v = 0;
            for (char c : value) { if (c < '0' || c > '9') return false; v = 10 * v + (unsigned)(c - '0'); }
            if (v >= cap) return false;
            out_value = (uint16_t)v; return true;
        };
        uint16_t layer = 0, expert = 0;
        if (!number(l, kMaxLayers, layer) || !number(e, kMaxExperts, expert) ||
            out.size() >= (size_t)kMaxLayers * kMaxExperts) {
            err = "assignment value or count is outside compiled bounds"; return false;
        }
        out.emplace_back(layer, expert);
    }
    return true;
#endif
}
} // namespace
int main(int argc, char **argv) {
    using namespace strata::worker;
    std::string err;
    if (argc == 5 && std::string(argv[1]) == "plan") {
        LocalManifest source;
        std::vector<std::pair<uint16_t, uint16_t>> assigned;
        GpuPlan plan;
        if (!read_local_manifest(argv[2], source, err) || !assignments(argv[3], assigned, err) ||
            !make_gpu_plan(source, assigned, plan, err) || !write_gpu_plan(argv[4], plan, err)) {
            std::fprintf(stderr, "gpu-manifest: %s\n", err.c_str()); return 1;
        }
        std::fprintf(stderr, "gpu-manifest: assigned=%zu assigned_bytes=%llu source_weight_read=0 "
                             "placement=%016llx%016llx; pin metadata with sha256sum\n", plan.assets.size(),
                     (unsigned long long)plan.assigned_bytes(), (unsigned long long)plan.placement_id.a,
                     (unsigned long long)plan.placement_id.b);
        return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "cleanup" && std::string(argv[5]) == "--confirm") {
        try {
            const std::string value(argv[4]);
            if (value.empty() || value.size() > 12 || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("invalid disk budget");
            const auto mib = std::stoull(value);
            if (!mib || mib > ((uint64_t)INT64_MAX >> 20)) throw std::runtime_error("disk budget overflow");
            GpuPlan keep;
            GpuCache cache;
            uint64_t removed = 0;
            if (!read_gpu_plan(argv[3], keep, err) || !cache.open(argv[2], mib << 20, err) ||
                !cache.cleanup_unassigned(keep, removed, err)) {
                std::fprintf(stderr, "gpu-manifest cleanup: %s\n", err.c_str()); return 1;
            }
            std::fprintf(stderr, "gpu-manifest cleanup: removed_bytes=%llu retained_disk_bytes=%llu\n",
                         (unsigned long long)removed, (unsigned long long)cache.stats().retained_disk_bytes);
            return 0;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "gpu-manifest cleanup: %s\n", e.what()); return 2;
        }
    }
    std::fprintf(stderr, "usage: strata-gpu-manifest plan LOCAL_MANIFEST ASSIGNMENTS OUTPUT\n"
                         "       strata-gpu-manifest cleanup CACHE_ROOT KEEP_PLAN DISK_BUDGET_MIB --confirm\n"
                         "ASSIGNMENTS: one zero-based LAYER EXPERT pair per line (# comments allowed).\n"
                         "The plan command reads only trusted metadata; it never scans source weights.\n"
                         "Cleanup is explicit and requires the worker to release its cache lock.\n");
    return 2;
}
