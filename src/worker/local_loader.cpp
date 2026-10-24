#include "local_loader.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

namespace strata::worker {
namespace {
uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

LocalLoadResult load_local_parallel(const LocalSource& source,
    const strata::kernels::cpu::NativeFmt& fmt, const std::vector<LocalLoadTask>& tasks,
    int threads, const std::function<bool()>& cancelled) {
    LocalLoadResult result;
    if (threads < 1 || threads > kMaxLocalLoadThreads || tasks.empty() || tasks.size() > kLocalBatchMax) {
        result.status = ST_INTERNAL;
        result.error = "parallel local batch outside compiled bounds";
        return result;
    }
    const int lanes = std::min<int>(threads, tasks.size());
    std::array<LocalLoadResult, kMaxLocalLoadThreads> outcomes;
    std::atomic<bool> failed{false};
    auto run = [&](int lane) {
        auto& out = outcomes[lane];
        auto refuse = [&](uint16_t status) {
            out.status = status;
            failed.store(true, std::memory_order_release);
        };
        auto stop = [&] {
            if (failed.load(std::memory_order_acquire)) return true;
            if (cancelled && cancelled()) { refuse(ST_DEADLINE); return true; }
            return false;
        };
        try {
            const auto& manifest = source.manifest();
            auto verify = [&](const uint8_t* data, size_t bytes, Digest128 want) {
                const auto start = now_ns();
                Digest128 digest{0xcbf29ce484222325ull, 0x9e3779b97f4a7c15ull};
                for (size_t i = 0; i < bytes;) {
                    if (stop()) { out.verify_ns += now_ns() - start; return false; }
                    const size_t end = std::min<size_t>(bytes, i + (64u << 10));
                    out.verified_bytes += end - i;
                    for (; i < end; ++i) {
                        digest.a = (digest.a ^ data[i]) * 0x100000001b3ull;
                        digest.b = (digest.b ^ data[bytes - 1 - i]) * 0x100000001b3ull;
                    }
                }
                out.verify_ns += now_ns() - start;
                if (digest != want) { refuse(ST_DIGEST_MISMATCH); return false; }
                return true;
            };
            // Contiguous expert ranges per lane; each expert is fully read before its own hash.
            // Another lane can read its next expert while this lane verifies resident bytes.
            for (size_t q = tasks.size() * lane / lanes; q < tasks.size() * (lane + 1) / lanes; ++q) {
                if (stop()) return;
                const auto& task = tasks[q];
                if (!task.dst || task.layer >= manifest.n_layers || task.expert >= manifest.n_expert) {
                    refuse(ST_INTERNAL); return;
                }
                const auto& ranges = manifest.layers[task.layer];
                size_t dst = 0;
                for (size_t r = 0; r < 3; ++r) {
                    const auto& range = ranges[r];
                    if (task.present[r]) for (size_t i = 0; i < range.bytes;) {
                        if (stop()) return;
                        const size_t bytes = std::min<size_t>(256u << 10, range.bytes - i);
                        const auto start = now_ns();
                        const bool ok = source.read_opened(range.file,
                            range.offset + (uint64_t)task.expert * range.bytes + i,
                            task.dst + dst + i, bytes, out.error);
                        out.read_ns += now_ns() - start;
                        if (!ok) { refuse(ST_LOCAL_INVALID); return; }
                        out.bytes += bytes;
                        i += bytes;
                    }
                    dst += range.bytes;
                }
                const size_t key = (size_t)task.layer * manifest.n_expert + task.expert;
                if (task.present[0] && task.present[1] && task.present[2]) {
                    if (!verify(task.dst, fmt.bytes, manifest.digests[key])) return;
                } else {
                    dst = 0;
                    for (size_t r = 0; r < 3; ++r) {
                        const size_t bytes = ranges[r].bytes;
                        if (task.present[r] && !verify(task.dst + dst, bytes, manifest.role_digests[key][r]))
                            return;
                        dst += bytes;
                    }
                }
                ++out.completed;
            }
        } catch (...) {
            // Includes read/cancellation callback exceptions. Never let a lane unwind past its thread.
            refuse(ST_INTERNAL);
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(lanes - 1); // allocation happens before any thread owns arena/file references
    bool launch_failed = false;
    try {
        for (int i = 1; i < lanes; ++i) workers.emplace_back(run, i);
    } catch (...) {
        launch_failed = true;
        failed.store(true, std::memory_order_release);
    }
    if (!launch_failed) run(0);
    for (auto& worker : workers) worker.join();
    // Only the serving caller merges counters and publishes slot readiness after every lane is reaped.
    for (int i = 0; i < lanes; ++i) {
        const auto& out = outcomes[i];
        result.bytes += out.bytes;
        result.verified_bytes += out.verified_bytes;
        result.read_ns += out.read_ns;
        result.verify_ns += out.verify_ns;
        result.completed += out.completed;
        if (result.status == ST_OK && out.status != ST_OK) {
            result.status = out.status;
            result.error = out.error;
        }
    }
    if (launch_failed || (result.status == ST_OK && result.completed != tasks.size()))
        result.status = ST_INTERNAL;
    return result;
}
} // namespace strata::worker
