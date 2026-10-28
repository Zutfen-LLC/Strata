// Bounded serial-request diagnostic. Native little-endian int64 headers, 4-byte payloads.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace strata::core {
class FirstRequestTrace {
    std::FILE* file_ = nullptr;
    bool begun_ = false, active_ = false, failed_ = false;
    size_t bytes_ = 0;
    int64_t next_position_ = 0;
public:
    static constexpr size_t limit = 64u << 20;
    int64_t window = -1, position = 0, slice = 0, prompt_count = 0;
    static bool enabled() {
        static const bool on = [] { const char* p = std::getenv("STRATA_FIRST_REQUEST_TRACE"); return p && *p; }();
        return on;
    }
    ~FirstRequestTrace() { if (file_) std::fclose(file_); }
    bool active() const { return active_; }
    bool good() const { return !failed_; }
    bool record(int64_t kind, int64_t layer, int64_t tb, int64_t rows, int64_t cols, const void* data) {
        if (!active_) return !failed_;
        if (failed_ || rows < 0 || cols < 0 || rows > 65536 || cols > (int64_t)(limit/4) ||
            (uint64_t) rows * cols > (limit - sizeof(int64_t)*8)/4) return !(failed_ = true);
        const size_t size = (size_t)rows * (size_t)cols * 4;
        if (bytes_ > limit - size - sizeof(int64_t)*8) return !(failed_ = true);
        const int64_t h[] = {kind, window, position, layer, tb, rows, cols, (int64_t)size};
        if (!file_ || std::fwrite(h, sizeof h, 1, file_) != 1 ||
            (size && std::fwrite(data, size, 1, file_) != 1)) return !(failed_ = true);
        bytes_ += sizeof h + size;
        return true;
    }
    bool begin(const int32_t* ids, size_t count) {
        if (!enabled() || begun_) return !failed_;
        begun_ = true;
        if (count > 64) return !(failed_ = true);
        // Exclusive creation prevents an accidental rerun from replacing evidence.
        file_ = std::fopen(std::getenv("STRATA_FIRST_REQUEST_TRACE"), "wbx");
        if (!file_) return !(failed_ = true);
#ifndef _WIN32
        if (fchmod(fileno(file_), 0600) != 0) return !(failed_ = true);
#endif
        prompt_count = (int64_t)count;
        active_ = true;
        const char magic[8] = {'S','T','R','F','T','0','1','\0'};
        if (std::fwrite(magic, sizeof magic, 1, file_) != 1) return !(failed_ = true);
        bytes_ = sizeof magic;
        return record(0, -1, 0, 1, (int64_t)count, ids);
    }
    bool begin(const int64_t* ids, size_t count) {
        if (!enabled() || begun_) return !failed_;
        if (count > 64) return !(failed_ = true);
        int32_t compact[64];
        for (size_t i=0; i<count; ++i) compact[i] = (int32_t)ids[i];
        return begin(compact, count);
    }
    bool start_window(int64_t pos, int T, const int32_t* ids) {
        if (!active_) return !failed_;
        if (++window >= 32 || pos != next_position_ || T < 1 || pos >= prompt_count)
            return !(failed_ = true);
        next_position_ = pos + T;
        position = pos; slice = 0;
        return record(1, -1, 0, 1, T, ids);
    }
    bool finish(const float* logits, size_t count) {
        if (!active_ || next_position_ < prompt_count) return false;
        const bool ok = record(11, -1, 0, 1, (int64_t)count, logits);
        const bool closed = std::fclose(file_) == 0;
        file_ = nullptr; active_ = false;
        failed_ = failed_ || !ok || !closed;
        return !failed_;
    }
};
inline FirstRequestTrace& first_request_trace() { static FirstRequestTrace trace; return trace; }
} // namespace strata::core
