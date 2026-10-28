#pragma once
// Private bounded framing and immutable-source guard for the resident CLI.
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::core::stdio_detail {
struct Request {
    size_t max_new = 0;
    std::vector<int32_t> tokens;
};
inline Request parse(const std::string &line, size_t context, size_t vocabulary) {
    if (line.size() > 4096 || line.compare(0, 4, "GEN ") != 0)
        throw std::runtime_error("expected GEN <max_new> <id,id,...> (<=4096 bytes)");
    size_t p = 4;
    auto number = [&](uint64_t cap) {
        if (p == line.size() || line[p] < '0' || line[p] > '9')
            throw std::runtime_error("expected unsigned decimal");
        uint64_t n = 0;
        do {
            const unsigned d = unsigned(line[p++] - '0');
            if (d > cap || n > (cap - d) / 10)
                throw std::runtime_error("request integer exceeds bound");
            n = n * 10 + d;
        } while (p < line.size() && line[p] >= '0' && line[p] <= '9');
        return n;
    };
    Request r;
    r.max_new = size_t(number(context));
    if (p == line.size() || line[p++] != ' ')
        throw std::runtime_error("missing prompt");
    for (;;) {
        if (r.tokens.size() == context || vocabulary == 0)
            throw std::runtime_error("prompt exceeds context");
        r.tokens.push_back(int32_t(number(vocabulary - 1)));
        if (p == line.size())
            break;
        if (line[p++] != ',')
            throw std::runtime_error("invalid token separator");
    }
    return r;
}
class Sources {
    struct Stamp {
        std::string path;
        uintmax_t size;
        std::filesystem::file_time_type time;
    };
    std::vector<Stamp> stamps;

  public:
    explicit Sources(const std::vector<std::string> &paths) {
        for (const auto &p : paths)
            stamps.push_back({p, std::filesystem::file_size(p), std::filesystem::last_write_time(p)});
    }
    void check() const {
        for (const auto &s : stamps)
            if (std::filesystem::file_size(s.path) != s.size || std::filesystem::last_write_time(s.path) != s.time)
                throw std::runtime_error("resident source changed; exit and reopen model: " + s.path);
    }
};
} // namespace strata::core::stdio_detail
