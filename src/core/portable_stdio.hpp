#pragma once
// Private bounded framing and immutable-source guard for the resident CLI.
#include <cstdint>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace strata::core::stdio_detail {
// GEN, a context-sized max_new, a space, and at most context signed-int32
// token IDs in unsigned decimal with commas. Legacy contexts retain 4096.
inline size_t request_limit(size_t context) {
    if (context == 0 || context > 4096) throw std::runtime_error("invalid resident context");
    const size_t max_digits = 10; // INT32_MAX
    const size_t derived = 4 + std::to_string(context).size() + 1 + context * max_digits + context - 1;
    return std::max(size_t(4096), derived);
}
class LineFramer {
    const size_t limit_;
    std::string line_;
  public:
    explicit LineFramer(size_t context) : limit_(request_limit(context)) {}
    bool append(char c) {
        if(c=='\n') return true;
        if(line_.size()>=limit_) throw std::runtime_error("resident request line exceeds context limit");
        line_.push_back(c);return false;
    }
    bool eof() const {
        if(!line_.empty()) throw std::runtime_error("incomplete request at EOF");
        return false;
    }
    std::string take() { return std::move(line_); }
};
struct Request {
    size_t max_new = 0;
    std::vector<int32_t> tokens;
};
inline Request parse(const std::string &line, size_t context, size_t vocabulary) {
    if (line.size() > request_limit(context) || line.compare(0, 4, "GEN ") != 0)
        throw std::runtime_error("expected bounded GEN <max_new> <id,id,...>");
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
        r.tokens.push_back(int32_t(number(std::min<uint64_t>(vocabulary - 1, uint64_t(INT32_MAX)))));
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
