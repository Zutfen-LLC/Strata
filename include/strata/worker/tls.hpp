#pragma once
#include <cstdint>
#include <cstddef>
#include <functional>
#include <string>

namespace strata::worker {
// Explicit SAN DNS identity, independent of the numeric routing address. No wildcard/CN matching.
struct TlsConfig {
    std::string ca_file, cert_file, key_file, peer_dns;
    bool enabled() const { return !ca_file.empty() || !cert_file.empty() || !key_file.empty() || !peer_dns.empty(); }
    bool validate(std::string& err) const;
};
// Does not own the descriptor. Destroy/reset before closing it. Single calling I/O thread.
// Failed I/O destroys TLS state; callers must close the descriptor, never reuse partial frames.
class TlsSession {
public:
    TlsSession() = default;
    ~TlsSession();
    TlsSession(const TlsSession&) = delete;
    TlsSession& operator=(const TlsSession&) = delete;
    bool handshake(uintptr_t fd, const TlsConfig&, bool server, int ms,
                   const std::function<bool()>& cancel, std::string& err);
    bool send_all(const void*, size_t, int ms, const std::function<bool()>& cancel);
    bool recv_all(void*, size_t, int ms, const std::function<bool()>& cancel);
    bool active() const { return ssl_ != nullptr; }
    void reset();
private:
    bool io(void*, size_t, int, const std::function<bool()>&, bool write);
    void* ssl_ = nullptr;
    uintptr_t fd_ = 0;
};
} // namespace strata::worker
