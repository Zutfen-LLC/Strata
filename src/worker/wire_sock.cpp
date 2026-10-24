// src/worker/wire_sock.cpp - inferswarm #348: the socket layer's implementation.
#include "wire_sock.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>
#ifndef _WIN32
#include <fcntl.h>
#endif

namespace strata::worker::sock {
namespace {

// One monotonic budget per full read/write. Neither partial progress nor EINTR renews it.
struct Budget {
    using Clock = std::chrono::steady_clock;
    bool finite;
    Clock::time_point end;
    explicit Budget(int ms) : finite(ms > 0), end(Clock::now() + std::chrono::milliseconds(ms > 0 ? ms : 0)) {}
    int remaining() const {
        if (!finite) return -1;
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - Clock::now()).count();
        if (ns <= 0) return 0;
        return (int) std::min<int64_t>((ns + 999999) / 1000000, std::numeric_limits<int>::max());
    }
};

bool wait_fd(sock_fd fd, bool write, const Budget &budget, const std::function<bool()> &cancel = {}) {
    for (;;) {
        if (cancel && cancel())
            return false;
        int ms = budget.remaining();
        if (cancel && (ms < 0 || ms > 50))
            ms = 50;
        if (ms == 0) return false;
#ifdef _WIN32
        fd_set set;
        FD_ZERO(&set);
        FD_SET(fd, &set);
        timeval tv{ms > 0 ? ms / 1000 : 0, ms > 0 ? (long) (ms % 1000) * 1000 : 0};
        const int r = ::select(0, write ? nullptr : &set, write ? &set : nullptr, nullptr,
                               ms < 0 ? nullptr : &tv);
        if (r < 0 && WSAGetLastError() == WSAEINTR) continue;
#else
        struct pollfd p{fd, (int16_t) (write ? POLLOUT : POLLIN), 0};
        const int r = ::poll(&p, 1, ms);
        if (r < 0 && errno == EINTR) continue;
#endif
        if (cancel && r == 0 && budget.remaining() != 0)
            continue;
        return r > 0 && (!cancel || !cancel());
    }
}

#ifdef _WIN32
bool nonblocking(sock_fd fd) {
    u_long mode = 1;
    return ::ioctlsocket((SOCKET) fd, FIONBIO, &mode) == 0;
}
#else
bool nonblocking(sock_fd fd) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    return flags >= 0 && ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}
#endif

}  // namespace

bool send_all(sock_fd fd, const void *data, size_t n, int deadline_ms) {
    return send_all_controlled(fd, data, n, deadline_ms, {});
}
bool send_all_controlled(sock_fd fd, const void *data, size_t n, int deadline_ms,
                         const std::function<bool()> &cancel) {
    const Budget budget(deadline_ms);
#ifdef _WIN32
    if (!nonblocking(fd)) return false;
#endif
    const uint8_t* p = (const uint8_t*) data;
    size_t off = 0;
    while (off < n) {
        if (!wait_fd(fd, true, budget, cancel))
            return false;
#ifdef _WIN32
        const int chunk = (int) (::send(fd, (const char*) (p + off), (int) (n - off), 0));
        const bool failed = chunk == SOCKET_ERROR;
        if (failed && (WSAGetLastError() == WSAEINTR || WSAGetLastError() == WSAEWOULDBLOCK)) continue;
#else
        const ssize_t chunk = ::send(fd, p + off, n - off, MSG_NOSIGNAL | MSG_DONTWAIT);
        const bool failed = chunk < 0 && errno != EINTR;
        if (chunk < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
#endif
        if (failed || chunk == 0) return false;
        off += (size_t) chunk;
    }
    return true;
}

bool recv_all(sock_fd fd, void *data, size_t n, int deadline_ms) {
    return recv_all_controlled(fd, data, n, deadline_ms, {});
}
bool recv_all_controlled(sock_fd fd, void *data, size_t n, int deadline_ms,
                         const std::function<bool()> &cancel) {
    const Budget budget(deadline_ms);
#ifdef _WIN32
    if (!nonblocking(fd)) return false;
#endif
    uint8_t* p = (uint8_t*) data;
    size_t off = 0;
    while (off < n) {
        if (!wait_fd(fd, false, budget, cancel))
            return false;
#ifdef _WIN32
        const int chunk = (int) ::recv(fd, (char*) (p + off), (int) (n - off), 0);
        const bool failed = chunk == SOCKET_ERROR;
        if (failed && (WSAGetLastError() == WSAEINTR || WSAGetLastError() == WSAEWOULDBLOCK)) continue;
#else
        const ssize_t chunk = ::recv(fd, p + off, n - off, MSG_DONTWAIT);
        const bool failed = chunk < 0 && errno != EINTR;
        if (chunk < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
#endif
        if (failed || chunk == 0) return false;      // error, or EOF before the frame was whole
        off += (size_t) chunk;
    }
    return true;
}

bool connect_numeric_controlled(const std::string &host, uint16_t port, int deadline_ms,
                                const std::function<bool()> &cancel, sock_fd &out, std::string &err) {
    out = kBadFd;
    if (cancel && cancel()) {
        err = "setup cancelled before connection";
        return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        err = "local setup needs a numeric IPv4 host (use 127.0.0.1 for an SSH forward)";
        return false;
    }
    sock_fd fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == kBadFd) {
        err = "socket failed";
        return false;
    }
#ifdef _WIN32
    if (!nonblocking(fd)) {
        close_fd(fd);
        err = "nonblocking connect failed";
        return false;
    }
#else
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        close_fd(fd);
        err = "nonblocking connect failed";
        return false;
    }
#endif
    const Budget budget(deadline_ms);
    const int rc = ::connect(fd, (sockaddr *)&addr, sizeof addr);
#ifdef _WIN32
    const bool pending =
        rc != 0 && (WSAGetLastError() == WSAEWOULDBLOCK || WSAGetLastError() == WSAEINPROGRESS);
#else
    const bool pending = rc != 0 && errno == EINPROGRESS;
#endif
    int error = 0;
    socklen_t size = sizeof error;
    if ((rc != 0 && !pending) || (pending && !wait_fd(fd, true, budget, cancel)) ||
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, (char *)&error, &size) != 0 || error || (cancel && cancel())) {
        close_fd(fd);
        err = "connect failed, cancelled or timed out";
        return false;
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one);
    out = fd;
    return true;
}

namespace {
bool unicast_ipv4(const std::string& text, in_addr& address) {
    if (text.find('\0') != std::string::npos || ::inet_pton(AF_INET, text.c_str(), &address) != 1)
        return false;
    const uint32_t host = ntohl(address.s_addr);
    // Exclude unspecified/"this network", multicast, reserved and limited broadcast addresses.
    return (host >> 24) != 0 && (host >> 24) < 224;
}
}

bool listener_policy(const std::string& bind_ip, const std::vector<std::string>& peers,
                     std::vector<uint32_t>& allowed, std::string& err) {
    allowed.clear();
    in_addr address{};
    if (!unicast_ipv4(bind_ip, address)) {
        err = "bind IP must be one numeric unicast IPv4 address (no wildcard, hostname or CIDR)";
        return false;
    }
    if ((ntohl(address.s_addr) >> 24) != 127 && peers.empty()) {
        err = "non-loopback bind requires at least one exact --allow-peer-ip; use a trusted LAN and firewall";
        return false;
    }
    std::vector<uint32_t> parsed;
    for (const auto& peer : peers) {
        if (!unicast_ipv4(peer, address)) {
            err = "each allowed peer must be one numeric unicast IPv4 address (no wildcard, hostname or CIDR)";
            return false;
        }
        if (std::find(parsed.begin(), parsed.end(), address.s_addr) == parsed.end())
            parsed.push_back(address.s_addr);
    }
    allowed = std::move(parsed);
    return true;
}

bool listen_loopback(uint16_t& port, sock_fd& out, std::string& err) {
    out = ::socket(AF_INET, SOCK_STREAM, 0);
    if (out == kBadFd) { err = "socket() failed"; return false; }
    int one = 1;
    ::setsockopt(out, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(0x7f000001u);       // 127.0.0.1 only: no public listener, by policy
    addr.sin_port = htons(port);
    if (::bind(out, (sockaddr*) &addr, sizeof addr) != 0) {
        err = "bind 127.0.0.1:" + std::to_string(port) + " failed";
        close_fd(out);
        return false;
    }
    if (::listen(out, 1) != 0) {   // one persistent connection: a second caller waits, none is dropped
        err = "listen failed";
        close_fd(out);
        return false;
    }
    socklen_t len = sizeof addr;
    if (::getsockname(out, (sockaddr*) &addr, &len) == 0) port = ntohs(addr.sin_port);
    return true;
}

bool listen_ipv4(const std::string& bind_ip, uint16_t& port, sock_fd& out, std::string& err) {
    out = kBadFd;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    if (!unicast_ipv4(bind_ip, addr.sin_addr)) {
        err = "listener needs a numeric unicast IPv4 address";
        return false;
    }
    out = ::socket(AF_INET, SOCK_STREAM, 0);
    if (out == kBadFd) { err = "socket() failed"; return false; }
    // poll readiness can disappear when a queued peer resets. Never block in accept after that race.
    if (!nonblocking(out)) {
        err = "nonblocking listener setup failed";
        close_fd(out);
        return false;
    }
    int one = 1;
    ::setsockopt(out, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
    addr.sin_port = htons(port);
    if (::bind(out, (sockaddr*) &addr, sizeof addr) != 0) {
        err = "bind " + bind_ip + ":" + std::to_string(port) + " failed";
        close_fd(out);
        return false;
    }
    if (::listen(out, 1) != 0) {   // one persistent connection: a second caller waits, none is dropped
        err = "listen failed";
        close_fd(out);
        return false;
    }
    socklen_t len = sizeof addr;
    if (::getsockname(out, (sockaddr*) &addr, &len) == 0) port = ntohs(addr.sin_port);
    return true;
}

bool accept_one(sock_fd listener, sock_fd& out, int deadline_ms, bool tcp_nodelay) {
    const Budget budget(deadline_ms);
    if (!wait_fd(listener, false, budget)) return false;
    out = ::accept(listener, nullptr, nullptr);
    if (out == kBadFd) return false;
    if (tcp_nodelay) {
        int one = 1;
        if (::setsockopt(out, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one) != 0) {
            close_fd(out);
            return false;
        }
    }
    return true;
}

bool accept_allowed(sock_fd listener, sock_fd& out, int deadline_ms, bool tcp_nodelay,
                    const std::vector<uint32_t>& allowed_peers) {
    out = kBadFd;
    if (!accept_one(listener, out, deadline_ms)) return false;
    sockaddr_in peer{};
    socklen_t peer_size = sizeof peer;
    if (::getpeername(out, (sockaddr*)&peer, &peer_size) != 0 || peer_size != sizeof peer ||
        peer.sin_family != AF_INET ||
        std::find(allowed_peers.begin(), allowed_peers.end(), peer.sin_addr.s_addr) == allowed_peers.end()) {
        close_fd(out);
        return false;
    }
    if (tcp_nodelay) {
        int one = 1;
        if (::setsockopt(out, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one) != 0) {
            close_fd(out);
            return false;
        }
    }
    return true;
}

bool connect_host(const std::string& host, uint16_t port, int deadline_ms, sock_fd& out, std::string& err) {
    out = kBadFd;
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* list = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &list) != 0 || list == nullptr) {
        err = "cannot resolve " + host;
        return false;
    }
    for (addrinfo* ai = list; ai != nullptr; ai = ai->ai_next) {
        sock_fd fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd == kBadFd) continue;
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            out = fd;
            break;
        }
        close_fd(fd);
    }
    ::freeaddrinfo(list);
    if (out == kBadFd) { err = "connect " + host + ":" + std::to_string(port) + " failed"; return false; }
    int one = 1;
    ::setsockopt(out, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    (void) deadline_ms;
    return true;
}

}  // namespace strata::worker::sock
