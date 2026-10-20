// src/worker/wire_sock.hpp - inferswarm #348: the worker's TCP loopback plumbing (internal).
//
// Deliberately small: blocking sockets with a poll() total budget on every full send/recv, full-read/full-write or
// failure, no partial-frame reuse.  A short read, EOF, a deadline or a malformed frame closes the
// connection - the caller never continues on stale bytes.  Loopback only by policy: the listener binds
// 127.0.0.1; the hardware recipe forwards a real link over an owned SSH tunnel.
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::worker {

#ifdef _WIN32
using sock_fd = uintptr_t;
static constexpr sock_fd kBadFd = (sock_fd) ~0;
#else
using sock_fd = int;
static constexpr sock_fd kBadFd = -1;
#endif

namespace sock {

inline void close_fd(sock_fd& fd) {
    if (fd == kBadFd) return;
#ifdef _WIN32
    closesocket((SOCKET) fd);
#else
    ::close(fd);
#endif
    fd = kBadFd;
}

/// Sends every byte or returns false (total budget in ms; <= 0 waits forever).
bool send_all(sock_fd fd, const void* data, size_t n, int deadline_ms);

/// Reads exactly n bytes or returns false (one total budget; EOF and expiry are failures).
bool recv_all(sock_fd fd, void* data, size_t n, int deadline_ms);

// Opt-in setup control. The old full-I/O functions retain their existing budgets.
bool send_all_controlled(sock_fd fd, const void *data, size_t n, int deadline_ms,
                         const std::function<bool()> &cancel);
bool recv_all_controlled(sock_fd fd, void *data, size_t n, int deadline_ms,
                         const std::function<bool()> &cancel);
bool connect_numeric_controlled(const std::string &host, uint16_t port, int deadline_ms,
                                const std::function<bool()> &cancel, sock_fd &out, std::string &err);

/// Binds and listens on 127.0.0.1:port (port 0 picks a free one; `port` receives it).  Loopback only.
bool listen_loopback(uint16_t& port, sock_fd& out, std::string& err);

/// Accepts one connection (deadline in ms; false on error/timeout).
bool accept_one(sock_fd listener, sock_fd& out, int deadline_ms);

/// Connects to host:port (deadline in ms).
bool connect_host(const std::string& host, uint16_t port, int deadline_ms, sock_fd& out, std::string& err);

inline void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD) ms);
#else
    struct pollfd p{0, 0, 0};
    ::poll(&p, 0, ms);
#endif
}

}  // namespace strata::worker::sock

}  // namespace strata::worker
