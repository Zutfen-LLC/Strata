// src/worker/wire_sock.cpp - inferswarm #348: the socket layer's implementation.
#include "wire_sock.hpp"

namespace strata::worker::sock {
namespace {

bool wait_fd(sock_fd fd, bool write, int deadline_ms) {
#ifdef _WIN32
    // WSAPoll exists but its semantics differ across versions; select on the raw socket is the portable
    // Winsock deadline, and this path is small enough that fd_set truncation below FD_SETSIZE never bites
    // (one loopback socket at a time).
    fd_set set;
    FD_ZERO(&set);
    FD_SET(fd, &set);
    timeval tv{deadline_ms / 1000, (long) (deadline_ms % 1000) * 1000};
    const int r = ::select(0, write ? nullptr : &set, write ? &set : nullptr, nullptr,
                           deadline_ms <= 0 ? nullptr : &tv);
    return r > 0;
#else
    struct pollfd p{fd, (int16_t) (write ? POLLOUT : POLLIN), 0};
    for (;;) {
        const int r = ::poll(&p, 1, deadline_ms);
        if (r > 0) return true;
        if (r == 0) return false;                    // deadline
        if (errno == EINTR) continue;
        return false;
    }
#endif
}

}  // namespace

bool send_all(sock_fd fd, const void* data, size_t n, int deadline_ms) {
    const uint8_t* p = (const uint8_t*) data;
    size_t off = 0;
    while (off < n) {
        if (!wait_fd(fd, true, deadline_ms)) return false;
#ifdef _WIN32
        const int chunk = (int) (::send(fd, (const char*) (p + off), (int) (n - off), 0));
        const bool failed = chunk == SOCKET_ERROR;
#else
        const ssize_t chunk = ::send(fd, p + off, n - off, MSG_NOSIGNAL);
        const bool failed = chunk < 0 && errno != EINTR;
        if (chunk < 0 && errno == EINTR) continue;
#endif
        if (failed || chunk == 0) return false;
        off += (size_t) chunk;
    }
    return true;
}

bool recv_all(sock_fd fd, void* data, size_t n, int deadline_ms) {
    uint8_t* p = (uint8_t*) data;
    size_t off = 0;
    while (off < n) {
        if (!wait_fd(fd, false, deadline_ms)) return false;
#ifdef _WIN32
        const int chunk = (int) ::recv(fd, (char*) (p + off), (int) (n - off), 0);
        const bool failed = chunk == SOCKET_ERROR;
#else
        const ssize_t chunk = ::recv(fd, p + off, n - off, 0);
        const bool failed = chunk < 0 && errno != EINTR;
        if (chunk < 0 && errno == EINTR) continue;
#endif
        if (failed || chunk == 0) return false;      // error, or EOF before the frame was whole
        off += (size_t) chunk;
    }
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

bool accept_one(sock_fd listener, sock_fd& out, int deadline_ms) {
    if (!wait_fd(listener, false, deadline_ms)) return false;
    out = ::accept(listener, nullptr, nullptr);
    return out != kBadFd;
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
