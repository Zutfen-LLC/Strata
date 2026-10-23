// Bounded I/O must use one budget across partial progress, including a peer that stops reading.
#include "wire_sock.hpp"
#include "strata/worker/expert_wire.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <netinet/tcp.h>

using namespace strata::worker;
using Clock = std::chrono::steady_clock;
int main() {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; }
    };
    int fd[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) return 2;
    std::atomic<bool> stop{false};
    std::thread trickle([&] {
        for (int i = 0; i < 4 && !stop; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const char byte = 'x';
            if (::send(fd[1], &byte, 1, MSG_NOSIGNAL) != 1) break;
        }
    });
    char buf[4];
    auto start = Clock::now();
    const bool read = sock::recv_all(fd[0], buf, sizeof buf, 250);
    const auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    check(!read, "trickled bytes cannot renew the read budget");
    check(read_ms < 750, "partial read has a bounded total duration");
    stop = true;
    trickle.join();
    sock::close_fd(fd[0]); sock::close_fd(fd[1]);

    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) return 2;
    int capacity = 4096;
    ::setsockopt(fd[0], SOL_SOCKET, SO_SNDBUF, &capacity, sizeof capacity);
    stop = false;
    std::thread slow_reader([&] {
        char chunk[1024];
        for (int i = 0; i < 20 && !stop; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (::recv(fd[1], chunk, sizeof chunk, MSG_DONTWAIT) < 0 && errno != EAGAIN) break;
        }
        sock::close_fd(fd[1]);
    });
    std::vector<uint8_t> large(kUploadFrameMax, 7);
    start = Clock::now();
    const bool wrote = sock::send_all(fd[0], large.data(), large.size(), 250);
    const auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    check(!wrote, "backpressure fails an incomplete write");
    check(write_ms < 750, "a writable poll cannot lead to an unbounded blocking send");
    stop = true;
    slow_reader.join();
    sock::close_fd(fd[0]);
    // A full 1 MiB setup frame also succeeds across many short writes/reads. No timing assertion.
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) return 2;
    ::setsockopt(fd[0], SOL_SOCKET, SO_SNDBUF, &capacity, sizeof capacity);
    std::vector<uint8_t> received(large.size());
    bool complete = false;
    std::thread reader([&] {
        complete = sock::recv_all(fd[1], received.data(), received.size(), 4000);
    });
    check(sock::send_all(fd[0], large.data(), large.size(), 4000), "full-cap frame survives backpressure");
    reader.join();
    check(complete && received == large, "fragmented full-cap transfer preserves every byte");
    sock::close_fd(fd[1]);
    check(!sock::send_all(fd[0], large.data(), large.size(), 250), "disconnected full-cap send fails");
    sock::close_fd(fd[0]);

    // Accepted CPU-worker TCP_NODELAY is a per-accept opt-in. The default arm also guards GPU callers,
    // which continue to use accept_one's default argument and must not inherit this experiment.
    uint16_t port = 0;
    sock_fd listener = kBadFd;
    std::string error;
    check(sock::listen_loopback(port, listener, error), "loopback listener for accepted option");
    if (listener != kBadFd) {
        for (bool enabled : {false, true}) {
            sock_fd client = kBadFd, accepted = kBadFd;
            std::thread connector([&] { sock::connect_host("127.0.0.1", port, 1000, client, error); });
            const bool accepted_ok = sock::accept_one(listener, accepted, 1000, enabled);
            connector.join();
            check(accepted_ok && client != kBadFd, "loopback connection for accepted option");
            int actual = -1;
            socklen_t option_bytes = sizeof actual;
            check(accepted_ok && ::getsockopt(accepted, IPPROTO_TCP, TCP_NODELAY, &actual, &option_bytes) == 0,
                  "accepted descriptor TCP_NODELAY is readable");
            check(actual == (enabled ? 1 : 0), enabled ? "accepted TCP_NODELAY opt-in is on"
                                                       : "accepted TCP_NODELAY baseline stays off");
            sock::close_fd(client);
            sock::close_fd(accepted);
        }
        sock_fd none = kBadFd;
        check(!sock::accept_one(listener, none, 10, true) && none == kBadFd,
              "accepted-option path preserves the accept timeout");
        sock::close_fd(listener);
    }
    std::printf("worker_socket_test: read=%lldms write=%lldms failures=%d\n",
                (long long) read_ms, (long long) write_ms, fails);
    return fails ? 1 : 0;
}
