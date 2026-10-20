// Bounded I/O must use one budget across partial progress, including a peer that stops reading.
#include "wire_sock.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <sys/socket.h>

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
    std::vector<uint8_t> large(16u << 20, 7);
    start = Clock::now();
    const bool wrote = sock::send_all(fd[0], large.data(), large.size(), 250);
    const auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    check(!wrote, "backpressure fails an incomplete write");
    check(write_ms < 750, "a writable poll cannot lead to an unbounded blocking send");
    stop = true;
    slow_reader.join();
    sock::close_fd(fd[0]);
    std::printf("worker_socket_test: read=%lldms write=%lldms failures=%d\n",
                (long long) read_ms, (long long) write_ms, fails);
    return fails ? 1 : 0;
}
