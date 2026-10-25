// Both policies share byte/error semantics; only readiness polling order is diagnostic.
#include "wire_sock.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <pthread.h>
using namespace strata::worker;
using Clock = std::chrono::steady_clock;
namespace {
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Pair {
    int fd[2] = {-1, -1};
    Pair() { need(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0, "socketpair"); }
    ~Pair() { sock::close_fd(fd[0]); sock::close_fd(fd[1]); }
};
long elapsed(Clock::time_point begin) { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - begin).count(); }
void signal_noop(int) {}
void test(bool first) {
    sock::IoCounts counts;
    sock::IoDiagnostic policy{first, &counts};
    {
        Pair p; const uint8_t bytes[] = {0, 128, 255, 7}; uint8_t got[4]{};
        need(sock::send_all_diagnostic(p.fd[0], bytes, 4, 1000, {}, policy), "ready send");
        need(counts.sends == 1 && counts.polls == (first ? 0u : 1u), "ready send polling order");
        counts = {};
        need(sock::recv_all_diagnostic(p.fd[1], got, 4, 1000, {}, policy) && !std::memcmp(got, bytes, 4), "ready receive exact bytes");
        need(counts.receives == 1 && counts.received_bytes == 4 && counts.polls == (first ? 0u : 1u), "ready receive polling order");
        counts = {};
        need(!sock::send_all_diagnostic(p.fd[0], bytes, 4, 1000, [] { return true; }, policy), "cancelled writable send refused");
        need(counts.sends == 0 && counts.polls == 0, "already cancelled does no I/O");
        need(sock::send_all(p.fd[0], bytes, 4, 1000), "prepare cancelled receive");
        need(!sock::recv_all_diagnostic(p.fd[1], got, 4, 1000, [] { return true; }, policy), "cancelled readable receive refused");
        need(counts.receives == 0 && sock::recv_all(p.fd[1], got, 4, 1000), "cancelled receive consumes nothing");
        need(sock::recv_all_diagnostic(p.fd[0], got, 0, 1, [] { return true; }, policy), "zero length retains legacy success");
    }
    {
        Pair p; uint8_t byte; counts = {}; auto start = Clock::now();
        need(!sock::recv_all_diagnostic(p.fd[0], &byte, 1, 80, {}, policy), "stalled receive deadline");
        need(elapsed(start) >= 50 && elapsed(start) < 1000 && counts.polls > 0, "stalled receive is bounded and polls");
        if (first) need(counts.would_block > 0, "operation-first reaches EAGAIN before polling");
    }
    for (bool sending : {false, true}) {
        Pair p; int cap = 4096; ::setsockopt(p.fd[0], SOL_SOCKET, SO_SNDBUF, &cap, sizeof cap);
        std::vector<uint8_t> data(sending ? 1u << 20 : 1u, 7); std::atomic<bool> cancel{false};
        std::thread canceller([&] { std::this_thread::sleep_for(std::chrono::milliseconds(60)); cancel = true; });
        counts = {}; const auto start = Clock::now();
        const bool ok = sending ? sock::send_all_diagnostic(p.fd[0], data.data(), data.size(), 0, [&] { return cancel.load(); }, policy)
                                : sock::recv_all_diagnostic(p.fd[0], data.data(), data.size(), 0, [&] { return cancel.load(); }, policy);
        const bool observed_cancel = cancel.load(); const auto io_ms = elapsed(start);
        canceller.join();
        need(!ok && observed_cancel && io_ms < 1000, "infinite-budget stalled I/O remains cancellable");
        if (sending) need(counts.sent_bytes > 0 && counts.sent_bytes < data.size(), "cancelled partial send returns failure");
    }
    {
        Pair p; int cap = 4096; ::setsockopt(p.fd[0], SOL_SOCKET, SO_SNDBUF, &cap, sizeof cap);
        std::vector<uint8_t> data(1u << 20, 42); counts = {}; auto start = Clock::now();
        need(!sock::send_all_diagnostic(p.fd[0], data.data(), data.size(), 80, {}, policy), "stalled send deadline");
        need(elapsed(start) >= 50 && elapsed(start) < 1000 && counts.sent_bytes && counts.sent_bytes < data.size(), "partial send keeps one deadline");
    }
    {
        Pair p; std::atomic<bool> stop{false};
        std::thread trickle([&] {
            // All requested bytes arrive by 160 ms: renewing a 100 ms budget would wrongly succeed.
            for (int i = 0; i < 8 && !stop; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                const char byte = 'x'; if (::send(p.fd[1], &byte, 1, MSG_NOSIGNAL) != 1) break;
            }
        });
        char data[8]; counts = {}; const auto start = Clock::now();
        const bool ok = sock::recv_all_diagnostic(p.fd[0], data, sizeof data, 100, {}, policy);
        const auto io_ms = elapsed(start);
        stop = true; trickle.join();
        need(!ok && io_ms >= 70 && io_ms < 1000 && counts.received_bytes > 0, "trickle progress never renews deadline");
    }
    {
        Pair p; int cap = 4096; ::setsockopt(p.fd[0], SOL_SOCKET, SO_SNDBUF, &cap, sizeof cap);
        std::vector<uint8_t> data(1u << 20), received(data.size());
        for (size_t i = 0; i < data.size(); ++i) data[i] = uint8_t(i * 17);
        sock::IoCounts reader_counts; bool read = false;
        std::thread reader([&] { read = sock::recv_all_diagnostic(p.fd[1], received.data(), received.size(), 4000, {}, {first, &reader_counts}); });
        const bool sent = sock::send_all_diagnostic(p.fd[0], data.data(), data.size(), 4000, {}, policy);
        reader.join(); need(sent && read && received == data, "full fragmented transfer exact");
        sock::close_fd(p.fd[1]);
        need(!sock::send_all_diagnostic(p.fd[0], data.data(), data.size(), 100, {}, policy), "closed-peer send fails without SIGPIPE");
    }
    {
        Pair p; const char data[] = "abc"; char got[8]; counts = {};
        need(sock::send_all(p.fd[1], data, 3, 1000), "prepare EOF"); sock::close_fd(p.fd[1]);
        need(!sock::recv_all_diagnostic(p.fd[0], got, sizeof got, 1000, {}, policy) && counts.received_bytes == 3, "partial EOF is failure");
    }
    {
        Pair p; struct sigaction action{}, old{}; action.sa_handler = signal_noop; sigemptyset(&action.sa_mask);
        need(sigaction(SIGUSR1, &action, &old) == 0, "temporary EINTR handler");
        const auto target = pthread_self();
        std::thread signals([&] {
            for (int i = 0; i < 6; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20)); pthread_kill(target, SIGUSR1);
            }
            // A renewed EINTR budget would accept this byte after the original deadline.
            const char byte = 'x'; ::send(p.fd[1], &byte, 1, MSG_NOSIGNAL);
        });
        char byte; counts = {}; const auto start = Clock::now();
        const bool ok = sock::recv_all_diagnostic(p.fd[0], &byte, 1, 100, {}, policy);
        const auto io_ms = elapsed(start);
        signals.join(); sigaction(SIGUSR1, &old, nullptr);
        need(!ok && counts.interrupted > 0 && io_ms >= 70 && io_ms < 1000, "EINTR retries preserve absolute deadline");
    }
}
}
int main() {
    ::alarm(25);
    try { test(false); test(true); std::puts("WORKER_SOCKET_OPERATION_TEST ok=1 policies=2"); return 0; }
    catch (const std::exception& e) { std::fprintf(stderr, "WORKER_SOCKET_OPERATION_TEST failure: %s\n", e.what()); return 1; }
}
