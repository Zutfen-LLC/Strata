// src/worker/expert_worker_main.cpp - inferswarm #348: the resident CPU-expert worker's command line.
//
//   strata-expert-worker [--port P] [--memory-budget-mib N] [--workers N] [--deadline-ms N] [--verbose]
//
// Binds 127.0.0.1 by default. Explicit LAN binding requires exact numeric source peers.
// Serves one connection at a time until SIGINT/SIGTERM. Source filtering is not authentication.
#include "strata/worker/expert_worker.hpp"
#include "strata/worker/local_source.hpp"

#include "wire_sock.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

void usage() {
    std::fprintf(
        stderr,
        "usage: strata-expert-worker [--port P] [--memory-budget-mib N] [--workers N] "
        "[--deadline-ms N] [--verbose] [--tcp-nodelay] [--rpc-timing] "
        "[--residency-ms N | --residency-forever]\n"
        "       [--local-root DIR --local-manifest FILE] [--local-upload-fallback] [--local-setup-ms N]\n"
        "       [--local-load-threads N] (1..4, default 1; separate from inference --workers)\n"
        "       [--bind-ip IPv4 --allow-peer-ip IPv4 [--allow-peer-ip IPv4 ...]]\n"
        "  binds 127.0.0.1 by default; LAN needs exact peers and an operator-managed firewall\n"
        "  source filtering provides neither authentication nor encryption; trusted LAN only\n");
}
}  // namespace

#ifdef _WIN32
struct WsaGuard {
    WsaGuard() {
        WSADATA d;
        started_ = WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }
    ~WsaGuard() {
        if (started_) WSACleanup();
    }
    bool started_;
};
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    WsaGuard wsa;   // untested on Windows: the Linux loopback path is the one this slice validates
#endif
    strata::worker::WorkerOptions opt;
    uint16_t port = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> long long {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-worker: %s needs a value\n", what);
                std::exit(2);
            }
            return std::atoll(argv[++i]);
        };
        if (a == "--port") port = (uint16_t) next("--port");
        else if (a == "--memory-budget-mib") opt.memory_budget = (uint64_t) next("--memory-budget-mib") << 20;
        else if (a == "--workers") opt.workers = (int) next("--workers");
        else if (a == "--deadline-ms") opt.deadline_ms = (int) next("--deadline-ms");
        else if (a == "--bind-ip" || a == "--allow-peer-ip") {
            if (i + 1 >= argc) { usage(); return 2; }
            if (a == "--bind-ip") opt.bind_ip = argv[++i];
            else opt.allow_peer_ips.emplace_back(argv[++i]);
        }
        else if (a == "--local-root" || a == "--local-manifest") {
            if (i + 1 >= argc) {
                usage();
                return 2;
            }
            (a == "--local-root" ? opt.local_root : opt.local_manifest) = argv[++i];
        } else if (a == "--local-upload-fallback")
            opt.local_upload_fallback = true;
        else if (a == "--local-setup-ms")
            opt.local_setup_ms = (int)next("--local-setup-ms");
        else if (a == "--local-load-threads") {
            if (i + 1 >= argc) { usage(); return 2; }
            errno = 0;
            char* end = nullptr;
            const auto value = std::strtoll(argv[++i], &end, 10);
            if (errno || !*argv[i] || *end || value < 1 || value > strata::worker::kMaxLocalLoadThreads) {
                std::fprintf(stderr, "strata-worker: local loading threads must be 1..4\n");
                return 2;
            }
            opt.local_load_threads = (int)value;
        }
        else if (a == "--residency-ms") {
            if (i + 1 >= argc) { usage(); return 2; }
            char* end = nullptr;
            const auto value = std::strtoll(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > strata::worker::kMaxResidencyMs) {
                std::fprintf(stderr, "strata-worker: residency must be 0..86400000 ms\n");
                return 2;
            }
            opt.residency_ms = (int)value;
        }
        else if (a == "--residency-forever") opt.residency_forever = true;
        else if (a == "--tcp-nodelay") opt.accepted_tcp_nodelay = true;
        else if (a == "--rpc-timing") opt.rpc_timing = true;
        else if (a == "--verbose") opt.verbose = 1;
        else {
            usage();
            return 2;
        }
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    strata::worker::ExpertWorkerService service(opt);
    std::string err;
    if (!service.listen(port, err)) {
        std::fprintf(stderr, "strata-worker: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata-worker: listening on %s:%u (budget %llu MiB, %d pool workers, "
                         "accepted TCP_NODELAY %s, RPC timing %s)\n",
                 opt.bind_ip.c_str(), (unsigned) port, (unsigned long long) (opt.memory_budget >> 20), opt.workers,
                 opt.accepted_tcp_nodelay ? "on" : "off", opt.rpc_timing ? "on" : "off");
    for (const auto& peer : opt.allow_peer_ips)
        std::fprintf(stderr, "strata-worker: allowed source %s (trusted LAN; no authentication/encryption)\n",
                     peer.c_str());
    if (opt.local_load_threads > 1)
        std::fprintf(stderr, "strata-worker: local loading lanes=%d; read/verify are summed lane work, "
                             "setup is elapsed wall time (scopes overlap)\n", opt.local_load_threads);
    std::fflush(stderr);
    // run() has no signal hook: poll the flag beside it and stop the service when it lands.
    std::thread watcher([&] {
        while (!g_stop.load()) strata::worker::sock::sleep_ms(200);
        service.stop();
    });
    service.run();
    g_stop.store(true);
    watcher.join();
    const strata::worker::WorkerCounters& c = service.counters();
    std::fprintf(stderr,
                 "strata-worker: stopped; setup %llu B, %llu requests, %llu failures, %llu ms cpu service\n",
                 (unsigned long long) c.setup_weight_bytes, (unsigned long long) c.requests,
                 (unsigned long long) c.failures, (unsigned long long) (c.cpu_service_ns / 1000000));
    std::fprintf(
        stderr,
        "strata-worker: local bytes %llu, hits %llu, misses %llu; read %.3f s, verify %.3f s, setup %.3f s\n",
        (unsigned long long)c.local_bytes, (unsigned long long)c.local_hits,
        (unsigned long long)c.local_misses, c.local_read_ns / 1e9, c.local_verify_ns / 1e9,
        c.local_setup_ns / 1e9);
    std::fprintf(stderr, "strata-worker: cache hits=%llu reused=%llu loaded=%llu verified=%llu "
                         "resident=%llu peak=%llu allocations=%llu expirations=%llu unloads=%llu\n",
                 (unsigned long long)c.cache_hits, (unsigned long long)c.reused_bytes,
                 (unsigned long long)c.loaded_bytes, (unsigned long long)c.verified_bytes,
                 (unsigned long long)c.resident_bytes, (unsigned long long)c.peak_arena_bytes,
                 (unsigned long long)c.arena_allocations, (unsigned long long)c.cache_expirations,
                 (unsigned long long)c.cache_unloads);
    const std::string timing = service.rpc_timing().report("worker");
    if (!timing.empty()) std::fprintf(stderr, "strata-worker: %s\n", timing.c_str());
    return 0;
}
