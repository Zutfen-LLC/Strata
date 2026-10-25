// Representative SEW1 transport replay: real CPU-worker framing and sockets, no expert computation.
#include "strata/worker/expert_wire.hpp"
#include "strata/worker/rpc_timing.hpp"
#include "wire_sock.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>

using namespace strata::worker;
using Clock = std::chrono::steady_clock;

namespace {

constexpr uint64_t kSession = 0x5250435245504c59ull; // "RPCREPLY", fixture-only
constexpr int kDefaultWarmup = 8;
constexpr int kDefaultExchanges = 64;
constexpr int kDefaultDeadlineMs = 10000;

struct Fd {
    sock_fd value = kBadFd;
    Fd() = default;
    explicit Fd(sock_fd fd) : value(fd) {}
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    ~Fd() { sock::close_fd(value); }
};

struct Join {
    std::thread& thread;
    ~Join() { if (thread.joinable()) thread.join(); }
};

void require(bool value, const std::string& why) {
    if (!value) throw std::runtime_error(why);
}

uint64_t ns_since(Clock::time_point start) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

uint64_t read_u64(const uint8_t* p) {
    uint64_t value = 0;
    for (int b = 0; b < 8; ++b) value |= (uint64_t) p[b] << (8 * b);
    return value;
}

void write_u64(uint8_t* p, uint64_t value) {
    for (int b = 0; b < 8; ++b) p[b] = (uint8_t) (value >> (8 * b));
}

struct Config {
    uint8_t tokens = 1;
    uint32_t entries = 9;
    size_t request_bytes = 0;
    size_t reply_bytes = 0;
    int warmup = kDefaultWarmup;
    int exchanges = kDefaultExchanges;
    int deadline_ms = kDefaultDeadlineMs;
    bool accepted_nodelay = false;
    bool operation_first = false, exact_rtt = false;
    int row_entries = -1;
    std::string bind_ip = "127.0.0.1";
    std::string allow_peer_ip = "127.0.0.1";
    std::string route = "loopback";
    bool bind_ip_explicit = false;
    bool allow_peer_ip_explicit = false;
};

in_addr selected_ipv4(const std::string& text, const char* name) {
    in_addr address{};
    require(::inet_pton(AF_INET, text.c_str(), &address) == 1, std::string(name) + " must be numeric IPv4");
    const uint32_t host = ntohl(address.s_addr);
    require(host != INADDR_ANY && host != INADDR_BROADCAST && (host & 0xf0000000u) != 0xe0000000u,
            std::string(name) + " must select one unicast IPv4 address");
    return address;
}

bool is_loopback(in_addr address) {
    return (ntohl(address.s_addr) & 0xff000000u) == 0x7f000000u;
}

constexpr size_t source_request_bytes(uint8_t tokens) {
    return kHeaderBytes + kInferFixed + (size_t) tokens * kMaxK * (1 + 2 + 4) +
           (size_t) tokens * kMaxNEmbd * 4;
}

constexpr size_t source_reply_bytes(uint8_t tokens) {
    return kHeaderBytes + 16 + (size_t) tokens * kMaxNEmbd * 4;
}
size_t selected_reply_bytes(const Config& c) {
    return c.row_entries < 0 ? source_reply_bytes(c.tokens) :
        kHeaderBytes + 20 + size_t(c.row_entries) * (2 + size_t(kMaxNEmbd) * 4);
}

static_assert(source_request_bytes(1) == 10362, "representative CPU request size drifted");
static_assert(source_reply_bytes(1) == 10272, "representative CPU reply size drifted");
static_assert(source_request_bytes(2) == 20672, "two-token CPU request size drifted");
static_assert(source_reply_bytes(2) == 20512, "two-token CPU reply size drifted");

void validate(Config& c) {
    require(c.tokens >= 1 && c.tokens <= kMaxT, "tokens must be 1..8");
    require(c.entries <= (uint32_t) c.tokens * kMaxK, "entries exceed T*K");
    require(c.row_entries >= -1 && c.row_entries <= c.tokens * kMaxK, "row entries exceed T*K");
    if (c.row_entries >= 0) c.entries = (uint32_t)c.row_entries;
    if (!c.request_bytes) c.request_bytes = source_request_bytes(c.tokens);
    if (!c.reply_bytes) c.reply_bytes = selected_reply_bytes(c);
    require(c.request_bytes >= kHeaderBytes + kInferFixed && c.request_bytes <= kInferFrameMax,
            "request frame bytes are outside CPU INFER bounds");
    require(c.reply_bytes >= kHeaderBytes + 16 && c.reply_bytes <= kHeaderBytes +
            (c.row_entries < 0 ? kReplyPayloadMax : kEntryRowsPayloadMax),
            "reply frame bytes are outside CPU INFER_ACK bounds");
    if (c.row_entries >= 0) require(c.reply_bytes == selected_reply_bytes(c), "row reply size must match exact schema");
    require(c.warmup >= 0 && c.warmup <= 1000000, "warmup is outside 0..1000000");
    require(c.exchanges >= 1 && c.exchanges <= 100000000, "exchanges are outside 1..100000000");
    if (c.exact_rtt) require(c.exchanges <= 4096 && c.warmup <= 4096, "exact RTT samples limited to 4096 per phase");
    require(c.deadline_ms >= 1 && c.deadline_ms <= 3600000, "deadline-ms is outside 1..3600000");
    require(c.bind_ip_explicit == c.allow_peer_ip_explicit,
            "--bind-ip and --allow-peer-ip must be selected together");
    const in_addr bind = selected_ipv4(c.bind_ip, "bind-ip");
    const in_addr peer = selected_ipv4(c.allow_peer_ip, "allow-peer-ip");
    require(c.route == "loopback" || c.route == "direct-lan",
            "route must be loopback or direct-lan");
    if (c.bind_ip_explicit) {
        require(c.route == "direct-lan", "an explicit bind and peer require --route direct-lan");
        require(!is_loopback(bind) && !is_loopback(peer),
                "direct-lan bind and peer addresses must not be loopback");
    }
}

bool listen_fixture(const Config& c, uint16_t& port, sock_fd& out, std::string& error) {
    // The shared explicit listener is nonblocking: readiness lost to a reset cannot strand accept().
    return sock::listen_ipv4(c.bind_ip, port, out, error);
}

// Fixture-only exposure restriction. This exact source-address comparison is not authentication.
bool peer_is_allowed(sock_fd fd, const Config& c, std::string& actual, std::string& error) {
    sockaddr_in peer{};
    socklen_t length = sizeof peer;
    if (::getpeername(fd, (sockaddr*) &peer, &length) != 0 || peer.sin_family != AF_INET) {
        error = "accepted peer address unavailable";
        return false;
    }
    char text[INET_ADDRSTRLEN]{};
    if (!::inet_ntop(AF_INET, &peer.sin_addr, text, sizeof text)) {
        error = "accepted peer address could not be formatted";
        return false;
    }
    actual = text;
    if (peer.sin_addr.s_addr != selected_ipv4(c.allow_peer_ip, "allow-peer-ip").s_addr) {
        error = "accepted peer " + actual + " is not allowed";
        return false;
    }
    return true;
}

InferRequest representative_request(const Config& c) {
    InferRequest request;
    request.session = kSession;
    request.model_id = 1;
    request.partition_id = 2;
    request.sequence = 1;
    request.layer = 0;
    request.T = c.tokens;
    request.K = kMaxK;
    const size_t n = (size_t) c.tokens * kMaxK;
    request.mask.assign(n, 0);
    request.ids.resize(n);
    request.weights.assign(n, 0.1f);
    for (size_t i = 0; i < n; ++i) {
        request.mask[i] = i < c.entries;
        request.ids[i] = (uint16_t) (i % kMaxK);
    }
    request.activations.resize((size_t) c.tokens * kMaxNEmbd);
    for (size_t i = 0; i < request.activations.size(); ++i)
        request.activations[i] = (float) ((int) (i % 17) - 8) * 0.01f;
    return request;
}

std::vector<uint8_t> request_frame(const Config& c) {
    auto request = representative_request(c);
    std::vector<uint8_t> payload;
    require(encode_infer(request, kMaxNEmbd, payload), "representative CPU request did not encode");
    payload.resize(c.request_bytes - kHeaderBytes, 0xa5); // measured record may override source-derived size
    std::vector<uint8_t> frame;
    encode_frame(frame, MSG_INFER, ST_OK, payload.data(), payload.size());
    return frame;
}

std::vector<uint8_t> reply_frame(const Config& c) {
    if (c.row_entries >= 0) {
        auto request = representative_request(c);
        std::vector<float> rows(size_t(c.tokens) * kMaxK * kMaxNEmbd);
        for (size_t i = 0; i < rows.size(); ++i) rows[i] = (int(i % 257) - 128) * .015625f;
        std::vector<uint8_t> payload, frame;
        require(encode_infer_rows_ack(request, kMaxNEmbd, rows.data(), payload), "row fixture encode");
        encode_frame(frame, MSG_INFER_ROWS_ACK, ST_OK, payload.data(), payload.size());
        return frame;
    }
    InferReply reply;
    reply.session = kSession;
    reply.sequence = 1;
    reply.partials.assign((size_t) c.tokens * kMaxNEmbd, 0.125f);
    auto payload = encode_infer_ack(reply);
    payload.resize(c.reply_bytes - kHeaderBytes, 0x5a); // fixed bytes; handler never executes experts
    std::vector<uint8_t> frame;
    encode_frame(frame, MSG_INFER_ACK, ST_OK, payload.data(), payload.size());
    return frame;
}

void set_request_sequence(std::vector<uint8_t>& frame, uint64_t sequence) {
    require(frame.size() >= kHeaderBytes + 32, "request frame cannot carry sequence");
    write_u64(frame.data() + kHeaderBytes + 24, sequence);
}

void set_reply_sequence(std::vector<uint8_t>& frame, uint64_t sequence) {
    require(frame.size() >= kHeaderBytes + 16, "reply frame cannot carry sequence");
    write_u64(frame.data() + kHeaderBytes + 8, sequence);
}

bool receive_frame(sock_fd fd, size_t payload_cap, int deadline_ms, WireFrame& frame, std::string& error,
                   sock::IoDiagnostic diagnostic = {}) {
    uint8_t header[kHeaderBytes];
    if (!sock::recv_all_diagnostic(fd, header, sizeof header, deadline_ms, {}, diagnostic)) {
        error = "closed, truncated or timed out reading header";
        return false;
    }
    const uint32_t magic = (uint32_t) header[0] | (uint32_t) header[1] << 8 |
                           (uint32_t) header[2] << 16 | (uint32_t) header[3] << 24;
    uint64_t bytes = 0;
    for (int b = 0; b < 8; ++b) bytes |= (uint64_t) header[8 + b] << (8 * b);
    if (magic != kWireMagic || bytes > payload_cap) {
        error = "malformed magic or oversized payload";
        return false;
    }
    frame.msg = (uint16_t) (header[4] | header[5] << 8);
    frame.status = (uint16_t) (header[6] | header[7] << 8);
    frame.payload.resize((size_t) bytes);
    if (bytes && !sock::recv_all_diagnostic(fd, frame.payload.data(), frame.payload.size(), deadline_ms, {}, diagnostic)) {
        error = "truncated or timed out reading payload";
        return false;
    }
    return true;
}

struct PhaseStats {
    uint64_t completed = 0, failures = 0, wire_bytes = 0;
    RpcLatencyDistribution rtt, send, receive, handler;
    sock::IoCounts io;
    std::vector<uint64_t> exact_rtt;
};
void print_io(const char* role, const Config& c, const PhaseStats& stats) {
    const auto& v = stats.io;
    std::printf("CPU_RPC_REPLAY_IO role=%s phase=measure operation_first=%d row_entries=%d polls=%llu sends=%llu receives=%llu would_block=%llu interrupted=%llu sent_bytes=%llu received_bytes=%llu\n",
        role, c.operation_first, c.row_entries, (unsigned long long)v.polls, (unsigned long long)v.sends,
        (unsigned long long)v.receives, (unsigned long long)v.would_block, (unsigned long long)v.interrupted,
        (unsigned long long)v.sent_bytes, (unsigned long long)v.received_bytes);
    if (!stats.exact_rtt.empty()) {
        auto values = stats.exact_rtt; std::sort(values.begin(), values.end());
        auto at = [&](size_t percent) { return values[(values.size() * percent + 99) / 100 - 1]; };
        std::printf("CPU_RPC_REPLAY_EXACT role=%s phase=measure operation_first=%d samples=%zu p50_ns=%llu p95_ns=%llu p99_ns=%llu max_ns=%llu\n",
            role, c.operation_first, values.size(), (unsigned long long)at(50), (unsigned long long)at(95),
            (unsigned long long)at(99), (unsigned long long)values.back());
    }
}

void print_distribution(const char* role, const char* phase, const char* scope,
                        const Config& c, const PhaseStats& stats) {
    const auto& d = std::strcmp(scope, "handler") == 0 ? stats.handler :
                    std::strcmp(scope, "send") == 0 ? stats.send :
                    std::strcmp(scope, "receive") == 0 ? stats.receive : stats.rtt;
    std::printf("CPU_RPC_REPLAY_V1 role=%s phase=%s scope=%s route=%s transport_only=1 expert_compute=0 "
                "accepted_nodelay=%d T=%u entries=%u source_request_bytes=%zu source_reply_bytes=%zu "
                "request_bytes=%zu reply_bytes=%zu completed=%llu failures=%llu completed_wire_bytes=%llu "
                "samples=%llu mean_us=%.1f p50_bucket_upper_us=%llu p95_bucket_upper_us=%llu "
                "p99_bucket_upper_us=%llu max_us=%.1f\n",
                role, phase, scope, c.route.c_str(), c.accepted_nodelay ? 1 : 0, c.tokens, c.entries,
                source_request_bytes(c.tokens), selected_reply_bytes(c), c.request_bytes, c.reply_bytes,
                (unsigned long long) stats.completed, (unsigned long long) stats.failures,
                (unsigned long long) stats.wire_bytes, (unsigned long long) d.count,
                d.count ? (double) d.total_ns / d.count / 1000.0 : 0.0,
                (unsigned long long) d.percentile_us(50), (unsigned long long) d.percentile_us(95),
                (unsigned long long) d.percentile_us(99), (double) d.max_ns / 1000.0);
}

bool client_exchange(sock_fd fd, std::vector<uint8_t>& request, std::vector<uint8_t>& expected, uint64_t sequence,
                     const Config& c, PhaseStats& stats) {
    set_request_sequence(request, sequence);
    const auto rtt_start = Clock::now();
    const auto send_start = Clock::now();
    if (!sock::send_all_diagnostic(fd, request.data(), request.size(), c.deadline_ms, {}, {c.operation_first, &stats.io})) {
        ++stats.failures;
        return false;
    }
    const uint64_t send_ns = ns_since(send_start);
    const auto receive_start = Clock::now();
    WireFrame reply;
    std::string error;
    if (!receive_frame(fd, c.reply_bytes - kHeaderBytes, c.deadline_ms, reply, error, {c.operation_first, &stats.io}) ||
        reply.msg != (c.row_entries < 0 ? MSG_INFER_ACK : MSG_INFER_ROWS_ACK) || reply.status != ST_OK || reply.payload.size() + kHeaderBytes != c.reply_bytes ||
        reply.payload.size() < 16 || read_u64(reply.payload.data()) != kSession ||
        read_u64(reply.payload.data() + 8) != sequence) {
        ++stats.failures;
        return false;
    }
    const auto receive_ns = ns_since(receive_start), rtt_ns = ns_since(rtt_start);
    set_reply_sequence(expected, sequence);
    if (std::memcmp(reply.payload.data(), expected.data() + kHeaderBytes, reply.payload.size())) { ++stats.failures; return false; }
    stats.send.add(send_ns);
    stats.receive.add(receive_ns);
    stats.rtt.add(rtt_ns);
    if (c.exact_rtt) stats.exact_rtt.push_back(rtt_ns);
    ++stats.completed;
    stats.wire_bytes += c.request_bytes + c.reply_bytes;
    return true;
}

int run_client(const std::string& host, uint16_t port, const Config& c) {
    Fd fd;
    std::string error;
    require(sock::connect_numeric_controlled(host, port, c.deadline_ms, {}, fd.value, error), error);
    std::printf("CPU_RPC_REPLAY_CONNECTED_V1 host=%s port=%u route=%s transport_only=1 expert_compute=0\n",
                host.c_str(), port, c.route.c_str());
    std::fflush(stdout);
    auto request = request_frame(c);
    auto expected = reply_frame(c);
    PhaseStats warmup, measure;
    if (c.exact_rtt) { warmup.exact_rtt.reserve(c.warmup); measure.exact_rtt.reserve(c.exchanges); }
    uint64_t sequence = 0;
    bool healthy = true;
    for (int i = 0; i < c.warmup && healthy; ++i)
        healthy = client_exchange(fd.value, request, expected, ++sequence, c, warmup);
    for (int i = 0; i < c.exchanges && healthy; ++i)
        healthy = client_exchange(fd.value, request, expected, ++sequence, c, measure);
    print_distribution("client", "warmup", "rtt", c, warmup);
    for (const char* scope : {"rtt", "send", "receive"})
        print_distribution("client", "measure", scope, c, measure);
    print_io("client", c, measure);
    return healthy ? 0 : 1;
}

int run_server(sock_fd listener, const Config& c) {
    Fd peer;
    require(sock::accept_one(listener, peer.value, c.deadline_ms, c.accepted_nodelay),
            "accept failed or timed out");
    std::string actual_peer, peer_error;
    require(peer_is_allowed(peer.value, c, actual_peer, peer_error), peer_error);
    std::printf("CPU_RPC_REPLAY_ACCEPTED_V1 peer=%s allowed_peer=%s route=%s source_ip_filter_only=1\n",
                actual_peer.c_str(), c.allow_peer_ip.c_str(), c.route.c_str());
    std::fflush(stdout);
    auto reply = reply_frame(c);
    auto expected = request_frame(c);
    PhaseStats warmup, measure;
    const int total = c.warmup + c.exchanges;
    bool healthy = true;
    for (int i = 0; i < total; ++i) {
        WireFrame request;
        std::string error;
        PhaseStats& stats = i < c.warmup ? warmup : measure;
        if (!receive_frame(peer.value, c.request_bytes - kHeaderBytes, c.deadline_ms, request, error, {c.operation_first, &stats.io}) ||
            request.msg != MSG_INFER || request.status != ST_OK ||
            request.payload.size() + kHeaderBytes != c.request_bytes || request.payload.size() < 32 ||
            read_u64(request.payload.data()) != kSession) {
            ++stats.failures;
            healthy = false;
            break;
        }
        set_request_sequence(expected, (uint64_t)i + 1);
        if (std::memcmp(request.payload.data(), expected.data() + kHeaderBytes, request.payload.size())) {
            ++stats.failures; healthy = false; break;
        }
        const auto handler_start = Clock::now(); // complete request received; no decode or expert compute
        const uint64_t sequence = read_u64(request.payload.data() + 24);
        set_reply_sequence(reply, sequence);
        if (!sock::send_all_diagnostic(peer.value, reply.data(), reply.size(), c.deadline_ms, {}, {c.operation_first, &stats.io})) {
            ++stats.failures;
            healthy = false;
            break;
        }
        stats.handler.add(ns_since(handler_start));
        ++stats.completed;
        stats.wire_bytes += c.request_bytes + c.reply_bytes;
    }
    print_distribution("server", "warmup", "handler", c, warmup);
    print_distribution("server", "measure", "handler", c, measure);
    print_io("server", c, measure);
    return healthy ? 0 : 1;
}

void framing_self_check(const Config& c) {
    {
        Fd listener;
        uint16_t bound = 0;
        std::string error;
        require(listen_fixture(c, bound, listener.value, error), error);
        const int flags = ::fcntl(listener.value, F_GETFL, 0);
        require(flags >= 0 && (flags & O_NONBLOCK), "replay listener could block after stale readiness");
        Fd reset;
        require(sock::connect_numeric_controlled("127.0.0.1", bound, c.deadline_ms, {}, reset.value, error), error);
        linger immediate{1, 0};
        require(::setsockopt(reset.value, SOL_SOCKET, SO_LINGER, &immediate, sizeof immediate) == 0,
                "configure owned reset peer");
        sock::close_fd(reset.value);
        Fd accepted;
        // The reset may remain queued or vanish before accept; either result must be bounded.
        sock::accept_one(listener.value, accepted.value, 5);
        try {
            run_client("localhost", bound, c);
            require(false, "replay unexpectedly resolved a hostname");
        } catch (const std::runtime_error& failure) {
            require(std::string(failure.what()).find("numeric IPv4") != std::string::npos,
                    "replay connect did not reject DNS before opening a socket");
        }
    }
    auto frame = request_frame(c);
    {
        int fds[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "fragmented socketpair");
        Fd sender(fds[0]), receiver(fds[1]);
        bool sent = true;
        std::thread thread([&] {
            for (size_t off = 0; off < frame.size(); off += 7) {
                const size_t count = std::min<size_t>(7, frame.size() - off);
                if (!sock::send_all(sender.value, frame.data() + off, count, c.deadline_ms)) { sent = false; break; }
            }
        });
        Join join{thread};
        WireFrame received;
        std::string error;
        require(receive_frame(receiver.value, c.request_bytes - kHeaderBytes, c.deadline_ms, received, error), error);
        thread.join();
        require(sent && received.msg == MSG_INFER && received.payload.size() + kHeaderBytes == c.request_bytes,
                "fragmented full frame was not preserved");
    }
    for (int kind = 0; kind < 3; ++kind) {
        int fds[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "bad-frame socketpair");
        Fd sender(fds[0]), receiver(fds[1]);
        auto bad = frame;
        if (kind == 0) bad[0] ^= 1;
        else if (kind == 1) bad.resize(kHeaderBytes + 13);
        if (kind < 2) {
            require(sock::send_all(sender.value, bad.data(), bad.size(), c.deadline_ms), "bad-frame send");
            sock::close_fd(sender.value);
        }
        WireFrame received;
        std::string error;
        const int deadline_ms = kind == 2 ? 5 : c.deadline_ms;
        require(!receive_frame(receiver.value, c.request_bytes - kHeaderBytes, deadline_ms, received, error),
                kind == 0 ? "malformed magic was accepted" :
                kind == 1 ? "truncated payload was accepted" : "receive deadline was not finite");
    }
    std::puts("CPU_RPC_REPLAY_SELF_CHECK_V1 fragmented=pass malformed=pass truncated=pass timeout=pass "
              "listener_nonblocking=pass reset_accept=pass numeric_connect=pass");
}

uint64_t number(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    require(*text && !errno && end && !*end && *text != '-', std::string("invalid ") + name);
    return (uint64_t) value;
}

template <typename T>
T option_number(const char* text, const char* name) {
    const uint64_t value = number(text, name);
    require(value <= (uint64_t) std::numeric_limits<T>::max(), std::string(name) + " is too large");
    return (T) value;
}

void options(int argc, char** argv, int start, Config& c) {
    for (int i = start; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) {
            require(i + 1 < argc, std::string(name) + " needs a value");
            return argv[++i];
        };
        if (arg == "--tokens") c.tokens = option_number<uint8_t>(next("--tokens"), "--tokens");
        else if (arg == "--entries") c.entries = option_number<uint32_t>(next("--entries"), "--entries");
        else if (arg == "--request-bytes")
            c.request_bytes = option_number<size_t>(next("--request-bytes"), "--request-bytes");
        else if (arg == "--reply-bytes")
            c.reply_bytes = option_number<size_t>(next("--reply-bytes"), "--reply-bytes");
        else if (arg == "--warmup") c.warmup = option_number<int>(next("--warmup"), "--warmup");
        else if (arg == "--exchanges") c.exchanges = option_number<int>(next("--exchanges"), "--exchanges");
        else if (arg == "--deadline-ms")
            c.deadline_ms = option_number<int>(next("--deadline-ms"), "--deadline-ms");
        else if (arg == "--accepted-nodelay") c.accepted_nodelay = true;
        else if (arg == "--operation-first") c.operation_first = true;
        else if (arg == "--exact-rtt") c.exact_rtt = true;
        else if (arg == "--row-entries") c.row_entries = option_number<int>(next("--row-entries"), "--row-entries");
        else if (arg == "--bind-ip") { c.bind_ip = next("--bind-ip"); c.bind_ip_explicit = true; }
        else if (arg == "--allow-peer-ip") {
            c.allow_peer_ip = next("--allow-peer-ip");
            c.allow_peer_ip_explicit = true;
        }
        else if (arg == "--route") c.route = next("--route");
        else throw std::runtime_error("unknown option: " + arg);
    }
    validate(c);
}

uint16_t port(const char* text, bool allow_zero) {
    const uint64_t value = number(text, "port");
    require(value <= 65535 && (allow_zero || value > 0), "port is outside the allowed range");
    return (uint16_t) value;
}

void usage() {
    std::fprintf(stderr,
        "usage: cpu_rpc_replay [--self-check | --loopback | --serve PORT | --connect IPv4 PORT] [options]\n"
        "options: --tokens N --entries N --request-bytes N --reply-bytes N --warmup N "
        "--exchanges N --deadline-ms N [--accepted-nodelay] [--route loopback|direct-lan] "
        "[--bind-ip IPV4 --allow-peer-ip IPV4] [--operation-first] [--exact-rtt] [--row-entries N]\n");
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config c;
        if (argc == 1 || std::string(argv[1]) == "--self-check") {
            options(argc, argv, argc == 1 ? 1 : 2, c);
            require(c.route == "loopback" && !c.bind_ip_explicit,
                    "self-check mode does not accept external route or peer controls");
            framing_self_check(c);
            if (argc != 1) return 0;
        }
        const std::string mode = argc == 1 ? "--loopback" : argv[1];
        if (mode == "--loopback") {
            options(argc, argv, argc == 1 ? 1 : 2, c);
            require(c.route == "loopback" && !c.bind_ip_explicit,
                    "loopback mode does not accept external route or peer controls");
            Fd listener;
            uint16_t bound = 0;
            std::string error;
            require(listen_fixture(c, bound, listener.value, error), error);
            std::exception_ptr failure;
            int server_rc = 1;
            std::thread server([&] { try { server_rc = run_server(listener.value, c); }
                                     catch (...) { failure = std::current_exception(); } });
            Join join{server};
            const int client_rc = run_client("127.0.0.1", bound, c);
            server.join();
            if (failure) std::rethrow_exception(failure);
            require(client_rc == 0 && server_rc == 0, "loopback exchange failed");
            return 0;
        }
        if (mode == "--serve" && argc >= 3) {
            const uint16_t requested = port(argv[2], true);
            options(argc, argv, 3, c);
            require((c.route == "direct-lan") == c.bind_ip_explicit,
                    "direct-lan server mode requires explicit --bind-ip and --allow-peer-ip");
            Fd listener;
            uint16_t bound = requested;
            std::string error;
            require(listen_fixture(c, bound, listener.value, error), error);
            std::printf("CPU_RPC_REPLAY_LISTENING_V1 address=%s port=%u allowed_peer=%s route=%s "
                        "source_ip_filter_only=1 accepted_nodelay=%d transport_only=1 expert_compute=0\n",
                        c.bind_ip.c_str(), bound, c.allow_peer_ip.c_str(), c.route.c_str(),
                        c.accepted_nodelay ? 1 : 0);
            std::fflush(stdout);
            return run_server(listener.value, c);
        }
        if (mode == "--connect" && argc >= 4) {
            const std::string host = argv[2];
            const uint16_t target = port(argv[3], false);
            options(argc, argv, 4, c);
            require(!c.bind_ip_explicit, "--bind-ip and --allow-peer-ip are server-only");
            if (c.route == "direct-lan")
                require(!is_loopback(selected_ipv4(host, "direct LAN host")),
                        "direct LAN host must not be loopback");
            return run_client(host, target, c);
        }
        usage();
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CPU_RPC_REPLAY_FAIL %s\n", error.what());
        return 1;
    }
}
