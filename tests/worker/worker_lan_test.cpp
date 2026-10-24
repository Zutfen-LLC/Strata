// Synthetic local files and 127/8 source addresses; no physical NIC, firewall or model assets.
#include "local_fixture.hpp"
#include <stdexcept>

namespace {
struct OwnedSocket {
    sock_fd fd = kBadFd;
    ~OwnedSocket() { sock::close_fd(fd); }
};
void source_connection(OwnedSocket& peer, uint16_t port, const char* address) {
    peer.fd = ::socket(AF_INET, SOCK_STREAM, 0);
    require(peer.fd != kBadFd, "create denied peer");
    sockaddr_in source{}, target{};
    source.sin_family = target.sin_family = AF_INET;
    require(::inet_pton(AF_INET, address, &source.sin_addr) == 1, "source IP");
    require(::inet_pton(AF_INET, "127.0.0.1", &target.sin_addr) == 1, "target IP");
    target.sin_port = htons(port);
    require(::bind(peer.fd, (sockaddr*)&source, sizeof source) == 0, "bind denied source");
    require(::connect(peer.fd, (sockaddr*)&target, sizeof target) == 0, "connect denied source");
}
void denied_source(uint16_t port, const std::vector<uint8_t>& frame) {
    OwnedSocket peer;
    source_connection(peer, port, "127.0.0.2");
    // It is valid for rejection to beat the send. No bytes may reach a protocol handler.
    if (!frame.empty()) sock::send_all(peer.fd, frame.data(), frame.size(), 1000);
    pollfd p{peer.fd, POLLIN, 0};
    check(::poll(&p, 1, 1500) > 0, "denied peer closes before the 5-second protocol deadline");
    char reply[16];
    const auto n = ::recv(peer.fd, reply, sizeof reply, MSG_DONTWAIT);
    check(n == 0 || (n < 0 && errno == ECONNRESET), "denied peer gets EOF/reset and no protocol reply");
}
void second_allowed_source(uint16_t port) {
    OwnedSocket peer;
    source_connection(peer, port, "127.0.0.3");
    std::vector<uint8_t> request, expected;
    encode_frame(request, MSG_STATUS, ST_OK, nullptr, 0);
    // Legacy STATUS always has 59 payload bytes. Check its unchanged header and consume the reply.
    std::vector<uint8_t> payload(59), reply(kHeaderBytes + 59);
    encode_frame(expected, MSG_STATUS_ACK, ST_OK, payload.data(), payload.size());
    require(sock::send_all(peer.fd, request.data(), request.size(), 1000), "second source status write");
    require(sock::recv_all(peer.fd, reply.data(), reply.size(), 1000), "second source status read");
    check(!std::memcmp(reply.data(), expected.data(), kHeaderBytes), "second exact allowlist entry is admitted");
}
void complete(ExpertWorkerClient& c, const Fixture& f) {
    std::string error;
    require(c.setup_local(f.setup(), f.m.identity, false, error, true), error);
    if (!c.cache_hit()) {
        std::vector<uint8_t> hits;
        require(c.load_local({{0,0},{0,1},{1,0},{1,1}}, hits, error), error);
    }
    require(c.mark_ready(error), error);
}
WorkerCacheInfo info(ExpertWorkerClient& c) {
    WorkerCacheInfo result; std::string error;
    require(c.cache_status(result, error), error);
    return result;
}
void policy() {
    std::vector<uint32_t> peers;
    std::string error;
    check(sock::listener_policy("127.0.0.1", {}, peers, error) && peers.empty(), "loopback default has no filter");
    for (const auto& address : {"", "0.0.0.0", "0.1.2.3", "224.0.0.1", "255.255.255.255",
                                "localhost", "::1", "10.1.2.3/24", "10.1.2.999"}) {
        check(!sock::listener_policy(address, {"10.1.2.4"}, peers, error), "unsafe/non-numeric bind refused");
        check(!sock::listener_policy("10.1.2.3", {address}, peers, error) && peers.empty(),
              "unsafe/non-exact peer refused without a partial allowlist");
    }
    check(!sock::listener_policy("10.1.2.3", {}, peers, error), "LAN bind without peers refused");
    check(sock::listener_policy("10.1.2.3", {"10.1.2.4", "10.1.2.5", "10.1.2.4"}, peers, error) &&
          peers.size() == 2, "multiple exact peers supported and duplicates collapsed");
    WorkerOptions invalid; invalid.bind_ip = "0.0.0.0";
    invalid.local_root = "/not/a/model"; invalid.local_manifest = "/not/a/manifest";
    ExpertWorkerService service(invalid); uint16_t port = 0;
    check(!service.listen(port, error) && error.find("bind IP") != std::string::npos,
          "unsafe policy fails before opening local sources or sockets");
    for (int deadline : {0, -1}) {
        WorkerOptions bounded;
        bounded.bind_ip = "192.0.2.1"; bounded.allow_peer_ips = {"192.0.2.2"};
        bounded.deadline_ms = deadline;
        ExpertWorkerService lan(bounded);
        check(!lan.listen(port, error) && error.find("positive --deadline-ms") != std::string::npos,
              "LAN cannot opt into an unbounded frame deadline");
        bounded.bind_ip = "127.0.0.1"; bounded.allow_peer_ips = {"127.0.0.1"};
        ExpertWorkerService filtered(bounded);
        check(!filtered.listen(port, error), "filtered loopback also requires bounded frame reads");
    }
}
}

int main() {
    alarm(60);
    try {
        policy();
        Fixture f;
        auto options = f.options();
        options.bind_ip = "127.0.0.1";
        options.allow_peer_ips = {"127.0.0.1", "127.0.0.3"};
        options.residency_forever = true;
        options.deadline_ms = 5000;
        options.accepted_tcp_nodelay = true;
        Service service(options);
        ExpertWorkerClient client; connect(client, service); complete(client, f);
        const auto first_session = client.session();
        const auto first_output = infer(client, f, 0, 1, 1);
        const auto cold = info(client);
        client.close();
        second_allowed_source(service.port);
        { OwnedSocket closed; source_connection(closed, service.port, "127.0.0.3"); }
        // Empty, partial and valid mutation/control frames must all be rejected before dispatch.
        denied_source(service.port, {});
        denied_source(service.port, {0x53, 0x45, 0x57});
        for (auto msg : {MSG_SHUTDOWN, MSG_UNLOAD, MSG_SETUP_BEGIN, MSG_INFER}) {
            std::vector<uint8_t> frame;
            encode_frame(frame, msg, ST_OK, nullptr, 0);
            denied_source(service.port, frame);
        }
        connect(client, service);
        const auto kept = info(client);
        WorkerStatusInfo status; std::string error;
        require(client.status(status, error), error);
        check(!kept.session && !kept.state && kept.complete && kept.resident_bytes == cold.resident_bytes &&
              kept.loaded_bytes == cold.loaded_bytes && kept.verified_bytes == cold.verified_bytes &&
              kept.arena_allocations == cold.arena_allocations && kept.cache_hits == cold.cache_hits &&
              kept.expirations == cold.expirations && kept.unloads == cold.unloads && !status.failures,
              "denied peers never mutate resident arena, session, protocol counters or stop the service");
        complete(client, f);
        check(client.cache_hit() && client.session() != first_session,
              "allowed reconnect reuses resident cache with fresh session");
        check(infer(client, f, 0, 1, 1) == first_output, "allowed reconnect preserves output bits");
        require(client.unload(error), error);
        check(!info(client).resident_bytes, "allowed unload control still works");
        require(client.shutdown(error), error);
        client.close();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what()); ++failures;
    }
    std::printf("CPU_WORKER_LAN_FIXTURES_V1 failures=%d physical_nic=0 models=0\n", failures);
    return failures ? 1 : 0;
}
