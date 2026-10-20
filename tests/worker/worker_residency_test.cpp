// Synthetic split-shard fixtures and real worker processes; no model assets.
#include "worker_binary_fixture.hpp"
namespace {
void conn(ExpertWorkerClient& c, const Binary& b) {
    std::string err;
    require(c.connect_local("127.0.0.1", b.port, 5000, {}, err), err);
}
WorkerCacheInfo info(ExpertWorkerClient& c) {
    WorkerCacheInfo out; std::string err;
    require(c.cache_status(out, err), err);
    return out;
}
void local_setup(ExpertWorkerClient& c, const Fixture& f, bool resident = true) {
    std::string err;
    require(c.setup_local(f.setup(), f.m.identity, false, err, resident), err);
}
void complete(ExpertWorkerClient& c, const Fixture& f, bool resident = true) {
    local_setup(c, f, resident);
    std::string err;
    if (!c.cache_hit()) {
        std::vector<uint8_t> hits;
        require(c.load_local({{0,0},{0,1},{1,0},{1,1}}, hits, err), err);
    }
    require(c.mark_ready(err), err);
}
InferRequest request(ExpertWorkerClient& c, const Fixture& f, float x, uint64_t seq = 1) {
    InferRequest r;
    const auto m = f.setup();
    r.session = c.session(); r.model_id = m.model_id; r.partition_id = m.partition_id;
    r.sequence = seq; r.layer = 0; r.T = 1; r.K = 2;
    r.mask = {1,1}; r.ids = {0,1}; r.weights = {0.7f,0.3f}; r.activations.assign(2560, x);
    return r;
}
std::vector<float> parity(ExpertWorkerClient& c, const Fixture& f, float x, uint64_t seq = 1) {
    const auto r = request(c, f, x, seq);
    std::string err; InferReply reply;
    require(c.infer(r, reply, err), err);
    std::vector<float> direct;
    const uint8_t* blobs[] = {f.blobs[0].data(), f.blobs[1].data()};
    require(ExpertWorkerService::direct_native_partial(f.fmt, 1, 2, r.mask.data(), r.weights.data(),
            r.activations.data(), blobs, direct, err), err);
    check(direct.size() == reply.partials.size() && !std::memcmp(direct.data(), reply.partials.data(), direct.size()*4),
          "resident response equals direct native bits for fresh activations");
    return reply.partials;
}
void harness_cleanup(Fixture& f) {
    check(!Binary::listening_port("listening on 127.0.0.1:") &&
          !Binary::listening_port("listening on 127.0.0.1:7") &&
          !Binary::listening_port("listening on 127.0.0.1:70000 (") &&
          Binary::listening_port("listening on 127.0.0.1:7431 (budget") == 7431,
          "partial/invalid listening lines are rejected without exceptions");
    // An owned worker killed after startup is reaped as a failed child. Destruction must
    // then do nothing; the old bool-return cleanup accidentally signalled PID -1 here.
    Binary b(f);
    b.signal_child(SIGKILL);
    check(!b.wait_exit() && b.pid == -1, "failed child is reaped and cannot be signalled again");
    b.signal_child(SIGTERM); // guarded no-op, never kill(-1, ...)
}
void lifecycle(Fixture& f) {
    Binary b(f);
    ExpertWorkerClient c; conn(c,b); complete(c,f);
    std::string err;
    auto cold = info(c);
    const auto bytes = 4*f.fmt.bytes, old = c.session();
    const auto a = parity(c,f,0.1f);
    auto stale = request(c,f,0.1f);
    check(cold.complete && cold.resident_bytes == bytes && cold.loaded_bytes == bytes &&
          cold.verified_bytes == bytes && cold.arena_allocations == 1 && !cold.cache_hits,
          "cold binary load verifies complete arena once inside exact budget");
    c.close(); conn(c,b);
    auto idle = info(c);
    check(idle.state == 0 && !idle.session && idle.complete && idle.resident_bytes == bytes,
          "disconnect clears session while keeping accounted immutable weights");
    c.close();
    // Actual idle gap beyond the original ten-second full receive deadline.
    conn(c,b); pause_ms(11000); c.close(); conn(c,b);
    local_setup(c,f);
    check(c.cache_hit() && c.reused_bytes() == bytes && c.session() && c.session() != old,
          "exact reconnect after 11 seconds hits cache with fresh session");
    InferReply rep;
    check(!c.infer(request(c,f,0.2f),rep,err) && c.last_status() == ST_NOT_READY,
          "cache hit still requires fresh READY before inference");
    std::vector<uint8_t> hits;
    check(!c.load_local({{0,0}},hits,err), "resident cache cannot be overwritten by LOCAL_LOAD");
    check(!c.upload(0,0,f.blobs[0].data(),f.fmt.bytes,err), "resident cache cannot be overwritten by UPLOAD");
    require(c.mark_ready(err),err);
    check(!c.infer(stale,rep,err) && c.last_status() == ST_STALE_SESSION, "old session inference rejected");
    const auto fresh = parity(c,f,0.2f);
    check(fresh != a, "new activations produce fresh results after reconnect");
    check(!c.infer(request(c,f,0.2f),rep,err) && c.last_status() == ST_BAD_REQUEST,
          "replay sequence remains rejected in fresh session");
    const auto again = parity(c,f,0.1f,2);
    check(a == again, "A/B/A result identity survives fresh session");
    const auto warm = info(c);
    check(warm.cache_hits == 1 && warm.reused_bytes == bytes && warm.loaded_bytes == cold.loaded_bytes &&
          warm.verified_bytes == cold.verified_bytes && warm.arena_allocations == 1 && warm.peak_arena_bytes == bytes,
          "cache hit has zero reread/hash and no duplicate arena");
    WorkerStatusInfo ws; require(c.status(ws,err),err);
    check(ws.setup_weight_bytes == 0, "resident binary sends zero network weight bytes");
    std::printf("RESIDENCY cold_loaded=%llu cold_verified=%llu hit=%llu reused=%llu loaded_delta=%llu verified_delta=%llu network=%llu session=%llx resident=%llu peak=%llu allocations=%llu\n",
          (unsigned long long)cold.loaded_bytes,(unsigned long long)cold.verified_bytes,
          (unsigned long long)warm.cache_hits,(unsigned long long)warm.reused_bytes,
          (unsigned long long)(warm.loaded_bytes-cold.loaded_bytes),(unsigned long long)(warm.verified_bytes-cold.verified_bytes),
          (unsigned long long)ws.setup_weight_bytes,(unsigned long long)warm.session,
          (unsigned long long)warm.resident_bytes,(unsigned long long)warm.peak_arena_bytes,(unsigned long long)warm.arena_allocations);
    for (int kind=0; kind<6; ++kind) {
        auto m = f.setup(); auto id = f.m.identity;
        if (kind==0) ++m.model_id;
        if (kind==1) ++id.b; // same first 64 bits, changed full manifest digest
        if (kind==2) { m.partition={1,3}; m.partition_id=blob_digest(m.partition).a; }
        if (kind==3) m.gu_type=2;
        if (kind==4) { m.partition={2,3}; /* same supplied partition id, different bitmap */ }
        if (kind==5) m.n_embd=1280;
        check(!c.setup_local(m,id,false,err,true) && c.last_status()==ST_CACHE_MISMATCH,
              "model/manifest/partition/layout/ownership/geometry mismatch fails closed");
        auto kept=info(c);
        check(!kept.session && kept.complete && kept.resident_bytes==bytes && kept.arena_allocations==1,
              "mismatch invalidates session and retains single frozen cache");
    }
    check(!c.setup(f.setup(),err) && c.last_status()==ST_CACHE_MISMATCH,
          "legacy setup cannot silently replace resident binding");
    // No source files are required for a hit: hide them and use the real engine-side tier.
    for (auto name : {"a.gguf","b.gguf","c.gguf"}) fs::rename(f.root/name,f.root/(std::string(name)+".saved"));
    c.close();
    {
        ExpertWorkerTier t; ExpertWorkerTier::Open o;
        o.port=b.port; o.local=&f.m; o.resident=true; o.n_layers=2; o.n_expert=2;
        o.n_embd=2560; o.n_ff=640; o.partition={3,3}; int fetched=0;
        o.fetch=[&](int64_t,int64_t,uint8_t*,uint32_t){++fetched;return false;};
        require(t.open(o,err),err);
        check(t.stats().cache_hit && t.stats().reused_bytes==bytes && !t.stats().local_bytes &&
              !t.stats().local_read_ns && !t.stats().local_verify_ns && !t.stats().setup_network_bytes && !fetched,
              "real tier skips file reads, payload hashes, fetch and upload on resident hit");
        std::string report; t.report(report); std::printf("TIER %s\n",report.c_str());
    }
    for (auto name : {"a.gguf","b.gguf","c.gguf"}) fs::rename(f.root/(std::string(name)+".saved"),f.root/name);
    conn(c,b); local_setup(c,f); require(c.mark_ready(err),err);
    require(c.unload(err),err); auto empty=info(c);
    check(!empty.complete && !empty.session && !empty.resident_bytes && empty.unloads==1,
          "explicit unload synchronously releases all weights and session");
    complete(c,f); check(!c.cache_hit() && info(c).arena_allocations==2,"replacement after unload allocates once");
    require(c.shutdown(err),err); c.close();
    check(b.wait_exit(),"protocol shutdown exits RC 0 within two seconds with resident cache");
    std::ifstream in(b.log); std::string log((std::istreambuf_iterator<char>(in)),{});
    check(log.find("shutdown: CPU pool joined; resident=0 B")!=std::string::npos,
          "shutdown reports arena release and completed pool destruction before summaries");
}
void slow_status_reader(Fixture& f) {
    Binary b(f,1000);
    ExpertWorkerClient c; conn(c,b); complete(c,f); c.close();
    struct OwnedSocket {
        sock_fd fd = kBadFd;
        ~OwnedSocket() { sock::close_fd(fd); }
    } slow;
    std::string err;
    require(sock::connect_numeric_controlled("127.0.0.1", b.port, 1000, {}, slow.fd, err),err);
    int receive_limit = 4096;
    require(::setsockopt(slow.fd,SOL_SOCKET,SO_RCVBUF,&receive_limit,sizeof receive_limit)==0,"small receive buffer");
    std::vector<uint8_t> frame, flood;
    encode_frame(frame,MSG_CACHE_STATUS,ST_OK,nullptr,0);
    flood.reserve(4u<<20);
    for (size_t i=0;i<(4u<<20)/frame.size();++i) flood.insert(flood.end(),frame.begin(),frame.end());
    // Leave telemetry replies unread. Backpressure may stop this bounded request write early.
    sock::send_all(slow.fd,flood.data(),flood.size(),200);
    pause_ms(1300);
    // The arena must expire before this peer disconnects. Once empty, the service may
    // still be sending telemetry to this one allowed connection, so close it before querying.
    std::ifstream in(b.log);
    const std::string log((std::istreambuf_iterator<char>(in)),{});
    check(log.find("reason=expiry")!=std::string::npos,
          "nonreading telemetry peer cannot defer idle expiry until its I/O deadline");
    sock::close_fd(slow.fd);
    conn(c,b);
    auto expired=info(c);
    check(!expired.resident_bytes && !expired.complete && expired.expirations==1,
          "idle expiry releases weights while telemetry peer stops reading replies");
}
void expiry_and_failure(Fixture& f) {
    std::string err;
    {
        Binary b(f,1000); ExpertWorkerClient c; conn(c,b); complete(c,f); c.close();
        pause_ms(1200); conn(c,b); auto expired=info(c);
        check(!expired.resident_bytes && !expired.session && !expired.complete && expired.expirations==1,
              "disconnected idle expiry releases arena without a connection");
        complete(c,f); check(!c.cache_hit() && info(c).arena_allocations==2,"expired cache reloads and verifies afresh");
        // While a connected peer waits on a header, expiry must still be bounded.
        pause_ms(1200); c.close(); conn(c,b); expired=info(c);
        check(!expired.resident_bytes && expired.expirations==2,"expiry cancels idle socket wait independently of I/O deadline");
    }
    {
        Binary b(f); ExpertWorkerClient c; conn(c,b); local_setup(c,f);
        std::vector<uint8_t> hits; require(c.load_local({{0,0}},hits,err),err);
        check(!c.mark_ready(err),"incomplete cache cannot promote");
        c.close(); conn(c,b); auto empty=info(c);
        check(!empty.complete && !empty.resident_bytes,"incomplete setup is destroyed at disconnect");
        local_setup(c,f); check(!c.cache_hit(),"incomplete setup cannot produce reconnect hit");
        const auto off=f.m.layers[0][0].offset; f.flip("a.gguf",off);
        check(!c.load_local({{0,0}},hits,err) && c.last_status()==ST_DIGEST_MISMATCH,"corrupt setup fails integrity");
        check(!c.mark_ready(err),"failed setup cannot promote");
        c.close(); f.flip("a.gguf",off); conn(c,b); complete(c,f);
        check(!c.cache_hit() && info(c).cache_hits==0,"failed setup cannot be reused");
        // SIGTERM must wake active recv even with a long frame deadline.
        b.signal_child(SIGTERM); check(b.wait_exit(),"SIGTERM during idle connection exits RC 0 within two seconds");
    }
    {
        Binary b(f); ExpertWorkerClient c; conn(c,b); complete(c,f,false); c.close(); conn(c,b);
        check(!info(c).resident_bytes, "configured TTL alone does not retain a legacy local session");
    }
    {
        Binary b(f,0); ExpertWorkerClient c; conn(c,b);
        check(!c.setup_local(f.setup(),f.m.identity,false,err,true) && c.last_status()==ST_LOCAL_UNAVAILABLE,
              "resident negotiation with disabled worker fails without upload fallback");
        complete(c,f,false); c.close(); conn(c,b);
        check(!info(c).resident_bytes,"disabled mode still frees weights at disconnect");
        local_setup(c,f,false); check(!c.mark_ready(err),"legacy local session does not reuse disabled cache");
    }
}
} // namespace
int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        Fixture f;
        std::puts("STAGE harness cleanup"); harness_cleanup(f);
        std::puts("STAGE persistent lifecycle"); lifecycle(f);
        std::puts("STAGE slow telemetry reader"); slow_status_reader(f);
        std::puts("STAGE expiry, failed setup, disabled mode"); expiry_and_failure(f);
    }
    catch (const std::exception& e) { std::fprintf(stderr,"FAIL exception: %s\n",e.what()); ++failures; }
    std::printf("worker_residency_test: %d failures\n",failures);
    return failures ? 1 : 0;
}
