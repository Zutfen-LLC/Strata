// Included inside expert_worker_test's anonymous namespace, after the shared fixture/raw helpers.
// Observe real client/service frames without changing their protocol. Delay is artificial, once per
// UPLOAD_ACK; it is neither a measured network RTT nor a physical-network speedup prediction.
struct SetupObserver {
    strata::worker::sock_fd listener = strata::worker::kBadFd;
    uint16_t port = 0;
    std::thread thread;
    uint64_t bytes = 0, chunks = 0, acks = 0, wire_bytes = 0, tail = 0, max_payload = 0;
    bool ok = true;
    SetupObserver(uint16_t worker_port, uint32_t cap, int delay_ms) {
        using namespace strata::worker;
        std::string err;
        if (!sock::listen_loopback(port, listener, err)) { ok = false; return; }
        thread = std::thread([this, worker_port, cap, delay_ms] {
            using namespace strata::worker;
            Raw client, worker;
            if (!sock::accept_one(listener, client.fd, 10000) || !worker.conn(worker_port)) {
                ok = false; return;
            }
            for (;;) {
                uint16_t msg = 0, status = 0;
                std::vector<uint8_t> p;
                if (!client.recv_reply(msg, status, p, 10000)) break;
                if (msg == MSG_UPLOAD) {
                    const size_t data = p.size() >= kUploadHeadBytes ? p.size() - kUploadHeadBytes : 0;
                    ok = ok && data > 0 && data <= cap;
                    bytes += data; ++chunks; tail = data;
                    max_payload = std::max<uint64_t>(max_payload, p.size());
                    wire_bytes += kHeaderBytes + p.size();
                }
                const bool uploading = msg == MSG_UPLOAD;
                if (!worker.send(msg, status, p.data(), p.size()) ||
                    !worker.recv_reply(msg, status, p, 10000)) { ok = false; break; }
                if (uploading) {
                    ok = ok && msg == MSG_UPLOAD_ACK && status == ST_OK;
                    ++acks; wire_bytes += kHeaderBytes + p.size();
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                }
                if (!client.send(msg, status, p.data(), p.size())) { ok = false; break; }
            }
        });
    }
    void finish() { if (thread.joinable()) thread.join(); }
    ~SetupObserver() { finish(); strata::worker::sock::close_fd(listener); }
};

void setup_cap_parity(uint32_t cap, int delay_ms, bool bench) {
    using namespace strata::worker;
    // Prepare before any short-idle connection. One 5,222,400 B slot has a partial final chunk at
    // all three caps: 45,056 / 503,808 / 1,028,096 B. No giant upload is involved.
    NativeFmt fmt;
    std::string err;
    check(strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, fmt, err), "setup fixture layout");
    auto blob = make_blob(44, fmt);
    SetupManifest m;
    m.nonce = 71; m.model_id = 91; m.partition_id = 8;
    m.gu_type = kGu; m.d_type = kD; m.n_layers = 1; m.n_expert = 1;
    m.n_embd = kEmbd; m.n_ff = kFf; m.partition = {1};
    Svc svc(WorkerOptions{.memory_budget = (uint64_t)fmt.bytes, .workers = 2,
                          .deadline_ms = 4000, .upload_chunk_max = cap});
    if (!svc.start()) return;
    SetupObserver observer(svc.port, cap, delay_ms);
    ExpertWorkerClient cl;
    if (!cl.connect_host("127.0.0.1", observer.port, 4000, err)) { check(false, "observed connect"); return; }
    const auto start = std::chrono::steady_clock::now();
    check(cl.setup(m, err), "negotiated setup");
    check(cl.upload(0, 0, blob.data(), (uint32_t)blob.size(), err), "negotiated multi-chunk upload");
    check(cl.mark_ready(err), "whole-slot digest verified before readiness");
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    WorkerStatusInfo status;
    check(cl.status(status, err) && status.setup_weight_bytes == blob.size() && status.ready,
          "worker byte counter agrees with uploaded slot");
    BlobMap blobs; blobs[{0, 0}] = std::move(blob);
    Rng rng(33);
    for (int T : {1, 4, 8}) {
        int ids[8] = {}, mask[8]; float weights[8];
        std::fill_n(mask, 8, 1); std::fill_n(weights, 8, 0.5f);
        parity_case(cl, m, blobs, fmt, (uint64_t)T, 0, T, 1, ids, mask, weights, rng,
                    "negotiated cap T=1/4/8 bitwise output parity");
    }
    cl.close(); observer.finish();
    const uint64_t expected = ((uint64_t)fmt.bytes + cap - 1) / cap;
    check(observer.ok && observer.bytes == (uint64_t)fmt.bytes && observer.chunks == expected &&
          observer.acks == expected && observer.tail == ((uint64_t)fmt.bytes - 1) % cap + 1 &&
          observer.max_payload <= cap + kUploadHeadBytes,
          "observed data bytes/chunks/ACKs/tail and bounded payload match the negotiated cap");
    if (bench) std::printf("setup_bench cap=%u bytes=%llu chunks=%llu acks=%llu tail=%llu "
                           "upload_wire_bytes=%llu max_payload=%llu ack_delay_ms=%d setup_wall_s=%.6f\n",
                           cap, (unsigned long long)observer.bytes, (unsigned long long)observer.chunks,
                           (unsigned long long)observer.acks, (unsigned long long)observer.tail,
                           (unsigned long long)observer.wire_bytes, (unsigned long long)observer.max_payload,
                           delay_ms, seconds);
}

// A deliberately untrusted SETUP_ACK. Both old-size caps and tiny positive caps are data lengths.
// All invalid caps/lengths must fail before upload can hash/dereference its supplied pointer.
void setup_ack_controls() {
    using namespace strata::worker;
    struct Ack { uint64_t sid, bytes; uint32_t cap; size_t length; bool valid; int bad_upload_ack = 0; };
    const Ack cases[] = {{1, 7, 1, 20, true}, {1, 131073, 65536, 20, true},
        {1, 7, 0, 20, false}, {1, 7, kUploadChunkMax + 1, 20, false},
        {1, 7, UINT32_MAX, 20, false}, {1, 1ull << 32, 65536, 20, false},
        {1, UINT64_MAX, 65536, 20, false}, {1, 0, 65536, 20, false},
        {1, 7, 65536, 20, true, 1}, {1, 7, 65536, 20, true, 2},
        {1, 7, 65536, 20, true, 3}, {0, 7, 65536, 20, false}, {1, 7, 65536, 19, false}, {1, 7, 65536, 21, false}};
    SetupManifest m;
    m.n_layers = 1; m.n_expert = 1; m.n_embd = kEmbd; m.n_ff = kFf;
    m.gu_type = kGu; m.d_type = kD; m.partition = {1};
    for (const Ack& a : cases) {
        sock_fd listener = kBadFd; uint16_t port = 0; std::string err;
        if (!sock::listen_loopback(port, listener, err)) { check(false, "ACK-control listener"); return; }
        bool peer_ok = true;
        std::thread peer([&] {
            Raw raw;
            if (!sock::accept_one(listener, raw.fd, 4000)) { peer_ok = false; return; }
            uint16_t msg = 0, st = 0; std::vector<uint8_t> p;
            if (!raw.recv_reply(msg, st, p) || msg != MSG_SETUP_BEGIN) { peer_ok = false; return; }
            p.assign(a.length, 0);
            for (int i = 0; i < 8 && i < (int)p.size(); ++i) p[i] = (uint8_t)(a.sid >> (8*i));
            for (int i = 0; i < 8 && 8+i < (int)p.size(); ++i) p[8+i] = (uint8_t)(a.bytes >> (8*i));
            for (int i = 0; i < 4 && 16+i < (int)p.size(); ++i) p[16+i] = (uint8_t)(a.cap >> (8*i));
            if (!raw.send(MSG_SETUP_ACK, ST_OK, p.data(), p.size())) { peer_ok = false; return; }
            if (!a.valid) return;
            uint32_t off = 0;
            while (off < a.bytes) {
                if (!raw.recv_reply(msg, st, p) || msg != MSG_UPLOAD ||
                    p.size() != kUploadHeadBytes + std::min<uint64_t>(a.cap, a.bytes - off)) {
                    peer_ok = false; return;
                }
                off += (uint32_t)(p.size() - kUploadHeadBytes);
                p.assign(12, 0); p[0] = 1;
                for (int i = 0; i < 4; ++i) p[8+i] = (uint8_t)(off >> (8*i));
                if (a.bad_upload_ack == 1) p[0] = 2;
                if (a.bad_upload_ack == 2) ++p[8];
                if (a.bad_upload_ack == 3) p.resize(11);
                if (!raw.send(MSG_UPLOAD_ACK, ST_OK, p.data(), p.size())) { peer_ok = false; return; }
                if (a.bad_upload_ack) return;
            }
        });
        ExpertWorkerClient cl;
        check(cl.connect_host("127.0.0.1", port, 4000, err), "ACK-control connect");
        check(cl.setup(m, err) == a.valid, "validate exact SETUP_ACK before using its fields");
        if (a.valid) {
            std::vector<uint8_t> blob((size_t)a.bytes, 8);
            check(cl.upload(0, 0, blob.data(), (uint32_t)blob.size(), err) == !a.bad_upload_ack,
                  "smaller cap honored and wrong-session/offset/short upload ACKs rejected");
            if (a.bad_upload_ack) check(cl.session() == 0 && cl.last_status() == ST_PROTOCOL,
                                       "bad upload ACK invalidates client session");
        } else {
            check(cl.session() == 0 && cl.blob_bytes() == 0 && cl.last_status() == ST_PROTOCOL,
                  "bad SETUP_ACK leaves no usable session or truncated blob size");
            check(!cl.upload(0, 0, nullptr, 7, err), "invalid ACK cannot reach upload allocation or digest");
        }
        cl.close(); peer.join(); sock::close_fd(listener);
        check(peer_ok, "fake peer observed bounded upload lengths");
    }
    for (uint32_t cap : {0u, kUploadChunkMax + 1, UINT32_MAX}) {
        ExpertWorkerService service(WorkerOptions{.upload_chunk_max = cap});
        uint16_t port = 0; std::string err;
        check(!service.listen(port, err), "worker refuses zero/oversized cap before frame arithmetic");
    }
}

void setup_upload_controls() {
    using namespace strata::worker;
    NativeFmt fmt; std::string err;
    strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, fmt, err);
    auto blob = make_blob(7, fmt);
    SetupManifest m;
    m.n_layers = 1; m.n_expert = 1; m.n_embd = kEmbd; m.n_ff = kFf;
    m.gu_type = kGu; m.d_type = kD; m.partition = {1};
    std::vector<uint8_t> begin;
    encode_setup_begin(m, begin);
    const Digest128 d = blob_digest(blob);
    for (uint32_t cap : {65536u, 512u*1024, 1024u*1024}) {
        Svc svc(WorkerOptions{.memory_budget = (uint64_t)fmt.bytes, .workers = 2,
                             .deadline_ms = 250, .upload_chunk_max = cap});
        if (!svc.start()) return;
        // Declared frame lengths above the per-message cap, including UINT64_MAX, close from just
        // the 16-byte header. The peer sends no payload, so allocating/reading that length would fail.
        for (const auto& malformed : std::vector<std::pair<uint16_t, uint64_t>>{
                {MSG_UPLOAD, kUploadHeadBytes + cap + 1}, {MSG_UPLOAD, UINT64_MAX},
                {MSG_INFER, kInferPayloadMax + 1}}) {
            const auto [msg, length] = malformed;
            Raw raw; check(raw.conn(svc.port), "oversized header connect");
            std::vector<uint8_t> h;
            encode_frame(h, msg, 0, nullptr, 0);
            for (int i = 0; i < 8; ++i) h[8+i] = (uint8_t)(length >> (8*i));
            check(sock::send_all(raw.fd, h.data(), h.size(), 2000), "send oversized header only");
            uint8_t byte;
            check(!sock::recv_all(raw.fd, &byte, 1, 2000), "oversized length closes without payload");
        }
        Raw raw; check(raw.conn(svc.port), "upload-control connect");
        raw.send(MSG_SETUP_BEGIN, 0, begin.data(), begin.size());
        uint64_t sid = 0; if (!setup_ack(raw, sid)) return;
        auto rejected = [&](uint32_t off, uint32_t length, size_t cut) {
            auto p = upload_frame(sid, 0, 0, (uint32_t)fmt.bytes, off, length, d, blob.data());
            p.resize(p.size() - cut);
            uint16_t msg = 0, status = 0; std::vector<uint8_t> reply;
            check(raw.send(MSG_UPLOAD, 0, p.data(), p.size()) && raw.recv_reply(msg, status, reply) &&
                  msg == MSG_UPLOAD_ACK && status == ST_BAD_REQUEST, "zero/offset-overflow/truncated chunk refused");
        };
        rejected(0, 0, 0); rejected(UINT32_MAX, 1, 0); rejected(0, cap, 1);
        // A full-sized frame cut on the socket expires; that upload must not survive reconnect.
        auto p = upload_frame(sid, 0, 0, (uint32_t)fmt.bytes, 0, cap, d, blob.data());
        std::vector<uint8_t> wire; encode_frame(wire, MSG_UPLOAD, 0, p.data(), p.size());
        check(sock::send_all(raw.fd, wire.data(), 100, 2000), "send partial full-cap upload");
        uint8_t byte;
        check(!sock::recv_all(raw.fd, &byte, 1, 2000), "truncated upload expires under the existing budget");
        sock::close_fd(raw.fd);
        Raw next; check(next.conn(svc.port), "reconnect after cut upload");
        uint8_t old[8]; for (int i = 0; i < 8; ++i) old[i] = (uint8_t)(sid >> (8*i));
        uint16_t msg = 0, status = 0; std::vector<uint8_t> reply;
        check(next.send(MSG_READY, 0, old, sizeof old) && next.recv_reply(msg, status, reply) &&
              status == ST_WRONG_STATE, "truncated upload invalidated its session");
    }
}
