#include "local_fixture.hpp"
namespace {
void parity(Fixture &f) {
    std::vector<std::vector<float>> control;
    for (bool local : {false, true}) {
        Service s(f.options());
        ExpertWorkerClient c;
        connect(c, s);
        std::string err;
        if (local)
            setup(c, f);
        else
            require(c.setup(f.setup(), err), err);
        if (local) {
            std::vector<uint8_t> hits;
            require(c.load_local({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, hits, err), err);
            check(hits == std::vector<uint8_t>({1, 1, 1, 1}), "all local hits including split-role layer");
            check(!c.load_local({{0, 0}}, hits, err) && c.last_status() == ST_DUPLICATE,
                  "local duplicate slot refused");
        } else
            for (int l = 0; l < 2; ++l)
                for (int e = 0; e < 2; ++e)
                    require(c.upload(l, e, f.blobs[l * 2 + e].data(), f.fmt.bytes, err), err);
        require(c.mark_ready(err), err);
        WorkerStatusInfo status;
        require(c.status(status, err), err);
        check(status.ready && status.setup_weight_bytes == (local ? 0 : 4 * f.fmt.bytes),
              local ? "all-local sends ZERO network weight bytes" : "legacy upload weight count");
        uint64_t seq = 0;
        for (int l = 0; l < 2; ++l)
            for (int T : {1, 4, 8}) {
                auto got = infer(c, f, l, T, ++seq);
                if (local)
                    check(!std::memcmp(got.data(), control[seq - 1].data(), got.size() * 4),
                          "upload/local inference parity T=1/4/8");
                else
                    control.push_back(got);
            }
        if (local) {
            const auto &st = c.local_stats();
            check(st.bytes == 4 * f.fmt.bytes && st.hits == 4 && !st.misses && st.read_ns && st.verify_ns,
                  "local bytes/hits/phase counters");
            std::printf("LOCAL_FIXTURE bytes=%llu network=%llu read_ns=%llu verify_ns=%llu setup_ns=%llu\n",
                        (unsigned long long)st.bytes, (unsigned long long)status.setup_weight_bytes,
                        (unsigned long long)st.read_ns, (unsigned long long)st.verify_ns,
                        (unsigned long long)st.setup_ns);
        }
    }
}
void mixed(Fixture &f) {
    fs::rename(f.root / "b.gguf", f.root / "b.saved");
    {
        Service s(f.options());
        ExpertWorkerClient c;
        connect(c, s);
        setup(c, f);
        std::string err;
        std::vector<uint8_t> hits;
        require(c.load_local({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, hits, err), err);
        check(hits == std::vector<uint8_t>({1, 1, 0, 0}), "missing-only local/missing split");
        check(!c.mark_ready(err) && c.last_status() == ST_INCOMPLETE, "missing set cannot become ready");
        check(!c.upload(0, 0, f.blobs[0].data(), f.fmt.bytes, err),
              "local hits cannot be overwritten by upload");
        auto wrong = f.blobs[2];
        wrong[0] ^= 1;
        check(!c.upload(1, 0, wrong.data(), f.fmt.bytes, err), "fallback bytes must match trusted manifest");
        for (int e = 0; e < 2; ++e)
            require(c.upload(1, e, f.blobs[2 + e].data(), f.fmt.bytes, err), err);
        require(c.mark_ready(err), err);
        infer(c, f, 1, 4, 1);
        WorkerStatusInfo status;
        require(c.status(status, err), err);
        check(status.setup_weight_bytes == 2 * f.fmt.bytes && c.local_stats().hits == 2 &&
                  c.local_stats().misses == 2,
              "network contains only missing experts");
    }
    // A present corrupt role must fail even when another role of the same expert is absent.
    auto off = f.m.layers[1][2].offset;
    f.flip("c.gguf", off);
    {
        Service s(f.options());
        ExpertWorkerClient c;
        connect(c, s);
        setup(c, f);
        std::string err;
        std::vector<uint8_t> hits;
        check(!c.load_local({{1, 0}}, hits, err) && c.last_status() == ST_DIGEST_MISMATCH,
              "corrupt present role is not hidden by missing role");
        check(!c.mark_ready(err), "corrupt partial source cannot become ready");
    }
    f.flip("c.gguf", off);
    fs::rename(f.root / "b.saved", f.root / "b.gguf");
}
void refusals(Fixture &f) {
    std::string err;
    std::vector<uint8_t> hits;
    {
        WorkerOptions o = f.options();
        o.local_root.clear();
        o.local_manifest.clear();
        Service s(o);
        ExpertWorkerClient c;
        connect(c, s);
        check(!c.setup_local(f.setup(), f.m.identity, false, err) && c.last_status() == ST_LOCAL_UNAVAILABLE,
              "new upload-only worker refuses local mode clearly");
        require(c.setup(f.setup(), err), err);
        check(!c.mark_ready(err), "default upload setup still needs bytes");
    }
    {
        Service s(f.options());
        ExpertWorkerClient c;
        connect(c, s);
        auto m = f.setup();
        m.model_id ^= 1;
        check(!c.setup_local(m, f.m.identity, false, err) && c.last_status() == ST_LOCAL_INVALID,
              "wrong model refused");
        auto id = f.m.identity;
        id.b ^= 1;
        check(!c.setup_local(f.setup(), id, false, err), "wrong full manifest identity refused");
        m = f.setup();
        m.partition_id ^= 1;
        check(!c.setup_local(m, f.m.identity, false, err), "inconsistent ownership identity refused");
        setup(c, f);
        check(!c.load_local({{0, 0}, {0, 0}}, hits, err) && c.last_status() == ST_DUPLICATE,
              "duplicate ownership list refused before reads");
        check(!c.mark_ready(err), "partial local setup cannot become ready");
    }
    {
        WorkerOptions o = f.options();
        o.memory_budget = 4 * f.fmt.bytes - 1;
        Service s(o);
        ExpertWorkerClient c;
        connect(c, s);
        check(!c.setup_local(f.setup(), f.m.identity, false, err) && c.last_status() == ST_OVER_BUDGET,
              "assigned arena memory limit enforced");
    }
    {
        WorkerOptions o = f.options();
        o.local_setup_ms = 5;
        Service s(o);
        ExpertWorkerClient c;
        connect(c, s);
        setup(c, f);
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        check(!c.load_local({{0, 0}}, hits, err) && c.last_status() == ST_DEADLINE,
              "total setup deadline enforced");
        check(!c.mark_ready(err), "deadline failure cannot become ready");
    }
    for (int mode = 0; mode < 4; ++mode) {
        fs::rename(f.root / "a.gguf", f.root / "a.saved");
        if (mode == 0) {
            fs::copy_file(f.root / "a.saved", f.root / "a.gguf");
            fs::resize_file(f.root / "a.gguf", 32);
        }
        if (mode == 1)
            fs::create_symlink(f.root / "a.saved", f.root / "a.gguf");
        if (mode == 2)
            fs::create_directory(f.root / "a.gguf");
        if (mode == 3) {
            fs::copy_file(f.root / "a.saved", f.root / "a.gguf");
            f.flip("a.gguf", f.m.layers[0][0].offset);
        }
        {
            Service s(f.options());
            ExpertWorkerClient c;
            connect(c, s);
            setup(c, f);
            check(!c.load_local({{0, 0}}, hits, err) &&
                      c.last_status() == (mode == 3 ? ST_DIGEST_MISMATCH : ST_LOCAL_INVALID),
                  "truncation/symlink/directory/corruption is fatal, not absent");
            check(!c.mark_ready(err), "invalid source cannot become ready");
        }
        fs::remove_all(f.root / "a.gguf");
        fs::rename(f.root / "a.saved", f.root / "a.gguf");
    }
    auto invalid = f.m;
    invalid.layers[0][0].offset = UINT64_MAX;
    check(!write_local_manifest((f.root / "bad.manifest").string(), invalid, err),
          "overflowed slices rejected");
    invalid = f.m;
    invalid.layers[0][0].offset = f.m.files[0].bytes;
    check(!write_local_manifest((f.root / "bad.manifest").string(), invalid, err),
          "out-of-bounds slices rejected");
    invalid = f.m;
    invalid.files[0].name = "../a.gguf";
    check(!write_local_manifest((f.root / "bad.manifest").string(), invalid, err), "path escape rejected");
    invalid = f.m;
    invalid.layers[0][1] = invalid.layers[0][0];
    check(!write_local_manifest((f.root / "bad.manifest").string(), invalid, err),
          "overlapping role ranges rejected");
    fs::copy_file(f.root / "local.manifest", f.root / "bad.manifest", fs::copy_options::overwrite_existing);
    fs::resize_file(f.root / "bad.manifest", fs::file_size(f.root / "bad.manifest") - 1);
    check(!read_local_manifest((f.root / "bad.manifest").string(), invalid, err),
          "truncated digest manifest rejected");
    std::ofstream(f.root / "native_experts.txt", std::ios::app) << "# changed\n";
    check(!match_local_index(f.m, f.root.string(), (f.root / "a.gguf").string(),
                             strata::kernels::cpu::expert_layout(), err),
          "changed native index invalidates reuse");
    std::ofstream(f.root / "native_experts.txt") << f.index;
}
void tier(Fixture &f) {
    for (int mode = 0; mode < 3; ++mode) {
        if (mode == 1)
            fs::rename(f.root / "b.gguf", f.root / "b.saved");
        {
            Service s(f.options());
            ExpertWorkerTier t;
            ExpertWorkerTier::Open o;
            o.host = "127.0.0.1";
            o.port = s.port;
            o.n_layers = 2;
            o.n_expert = 2;
            o.n_embd = 2560;
            o.n_ff = 640;
            o.partition = {3, 3};
            o.local = &f.m;
            o.local_fallback = mode == 1;
            int fetched = 0;
            o.fetch = [&](int64_t l, int64_t e, uint8_t *dst, uint32_t n) {
                ++fetched;
                std::memcpy(dst, f.blobs[l * 2 + e].data(), n);
                return true;
            };
            if (mode == 2)
                o.cancelled = [] { return true; };
            std::string err;
            const auto ok = t.open(o, err);
            check(ok == (mode != 2), "real engine tier local setup / cancellation");
            check(fetched == (mode == 1 ? 2 : 0), "tier fetches zero local hits and only missing blobs");
            if (ok) {
                check(t.stats().setup_network_bytes == (mode == 1 ? 2 * f.fmt.bytes : 0),
                      "tier setup network evidence");
                std::vector<float> x(2560, 0.2f);
                int32_t ids[] = {0, 1};
                float w[] = {0.5f, 0.5f};
                uint8_t mask[] = {1, 1};
                require(t.post(1, x.data(), ids, w, mask, 1, 2, err) && t.finish(err), err);
                check(t.served(), "tier inference after local readiness");
            } else
                check(!t.ready(), "cancelled tier cannot infer");
        }
        if (mode == 1)
            fs::rename(f.root / "b.saved", f.root / "b.gguf");
    }
    {
        Service s(f.options());
        ExpertWorkerClient c;
        connect(c, s);
        setup(c, f);
        auto old = c.session();
        c.close();
        connect(c, s);
        std::string err;
        check(!c.mark_ready(err), "disconnect clears client session");
        setup(c, f);
        check(c.session() != old, "reconnect requires a new setup session");
        check(!c.mark_ready(err), "reconnect does not retain local readiness");
    }
}
void sparse(Fixture &f) {
    const auto off = f.m.layers[0][0].offset + f.m.layers[0][0].bytes;
    f.flip("a.gguf", off);
    {
        Service s(f.options());
        ExpertWorkerClient c;
        connect(c, s);
        auto m = f.setup();
        m.partition = {1, 2};
        m.partition_id = blob_digest(m.partition).a;
        std::string err;
        require(c.setup_local(m, f.m.identity, false, err), err);
        std::vector<uint8_t> hits;
        check(!c.load_local({{0, 1}}, hits, err) && c.last_status() == ST_NOT_OWNED,
              "local load refuses unassigned expert");
        require(c.load_local({{0, 0}, {1, 1}}, hits, err), err);
        require(c.mark_ready(err), err);
        check(c.local_stats().bytes == 2 * f.fmt.bytes && c.local_stats().hits == 2,
              "sparse partition reads ONLY assigned bytes; unassigned corruption is untouched");
    }
    f.flip("a.gguf", off);
}
void negotiation_budget(Fixture &f) {
    using namespace strata::worker;
    for (int mode = 0; mode < 4; ++mode) {
        uint16_t port = 0;
        sock_fd listener = kBadFd;
        std::string err;
        require(sock::listen_loopback(port, listener, err), err);
        std::atomic<bool> cancel{mode == 3}, accepted{false};
        std::thread peer([&] {
            sock_fd fd = kBadFd;
            if (sock::accept_one(listener, fd, 200)) {
                accepted = true;
                uint8_t head[16];
                if (sock::recv_all(fd, head, 16, 300)) {
                    uint64_t n = 0;
                    for (int i = 0; i < 8; ++i)
                        n |= (uint64_t)head[8 + i] << (8 * i);
                    std::vector<uint8_t> p(n);
                    sock::recv_all(fd, p.data(), p.size(), 300);
                    if (mode == 1) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(40));
                        std::vector<uint8_t> reply;
                        encode_frame(reply, MSG_LOCAL_SETUP_ACK, ST_OK, nullptr, 0);
                        reply[8] = 28;
                        sock::send_all(fd, reply.data(), reply.size(), 100);
                    }
                    if (mode == 2) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(25));
                        cancel = true;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                sock::close_fd(fd);
            }
        });
        ExpertWorkerTier t;
        ExpertWorkerTier::Open o;
        o.host = "127.0.0.1";
        o.port = port;
        o.n_layers = 2;
        o.n_expert = 2;
        o.n_embd = 2560;
        o.n_ff = 640;
        o.partition = {3, 3};
        o.local = &f.m;
        o.setup_ms = mode >= 2 ? 1000 : 60;
        o.deadline_ms = 1000;
        o.cancelled = [&] { return cancel.load(); };
        const auto start = std::chrono::steady_clock::now();
        const bool ok = t.open(o, err);
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                .count();
        check(!ok && !t.ready() && ms < 180, "negotiation header/payload obey total budget and cancellation");
        peer.join();
        sock::close_fd(listener);
        if (mode == 3)
            check(!accepted, "already-cancelled setup does not connect or allocate worker arena");
        std::printf("NEGOTIATION mode=%d elapsed_ms=%lld\n", mode, (long long)ms);
    }
}
void binary(Fixture &f, const char *executable, const char *loading_threads = nullptr) {
    int pipefd[2];
    require(::pipe(pipefd) == 0, "pipe");
    const auto pid = ::fork();
    require(pid >= 0, "fork");
    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], 2);
        ::close(pipefd[1]);
        const std::string root = f.root.string(), manifest = (f.root / "local.manifest").string();
        std::vector<const char *> args{executable,
                              "--port",
                              "0",
                              "--workers",
                              "2",
                              "--memory-budget-mib",
                              "32",
                              "--local-root",
                              root.c_str(),
                              "--local-manifest",
                              manifest.c_str()};
        if (loading_threads) {
            args.push_back("--local-load-threads");
            args.push_back(loading_threads);
        }
        args.push_back(nullptr);
        ::execv(executable, const_cast<char **>(args.data()));
        _exit(127);
    }
    ::close(pipefd[1]);
    uint16_t port = 0;
    std::string log;
    char buf[512];
    while (!port) {
        const auto n = ::read(pipefd[0], buf, sizeof buf);
        if (n <= 0)
            break;
        log.append(buf, n);
        const auto at = log.find("listening on 127.0.0.1:");
        if (at != std::string::npos)
            port = std::stoi(log.substr(at + 23));
    }
    bool done = false;
    try {
        require(port != 0, "worker binary did not listen: " + log);
        ExpertWorkerClient c;
        std::string err;
        require(c.connect_host("127.0.0.1", port, 5000, err), err);
        setup(c, f, false);
        std::vector<uint8_t> hits;
        require(c.load_local({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, hits, err), err);
        require(c.mark_ready(err), err);
        infer(c, f, 1, 8, 1);
        WorkerStatusInfo status;
        require(c.status(status, err), err);
        check(status.setup_weight_bytes == 0 && c.local_stats().bytes == 4 * f.fmt.bytes,
              "actual binary CLI local-load smoke: zero network weights");
        require(c.shutdown(err), err);
        c.close();
        done = true;
    } catch (...) {
        ::kill(pid, SIGTERM);
        ::waitpid(pid, nullptr, 0);
        ::close(pipefd[0]);
        throw;
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    while (auto n = ::read(pipefd[0], buf, sizeof buf)) {
        if (n < 0)
            break;
        log.append(buf, n);
    }
    ::close(pipefd[0]);
    check(done && WIFEXITED(status) && WEXITSTATUS(status) == 0, "actual worker binary clean shutdown");
    const int expected_lanes = loading_threads ? std::stoi(loading_threads) : 4;
    if (expected_lanes > 1)
        check(log.find("local loading lanes=" + std::to_string(expected_lanes) + ";") != std::string::npos,
              "omitted loading flag selects four lanes; explicit parallel count overrides it");
    else
        check(log.find("local loading lanes=") == std::string::npos,
              "explicit one loading thread retains the serial path");
    check(log.find("2 pool workers") != std::string::npos, "loading selection leaves inference worker count separate");
    std::printf("%s", log.c_str());
}
} // namespace
int main(int argc, char **argv) {
    try {
        check(WorkerOptions{}.local_load_threads == 4 && WorkerOptions{}.workers == 0,
              "service loading default is four; inference worker default is unchanged");
        Fixture f;
        if (argc == 3 && std::string(argv[1]) == "--binary")
            binary(f, argv[2]);
        else if (argc == 5 && std::string(argv[1]) == "--binary" && std::string(argv[3]) == "--load-threads")
            binary(f, argv[2], argv[4]);
        else {
            parity(f);
            mixed(f);
            refusals(f);
            sparse(f);
            tier(f);
            negotiation_budget(f);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        ++failures;
    }
    std::printf("worker_local_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
