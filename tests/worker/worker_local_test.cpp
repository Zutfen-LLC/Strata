// Real client, service, native index and GGUF fixtures. No external model is needed.
#include "ggml.h"
#include "strata/worker/expert_tier.hpp"
#include "strata/worker/expert_worker.hpp"
#include "strata/worker/local_source.hpp"
#include "wire_sock.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
using namespace strata::worker;
namespace fs = std::filesystem;
namespace {
int failures = 0;
void check(bool b, const char *s) {
    std::printf("%s %s\n", b ? "PASS" : "FAIL", s);
    if (!b)
        ++failures;
}
void require(bool b, const std::string &s) {
    if (!b)
        throw std::runtime_error(s);
}
template <class T> void put(std::vector<uint8_t> &b, T v) {
    for (size_t i = 0; i < sizeof(T); ++i)
        b.push_back((uint8_t)(v >> (8 * i)));
}
void str(std::vector<uint8_t> &b, const std::string &s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
std::vector<uint8_t> blob(uint32_t seed, const strata::kernels::cpu::NativeFmt &f) {
    std::vector<uint8_t> b(f.bytes);
    std::vector<float> row(2560);
    auto pack = [&](size_t rows, size_t cols, size_t stride, uint8_t *dst) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t i = 0; i < cols; ++i) {
                seed = seed * 1664525 + 1013904223;
                row[i] = ((int32_t)(seed >> 9) / (float)(1 << 23)) - 0.5f;
            }
            ggml_quantize_chunk(GGML_TYPE_Q8_0, row.data(), dst + r * stride, 0, 1, cols, nullptr);
        }
    };
    pack(f.n_ff, f.n_embd, f.gu_row, b.data());
    pack(f.n_ff, f.n_embd, f.gu_row, b.data() + f.up_off);
    pack(f.n_embd, f.n_ff, f.d_row, b.data() + f.down_off);
    return b;
}
struct Fixture {
    fs::path root;
    strata::kernels::cpu::NativeFmt fmt;
    std::array<std::vector<uint8_t>, 4> blobs;
    LocalManifest m;
    std::string index;
    Fixture() {
        char path[] = "/tmp/strata-worker-local-XXXXXX";
        const auto r = ::mkdtemp(path);
        require(r, "mkdtemp");
        root = r;
        std::string err;
        require(strata::kernels::cpu::native_fmt(8, 8, 2560, 640, fmt, err), err);
        for (size_t i = 0; i < 4; ++i)
            blobs[i] = blob(17 + i, fmt);
        struct Tensor {
            int l, r;
            uint64_t off;
        };
        std::array<std::vector<Tensor>, 3> tables{
            {{{0, 0, 0}, {0, 1, 0}, {0, 2, 0}}, {{1, 0, 0}, {1, 1, 0}}, {{1, 2, 0}}}};
        const char *roles[] = {"gate", "up", "down"};
        uint64_t offsets[2][3]{};
        const uint64_t per[] = {fmt.up_off, fmt.up_off, fmt.bytes - fmt.down_off};
        const size_t src[] = {0, fmt.up_off, fmt.down_off};
        const char *names[] = {"a.gguf", "b.gguf", "c.gguf"};
        for (size_t fi = 0; fi < tables.size(); ++fi) {
            auto &ts = tables[fi];
            uint64_t off = 0;
            std::vector<uint8_t> header;
            put<uint32_t>(header, 0x46554747);
            put<uint32_t>(header, 3);
            put<uint64_t>(header, ts.size());
            put<uint64_t>(header, 0);
            for (auto &t : ts) {
                t.off = off;
                off += 2 * per[t.r];
                str(header, "blk." + std::to_string(t.l) + ".ffn_" + roles[t.r] + "_exps.weight");
                put<uint32_t>(header, 3);
                put<uint64_t>(header, t.r < 2 ? 2560 : 640);
                put<uint64_t>(header, t.r < 2 ? 640 : 2560);
                put<uint64_t>(header, 2);
                put<uint32_t>(header, 8);
                put<uint64_t>(header, t.off);
            }
            header.resize((header.size() + 31) / 32 * 32);
            std::ofstream out(root / names[fi], std::ios::binary);
            out.write((const char *)header.data(), header.size());
            for (auto t : ts) {
                offsets[t.l][t.r] = header.size() + t.off;
                for (int e = 0; e < 2; ++e)
                    out.write((const char *)blobs[t.l * 2 + e].data() + src[t.r], per[t.r]);
            }
        }
        index =
            "# strata native experts v4: fixture (n_expert 2, total " + std::to_string(4 * fmt.bytes) + ")\n";
        for (int l = 0; l < 2; ++l)
            index += std::to_string(l) + " 8 8 " + std::to_string(l * 2 * fmt.bytes) + " " +
                     std::to_string(fmt.bytes) + " " + std::to_string(offsets[l][0]) + " " +
                     std::to_string(offsets[l][1]) + " " + std::to_string(offsets[l][2]) +
                     (l == 0 ? "\n" : " b.gguf,b.gguf,c.gguf\n");
        std::ofstream(root / "native_experts.txt") << index;
        require(prepare_local_manifest(root.string(), (root / "a.gguf").string(), 2, 2, m, err), err);
        require(write_local_manifest((root / "local.manifest").string(), m, err), err);
        require(read_local_manifest((root / "local.manifest").string(), m, err), err);
        require(match_local_index(m, root.string(), (root / "a.gguf").string(),
                                  strata::kernels::cpu::expert_layout(), err),
                err);
    }
    ~Fixture() { fs::remove_all(root); }
    SetupManifest setup() const {
        SetupManifest s;
        s.nonce = 123;
        s.model_id = m.identity.a;
        std::strcpy(s.quant_name, "Q8_0");
        s.n_layers = 2;
        s.n_expert = 2;
        s.n_embd = 2560;
        s.n_ff = 640;
        s.gu_type = s.d_type = 8;
        s.partition = {3, 3};
        s.partition_id = blob_digest(s.partition).a;
        return s;
    }
    WorkerOptions options() const {
        WorkerOptions o;
        o.workers = 2;
        o.local_root = root.string();
        o.local_manifest = (root / "local.manifest").string();
        o.local_upload_fallback = true;
        o.memory_budget = 4 * fmt.bytes;
        return o;
    }
    void flip(const char *name, uint64_t offset) {
        std::fstream f(root / name, std::ios::binary | std::ios::in | std::ios::out);
        char c;
        f.seekg(offset);
        f.get(c);
        f.seekp(offset);
        f.put(c ^ 1);
    }
};
struct Service {
    ExpertWorkerService s;
    uint16_t port = 0;
    std::thread t;
    explicit Service(WorkerOptions o) : s(o) {
        std::string err;
        require(s.listen(port, err), err);
        t = std::thread([&] { s.run(); });
    }
    ~Service() {
        s.stop();
        t.join();
    }
};
void connect(ExpertWorkerClient &c, Service &s) {
    std::string err;
    require(c.connect_host("127.0.0.1", s.port, 5000, err), err);
}
void setup(ExpertWorkerClient &c, const Fixture &f, bool fallback = true) {
    std::string err;
    require(c.setup_local(f.setup(), f.m.identity, fallback, err), err);
}
std::vector<float> infer(ExpertWorkerClient &c, const Fixture &f, int l, int T, uint64_t seq) {
    auto m = f.setup();
    InferRequest r;
    r.session = c.session();
    r.model_id = m.model_id;
    r.partition_id = m.partition_id;
    r.sequence = seq;
    r.layer = l;
    r.T = T;
    r.K = 2;
    r.mask.assign(T * 2, 1);
    r.ids.resize(T * 2);
    r.weights.resize(T * 2);
    r.activations.resize(T * 2560);
    for (int i = 0; i < T * 2; ++i) {
        r.ids[i] = i % 2;
        r.weights[i] = i % 2 ? 0.3f : 0.7f;
    }
    for (size_t i = 0; i < r.activations.size(); ++i)
        r.activations[i] = 0.1f * (int(i % 11) - 5);
    InferReply reply;
    std::string err;
    require(c.infer(r, reply, err), err);
    std::vector<const uint8_t *> pointers(T * 2);
    for (int i = 0; i < T * 2; ++i)
        pointers[i] = f.blobs[l * 2 + i % 2].data();
    std::vector<float> direct;
    require(ExpertWorkerService::direct_native_partial(f.fmt, T, 2, r.mask.data(), r.weights.data(),
                                                       r.activations.data(), pointers.data(), direct, err),
            err);
    check(direct.size() == reply.partials.size() &&
              !std::memcmp(direct.data(), reply.partials.data(), direct.size() * 4),
          "transport equals direct native bits");
    return reply.partials;
}
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
void binary(Fixture &f, const char *executable) {
    int pipefd[2];
    require(::pipe(pipefd) == 0, "pipe");
    const auto pid = ::fork();
    require(pid >= 0, "fork");
    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], 2);
        ::close(pipefd[1]);
        const std::string root = f.root.string(), manifest = (f.root / "local.manifest").string();
        const char *args[] = {executable,
                              "--port",
                              "0",
                              "--workers",
                              "2",
                              "--memory-budget-mib",
                              "32",
                              "--local-root",
                              root.c_str(),
                              "--local-manifest",
                              manifest.c_str(),
                              nullptr};
        ::execv(executable, const_cast<char **>(args));
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
    std::printf("%s", log.c_str());
}
} // namespace
int main(int argc, char **argv) {
    try {
        Fixture f;
        if (argc == 3 && std::string(argv[1]) == "--binary")
            binary(f, argv[2]);
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
