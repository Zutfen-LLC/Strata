#pragma once
#ifdef STRATA_TEST_TLS
#include "tls_fixture.hpp"
inline TlsFixture* transport_certificates = nullptr;
#endif
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
#include <cstdlib>
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
    std::vector<std::vector<uint8_t>> blobs;
    int n_expert = 2;
    LocalManifest m;
    std::string index;
    explicit Fixture(int experts = 2, int paired_from = -1) {
        require(experts >= 1 && experts <= 32, "bounded synthetic expert count");
        require(paired_from == -1 || (paired_from >= 0 && paired_from < experts && paired_from % 2 == 0),
                "paired synthetic range starts on an even expert");
        n_expert = experts;
        blobs.resize(2 * experts);
        char path[] = "/tmp/strata-worker-local-XXXXXX";
        const auto r = ::mkdtemp(path);
        require(r, "mkdtemp");
        root = r;
        struct Construction {
            fs::path root; bool complete = false;
            ~Construction() { if (!complete) { std::error_code error; fs::remove_all(root, error); } }
        } construction{root};
        std::string err;
        require(strata::kernels::cpu::native_fmt(8, 8, 2560, 640, fmt, err), err);
        for (size_t i = 0; i < blobs.size(); ++i)
            blobs[i] = paired_from >= 0 && i % 2 && int(i % n_expert) >= paired_from ?
                blobs[i - 1] : blob(17 + i, fmt);
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
                off += n_expert * per[t.r];
                str(header, "blk." + std::to_string(t.l) + ".ffn_" + roles[t.r] + "_exps.weight");
                put<uint32_t>(header, 3);
                put<uint64_t>(header, t.r < 2 ? 2560 : 640);
                put<uint64_t>(header, t.r < 2 ? 640 : 2560);
                put<uint64_t>(header, n_expert);
                put<uint32_t>(header, 8);
                put<uint64_t>(header, t.off);
            }
            header.resize((header.size() + 31) / 32 * 32);
            std::ofstream out(root / names[fi], std::ios::binary);
            out.write((const char *)header.data(), header.size());
            for (auto t : ts) {
                offsets[t.l][t.r] = header.size() + t.off;
                for (int e = 0; e < n_expert; ++e)
                    out.write((const char *)blobs[t.l * n_expert + e].data() + src[t.r], per[t.r]);
            }
        }
        index =
            "# strata native experts v4: fixture (n_expert " + std::to_string(n_expert) + ", total " +
            std::to_string(blobs.size() * fmt.bytes) + ")\n";
        for (int l = 0; l < 2; ++l)
            index += std::to_string(l) + " 8 8 " + std::to_string(l * n_expert * fmt.bytes) + " " +
                     std::to_string(fmt.bytes) + " " + std::to_string(offsets[l][0]) + " " +
                     std::to_string(offsets[l][1]) + " " + std::to_string(offsets[l][2]) +
                     (l == 0 ? "\n" : " b.gguf,b.gguf,c.gguf\n");
        std::ofstream(root / "native_experts.txt") << index;
        require(prepare_local_manifest(root.string(), (root / "a.gguf").string(), 2, n_expert, m, err), err);
        require(write_local_manifest((root / "local.manifest").string(), m, err), err);
        require(read_local_manifest((root / "local.manifest").string(), m, err), err);
        require(match_local_index(m, root.string(), (root / "a.gguf").string(),
                                  strata::kernels::cpu::expert_layout(), err),
                err);
        construction.complete = true;
    }
    ~Fixture() {
        if (std::getenv("STRATA_WORKER_FIXTURE_KEEP"))
            std::fprintf(stderr, "retained synthetic fixture: %s\n", root.c_str());
        else
            fs::remove_all(root);
    }
    SetupManifest setup() const {
        SetupManifest s;
        s.nonce = 123;
        s.model_id = m.identity.a;
        std::strcpy(s.quant_name, "Q8_0");
        s.n_layers = 2;
        s.n_expert = n_expert;
        s.n_embd = 2560;
        s.n_ff = 640;
        s.gu_type = s.d_type = 8;
        const size_t stride = (n_expert + 7) / 8;
        s.partition.assign(2 * stride, 0);
        for (int l = 0; l < 2; ++l)
            for (int e = 0; e < n_expert; ++e)
                s.partition[l * stride + e / 8] |= (uint8_t)(1u << (e % 8));
        s.partition_id = blob_digest(s.partition).a;
        return s;
    }
    WorkerOptions options() const {
        WorkerOptions o;
#ifdef STRATA_TEST_TLS
        if(transport_certificates) o.tls=transport_certificates->config("worker","coordinator.test");
#endif
        o.workers = 2;
        o.local_root = root.string();
        o.local_manifest = (root / "local.manifest").string();
        o.local_upload_fallback = true;
        o.memory_budget = blobs.size() * fmt.bytes;
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
#ifdef STRATA_TEST_TLS
    if(transport_certificates) {
        require(c.connect_transport("127.0.0.1",s.port,5000,{},
                transport_certificates->config("coordinator","worker.test"),false,err),err); return;
    }
#endif
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
        pointers[i] = f.blobs[l * f.n_expert + i % 2].data();
    std::vector<float> direct;
    require(ExpertWorkerService::direct_native_partial(f.fmt, T, 2, r.mask.data(), r.weights.data(),
                                                       r.activations.data(), pointers.data(), direct, err),
            err);
    check(direct.size() == reply.partials.size() &&
              !std::memcmp(direct.data(), reply.partials.data(), direct.size() * 4),
          "transport equals direct native bits");
    return reply.partials;
}
} // namespace
