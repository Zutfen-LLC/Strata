// tests/worker/expert_worker_e2e_test.cpp - inferswarm #348: the minimal local reproduction, as a test.
//
// Spawns the real strata-expert-worker binary on a free loopback port, drives it through the client
// library (setup, bounded chunk upload, readiness, one mixed-ownership request) and checks the
// transported partial sums bitwise against the direct native CPU call - the same flow
// docs/CPU_WORKER.md hands to the engine-integration issue (#349).
#include "strata/worker/expert_client.hpp"
#include "strata/worker/expert_worker.hpp"
#include "strata/worker/expert_wire.hpp"

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/wait.h>

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}

constexpr uint16_t kGu = 8, kD = 8, kLayers = 2, kExperts = 8, kEmbd = 2560, kFf = 640;

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 1u) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return (float) ((int32_t) (s >> 9)) / (float) (1 << 23) - 1.f;
    }
    void fill(float* v, size_t n) {
        for (size_t i = 0; i < n; ++i) v[i] = next();
    }
};

}  // namespace

int main() {
#ifndef STRATA_WORKER_BIN
    std::fprintf(stderr, "e2e: STRATA_WORKER_BIN not defined\n");
    return 1;
#else
    // spawn the worker with its stderr on a pipe and read the port from its start-up line
    int pipefd[2];
    if (::pipe(pipefd) != 0) return 1;
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], 2);
        ::close(pipefd[1]);
        const char* argv[] = {STRATA_WORKER_BIN, "--port", "0", "--workers", "2", nullptr};
        ::execv(STRATA_WORKER_BIN, const_cast<char**>(argv));
        _exit(127);
    }
    ::close(pipefd[1]);
    uint16_t port = 0;
    {
        char buf[256];
        std::string acc;
        for (;;) {
            const ssize_t n = ::read(pipefd[0], buf, sizeof buf);
            if (n <= 0) break;
            acc.append(buf, (size_t) n);
            const size_t at = acc.find("listening on 127.0.0.1:");
            if (at != std::string::npos) {
                port = (uint16_t) std::atoi(acc.c_str() + at + 23);
                break;
            }
        }
    }
    check(port != 0, "the worker printed its loopback port");

    strata::kernels::cpu::NativeFmt fmt;
    std::string nerr;
    check(strata::kernels::cpu::native_fmt(kGu, kD, kEmbd, kFf, fmt, nerr), "the layout resolves");

    // Q8_0 packed fixtures
    Rng fr(3);
    auto pack_rows = [&](int64_t rows, int64_t n, size_t row_bytes, uint8_t* dst) {
        std::vector<float> src((size_t) n);
        for (int64_t r = 0; r < rows; ++r) {
            fr.fill(src.data(), src.size());
            ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), dst + (size_t) r * row_bytes, 0, 1, n, nullptr);
        }
    };
    auto make_blob = [&]() {
        std::vector<uint8_t> blob(fmt.bytes, 0);
        pack_rows(fmt.n_ff, fmt.n_embd, fmt.gu_row, blob.data());
        pack_rows(fmt.n_ff, fmt.n_embd, fmt.gu_row, blob.data() + fmt.up_off);
        pack_rows(fmt.n_embd, fmt.n_ff, fmt.d_row, blob.data() + fmt.down_off);
        return blob;
    };
    std::vector<uint8_t> blob0 = make_blob(), blob5 = make_blob();

    strata::worker::SetupManifest m;
    m.nonce = 0xe2e2;
    m.model_id = 77;
    std::snprintf(m.model_name, sizeof m.model_name, "e2e-model");
    std::snprintf(m.quant_name, sizeof m.quant_name, "Q8_0");
    m.gu_type = kGu;
    m.d_type = kD;
    m.n_layers = kLayers;
    m.n_expert = kExperts;
    m.n_embd = kEmbd;
    m.n_ff = kFf;
    m.partition.assign(kLayers, 0);
    m.partition[0] = 0x21;   // layer 0 owns experts 0 and 5
    m.partition_id = strata::worker::blob_digest(m.partition.data(), m.partition.size()).a;

    strata::worker::ExpertWorkerClient cl;
    std::string err;
    check(cl.connect_host("127.0.0.1", port, 8000, err), "connect");
    check(cl.setup(m, err) && cl.session() != 0, "setup");
    check(cl.upload(0, 0, blob0.data(), (uint32_t) blob0.size(), err), "upload expert 0");
    check(cl.upload(0, 5, blob5.data(), (uint32_t) blob5.size(), err), "upload expert 5");
    check(cl.mark_ready(err), "ready");

    strata::worker::InferRequest r;
    r.session = cl.session();
    r.model_id = m.model_id;
    r.partition_id = m.partition_id;
    r.sequence = 1;
    r.layer = 0;
    r.T = 2;
    r.K = 3;
    r.mask = {1, 0, 1, 1, 0, 0};
    r.ids = {0, 3, 5, 5, 6, 7};
    r.weights = {0.55f, 0.2f, 0.25f, 0.7f, 0.2f, 0.1f};
    r.activations.resize((size_t) 2 * kEmbd);
    Rng ar(17);
    ar.fill(r.activations.data(), r.activations.size());

    strata::worker::InferReply got;
    check(cl.infer(r, got, err), "inference served");
    const uint8_t* blobs[6] = {blob0.data(), nullptr, blob5.data(), blob5.data(), nullptr, nullptr};
    std::vector<float> want;
    check(strata::worker::ExpertWorkerService::direct_native_partial(fmt, 2, 3, r.mask.data(),
                                                                     r.weights.data(),
                                                                     r.activations.data(), blobs, want,
                                                                     nerr),
          "the direct reference computes");
    check(got.partials.size() == want.size() &&
              std::memcmp(got.partials.data(), want.data(), want.size() * sizeof(float)) == 0,
          "the transported partial sums are bitwise the direct native call");

    strata::worker::WorkerStatusInfo st;
    check(cl.status(st, err) && st.ready == 1 && st.requests == 1 && st.setup_weight_bytes == 2 * fmt.bytes,
          "the counters saw the run");
    check(cl.shutdown(err), "shutdown asked");
    cl.close();

    int status = 0;
    const pid_t done = ::waitpid(pid, &status, 0);
    check(done == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0, "the worker exits cleanly");
    ::close(pipefd[0]);
    if (fails == 0) std::printf("expert_worker_e2e_test: all checks passed\n");
    else std::printf("expert_worker_e2e_test: %d checks failed\n", fails);
    return fails == 0 ? 0 : 1;
#endif
}
