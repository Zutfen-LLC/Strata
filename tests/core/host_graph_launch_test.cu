#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <mutex>

namespace {
bool blocking_launch_fixture = false;
cudaError_t fixture_graph_launch(cudaGraphExec_t graph, cudaStream_t stream) {
    const auto error = cudaGraphLaunch(graph, stream);
    // Test only: an instrumented launch may not return until the real graph
    // finishes. Force that ordering without depending on a driver/tool version.
    return error != cudaSuccess || !blocking_launch_fixture ? error : cudaStreamSynchronize(stream);
}
}
#define cudaGraphLaunch fixture_graph_launch
#include "strata/core/host_graph_launch.hpp"
#undef cudaGraphLaunch

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
static void gpu(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }
__global__ void wait_for_cpu(volatile unsigned* flag) {
    while (*flag == 0) { }
}
__global__ void tick(unsigned* value) { ++*value; }

namespace {
constexpr int trace_H = 2560, trace_K = 10, trace_layers = 4;
constexpr int trace_stride = (trace_K + 3) * trace_H + trace_K;
struct Handshake { unsigned seq, plan, copies, served; int mask[trace_K]; };
__global__ void traced_wait(volatile Handshake* words, unsigned layer) {
    words->seq = layer;
    __threadfence_system();
    while (words->plan < layer || words->copies < layer || words->served < layer) { }
}
__global__ void traced_rows(float* parts, float* shared, float* combined, int layer) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < trace_K * trace_H) parts[i] = layer + (i % 17) * .0625f;
    if (i < trace_H) { shared[i] = .5f; combined[i] = layer + (i % 13) * .125f; }
}
__global__ void traced_add(float* combined, int layer) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < trace_H) combined[i] += layer * .25f;
}

void layer_trace_case(bool threaded) {
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    Handshake *host, *mapped;
    float *device, *trace;
    gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    gpu(cudaHostAlloc(&host, sizeof(Handshake), cudaHostAllocPortable | cudaHostAllocMapped));
    gpu(cudaHostGetDevicePointer(&mapped, host, 0));
    gpu(cudaHostAlloc(&trace, trace_layers * trace_stride * sizeof(float), cudaHostAllocPortable));
    gpu(cudaMalloc(&device, (trace_K + 2) * trace_H * sizeof(float)));
    std::memset(host, 0, sizeof(*host));
    std::memset(trace, 0, trace_layers * trace_stride * sizeof(float));
    float* shared = device + trace_K * trace_H;
    float* combined = shared + trace_H;
    gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int layer = 1; layer <= trace_layers; ++layer) {
        float* row = trace + (layer - 1) * trace_stride;
        traced_wait<<<1, 1, 0, stream>>>(mapped, layer);
        traced_rows<<<(trace_K * trace_H + 255) / 256, 256, 0, stream>>>(device, shared, combined, layer);
        // Same copy directions, pinned destinations, mapped-mask alias and
        // ordering as Verifier::record_window's five first-layer trace fields.
        gpu(cudaMemcpyAsync(row, device, trace_K * trace_H * 4, cudaMemcpyDeviceToHost, stream));
        gpu(cudaMemcpyAsync(row + trace_K * trace_H, shared, trace_H * 4, cudaMemcpyDeviceToHost, stream));
        gpu(cudaMemcpyAsync(row + (trace_K + 1) * trace_H, combined, trace_H * 4, cudaMemcpyDeviceToHost, stream));
        traced_add<<<(trace_H + 255) / 256, 256, 0, stream>>>(combined, layer);
        gpu(cudaMemcpyAsync(row + (trace_K + 2) * trace_H, combined, trace_H * 4, cudaMemcpyDeviceToHost, stream));
        gpu(cudaMemcpyAsync(row + (trace_K + 3) * trace_H, mapped->mask, trace_K * 4, cudaMemcpyDeviceToHost, stream));
    }
    gpu(cudaStreamEndCapture(stream, &graph));
    gpu(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::atomic<bool> done{false}, timed_out{false};
    std::mutex host_updates;
    auto release = [&] {
        const std::lock_guard<std::mutex> lock(host_updates);
        auto* words = (volatile Handshake*)host;
        words->plan = words->copies = words->served = ~0u;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    };
    std::thread watchdog([&] {
        const auto start = std::chrono::steady_clock::now();
        while (!done && std::chrono::steady_clock::now() - start < std::chrono::seconds(8))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!done) { timed_out = true; release(); }
    });
    int served = 0;
    {
        strata::core::HostGraphLaunch launch(exec, stream, 0, threaded, release);
        gpu(launch.status());
        if (!timed_out) check(launch.query() == cudaErrorNotReady, "traced graph waits for CPU service");
        for (int layer = 1; layer <= trace_layers && !timed_out; ++layer) {
            while (((volatile Handshake*)host)->seq < (unsigned)layer && !timed_out) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                if (!timed_out) {
                    const auto query = launch.query();
                    check(query == cudaSuccess || query == cudaErrorNotReady, "traced launch/query error propagation");
                }
            }
            if (timed_out) break;
            const std::lock_guard<std::mutex> lock(host_updates);
            if (timed_out) break;
            for (int i = 0; i < trace_K; ++i) host->mask[i] = (i + layer) % 2;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            auto* words = (volatile Handshake*)host;
            words->plan = words->copies = words->served = layer;
            ++served;
        }
        gpu(launch.finish());
        gpu(cudaStreamSynchronize(stream));
    }
    done = true; watchdog.join();
    check(!timed_out && served == trace_layers, "traced graph CPU service before launch completion");
    for (int layer = 1; layer <= trace_layers; ++layer) {
        const float* row = trace + (layer - 1) * trace_stride;
        for (int i = 0; i < trace_K * trace_H; ++i)
            check(row[i] == layer + (i % 17) * .0625f, "actual per-entry trace rows");
        for (int i = 0; i < trace_H; ++i) {
            check(row[trace_K * trace_H + i] == .5f, "shared trace");
            check(row[(trace_K + 1) * trace_H + i] == layer + (i % 13) * .125f, "pre-add trace");
            check(row[(trace_K + 2) * trace_H + i] == layer + (i % 13) * .125f + layer * .25f, "post-add trace");
        }
        const auto* mask = (const int*)(row + (trace_K + 3) * trace_H);
        for (int i = 0; i < trace_K; ++i) check(mask[i] == (i + layer) % 2, "mapped ownership mask after CPU service");
    }
    gpu(cudaGraphExecDestroy(exec)); gpu(cudaGraphDestroy(graph));
    gpu(cudaFree(device)); gpu(cudaFreeHost(trace)); gpu(cudaFreeHost(host)); gpu(cudaStreamDestroy(stream));
    std::printf("FIRST_LAYER_CUDA_HANDSHAKE_V2 layers=%d served=%d trace_fields=5 blocking_launch=%d models=0 PASS\n",
                trace_layers, served, blocking_launch_fixture);
}
}

int main(int argc, char** argv) {
    bool threaded = true, layer_trace = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--synchronous") == 0) threaded = false;
        if (std::strcmp(argv[i], "--layer-trace") == 0) layer_trace = true;
        if (std::strcmp(argv[i], "--blocking-launch") == 0) blocking_launch_fixture = true;
    }
    if (layer_trace) { layer_trace_case(threaded); return 0; }
    cudaStream_t stream;
    gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned *host, *flag, *value;
    gpu(cudaHostAlloc(&host, sizeof(unsigned), cudaHostAllocMapped));
    gpu(cudaHostGetDevicePointer(&flag, host, 0));
    gpu(cudaMalloc(&value, sizeof(unsigned)));
    gpu(cudaMemset(value, 0, sizeof(unsigned)));
    *host = 0;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    wait_for_cpu<<<1, 1, 0, stream>>>(flag);
    // Enough nodes to exercise launch backpressure under Compute Sanitizer.
    for (int i = 0; i < 4096; ++i) tick<<<1, 1, 0, stream>>>(value);
    gpu(cudaStreamEndCapture(stream, &graph));
    gpu(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::atomic<bool> done{false}, timed_out{false};
    std::thread watchdog([&] {
        const auto start = std::chrono::steady_clock::now();
        while (!done && std::chrono::steady_clock::now() - start < std::chrono::seconds(8))
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!done) { timed_out = true; *(volatile unsigned*)host = 1; }
    });
    {
        strata::core::HostGraphLaunch launch(exec, stream, 0, threaded,
                                            [&] { *(volatile unsigned*)host = 1; });
        gpu(launch.status());
        check(launch.query() == cudaErrorNotReady, "graph waits for CPU service");
        *(volatile unsigned*)host = 1;
        gpu(launch.finish());
        gpu(cudaStreamSynchronize(stream));
    }
    done = true;
    watchdog.join();
    check(!timed_out, "CPU service proceeds without launch deadlock");
    unsigned count = 0;
    gpu(cudaMemcpy(&count, value, sizeof(unsigned), cudaMemcpyDeviceToHost));
    check(count == 4096, "all graph nodes executed");
    if (threaded) {
        // An early service failure must release the graph before joining.
        *host = 0;
        std::atomic<bool> released{false};
        {
            strata::core::HostGraphLaunch launch(exec, stream, 0, true,
                [&] { released = true; *(volatile unsigned*)host = 1; });
        }
        check(released, "early return releases device waits");
        gpu(cudaStreamSynchronize(stream));
        gpu(cudaMemcpy(&count, value, sizeof(unsigned), cudaMemcpyDeviceToHost));
        check(count == 8192, "early return drains the actual graph");
        if (argc > 1 && std::strcmp(argv[1], "--errors") == 0) {
            strata::core::HostGraphLaunch invalid(exec, stream, -1, true, [] {});
            check(invalid.finish() == cudaErrorInvalidDevice, "launch-thread device error propagated");
            (void)cudaGetLastError();
        }
    }
    gpu(cudaGraphExecDestroy(exec));
    gpu(cudaGraphDestroy(graph));
    gpu(cudaFree(value));
    gpu(cudaFreeHost(host));
    gpu(cudaStreamDestroy(stream));
    std::puts("PASS: graph launch, CPU doorbell service and cleanup");
}
