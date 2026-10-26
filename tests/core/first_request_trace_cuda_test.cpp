// Model-free: real CUDA graph capture/replay, shared production snapshot function.
#include "strata/core/first_request_trace_cuda.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace strata::core;
static void check(bool ok) { if (!ok) { std::fprintf(stderr, "snapshot graph alignment failed\n"); std::exit(1); } }
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    FirstTraceShape shape{4, 8, 16, 2, 4};
    float *storage=nullptr, *scratch=nullptr;
    check(cudaMalloc((void**)&storage, 2*shape.stride()*4) == cudaSuccess);
    check(cudaMalloc((void**)&scratch, 4*shape.width(5)*4) == cudaSuccess);
    cudaStream_t stream;
    check(cudaStreamCreate(&stream) == cudaSuccess);
    // Disabled: no runtime call even with invalid stream/source.
    check(first_trace_snapshot(shape, nullptr, 0, 0, 0, 1, nullptr, stream));
    for (size_t T : {1u, 3u, 4u}) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
        for (size_t layer=0; layer<2; ++layer) for (int field=0; field<7; ++field) {
            for (size_t tb=0; tb<T; ++tb) {
                // Fill the scratch with a unique byte pattern, snapshot, then overwrite it.
                int marker = 1 + (int)layer*40 + field*5 + (int)tb;
                check(cudaMemsetAsync(scratch, marker, shape.width(field)*4, stream) == cudaSuccess);
                check(first_trace_snapshot(shape, storage, layer, field, tb, 1, scratch, stream));
                check(cudaMemsetAsync(scratch, 0, shape.width(field)*4, stream) == cudaSuccess);
            }
        }
        cudaGraph_t graph;
        check(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
        cudaGraphExec_t exec;
        check(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
        for (int replay=0; replay<2; ++replay) {
            check(cudaMemset(storage, 0, 2*shape.stride()*4) == cudaSuccess);
            check(cudaGraphLaunch(exec, stream) == cudaSuccess);
            check(cudaStreamSynchronize(stream) == cudaSuccess);
            std::vector<unsigned char> host(2*shape.stride()*4);
            check(cudaMemcpy(host.data(), storage, host.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (size_t layer=0; layer<2; ++layer) for (int field=0; field<7; ++field)
                for (size_t tb=0; tb<shape.tokens; ++tb) for (size_t i=0; i<shape.width(field)*4; ++i) {
                    int marker = tb<T ? 1+(int)layer*40+field*5+(int)tb : 0;
                    check(host[shape.offset(layer, field, tb)*4+i] == marker);
                }
        }
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream); cudaFree(scratch); cudaFree(storage);
    std::puts("first-request snapshot graph alignment passed (T=1,3,4; two replays)");
}
