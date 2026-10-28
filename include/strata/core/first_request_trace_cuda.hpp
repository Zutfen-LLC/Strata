// Fixed seven-tensor snapshot layout shared by the verifier and graph-alignment test.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace strata::core {
struct FirstTraceShape {
    size_t tokens, embd, experts, topk, channels;
    size_t width(int field) const {
        const size_t widths[] = {embd, experts, topk, topk, embd, channels*embd, channels};
        return widths[field];
    }
    size_t stride() const {
        size_t sum = 0;
        for (int f=0; f<7; ++f) sum += tokens*width(f);
        return sum;
    }
    size_t offset(size_t layer, int field, size_t tb) const {
        size_t sum = layer*stride() + tb*width(field);
        for (int f=0; f<field; ++f) sum += tokens*width(f);
        return sum;
    }
};
inline bool first_trace_snapshot(const FirstTraceShape& shape, float* storage, size_t layer,
                                  int field, size_t tb, size_t rows, const void* src, cudaStream_t stream) {
    if (!storage) return true;
    return cudaMemcpyAsync(storage + shape.offset(layer, field, tb), src, rows*shape.width(field)*4,
                           cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
}
} // namespace strata::core
