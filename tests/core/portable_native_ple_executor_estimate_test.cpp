#include "portable_native_ple_executor.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <memory>
#include <stdexcept>

using namespace strata::core::portable_detail;
static_assert(PortableNativePleConv::host_bound(2) == 16384 + 40960*2 + (5*92160 + 20*10240*2)*sizeof(float));
static_assert(PortableNativePleConvCopyGraph::host_bound(2) == 1024*1024 + PortableNativePleConv::host_bound(2));
static_assert(PortableNativePleConvCopyGraph::estimate_host_bound(1) + 16384 == PortableNativePleConvCopyGraph::host_bound(1));
static_assert(PortableNativePleConvCopyGraph::estimate_host_bound(2) + 16384 == PortableNativePleConvCopyGraph::host_bound(2));
static_assert(PortableNativePleKeyBatch::host_bound(2) == 16384 + 65536);
static_assert(PortableNativePleGate::host_bound(2) == 16384 + 2*(3*40960+16));
static_assert(PortableNativePleGroupedNorm::host_bound(1) == 16384 + 2*40960);
static_assert(PortableNativePleCopyGraph::host_bound(1) == 1024*1024 + PortableNativePleGroupedNorm::host_bound(1));
static_assert(PortableNativePleGatedValue::host_bound(2) == 1024*1024);
static_assert(PortableNativePleValue::host_bound(2) == 32768 + 2560*2560*2 + 65536 + 2560*4 + 2*4*2560*4);

int main() {
    try {
        bool overflow = false;
        try { (void)PortableNativePleConvCopyGraph::host_bound(SIZE_MAX); }
        catch (const std::overflow_error&) { overflow = true; }
        if (!overflow) throw std::runtime_error("conv host overflow accepted");
        overflow = false;
        try { (void)PortableNativePleConvCopyGraph::estimate_host_bound(SIZE_MAX); }
        catch (const std::overflow_error&) { overflow = true; }
        if (!overflow) throw std::runtime_error("conv mixed estimate host overflow accepted");
        overflow = false;
        try { (void)PortableNativePleGate::host_bound(SIZE_MAX); }
        catch (const std::overflow_error&) { overflow = true; }
        if (!overflow) throw std::runtime_error("host bound overflow accepted");
        bool rejected = false;
        try { (void)PortableNativePleExecutor::estimate(nullptr); }
        catch (const std::exception&) { rejected = true; }
        if (!rejected) throw std::runtime_error("null backend accepted");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        if (!cpu) throw std::runtime_error("CPU backend unavailable");
        const auto breakdown = PortableNativePleExecutor::estimate_breakdown(cpu.get());
        if (breakdown.gather_host != 3*2560*sizeof(float) + 2*sizeof(std::array<uint32_t,16>))
            throw std::runtime_error("executor gather omits simultaneous decoded row");
        const auto first = PortableNativePleExecutor::estimate(cpu.get());
        const auto second = PortableNativePleExecutor::estimate(cpu.get());
        if (first != second || first < 2ull * 2560 * 2560 * sizeof(uint16_t))
            throw std::runtime_error("executor estimate is unstable or omits retained weights");
        std::puts("PLE executor CPU estimate passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
