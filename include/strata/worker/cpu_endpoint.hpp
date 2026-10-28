#pragma once
#include "strata/worker/gpu_endpoint.hpp" // shared numeric syntax only; no GPU feature/link dependency
namespace strata::worker {
inline constexpr size_t kMaxCpuWorkers = 2;
struct CpuEndpoint { std::string host; uint16_t port = 0; };
inline bool parse_cpu_endpoint(const std::string& text, CpuEndpoint& out) {
    GpuEndpoint parsed;
    if (!parse_gpu_endpoint(text, parsed)) return false;
    out = {parsed.host, parsed.port}; return true;
}
}
