// SGW1 GPU contract: default CUDA or explicit Vulkan hybrid. SEW1 is unchanged.
#pragma once
#include "strata/worker/gpu_assets.hpp"
#include <functional>

namespace strata::worker {
inline constexpr uint32_t kGpuWireMagic = 0x31574753u; // SGW1
inline constexpr uint32_t kGpuWireVersion = 1;
inline constexpr uint32_t kGpuCapabilityCuda = 1;
inline constexpr uint32_t kGpuCapabilityEntryRows = 2;
inline constexpr uint32_t kGpuCapabilityVulkan = 4; // explicit opt-in, entry rows only
// NativeExpertLayout Q8_0 weights, q8_1 activations, FP32 SwiGLU/output. Approximate,
// never the CPU worker's native-bitwise numerical contract.
inline constexpr uint32_t kGpuNumericsQ8_1 = 1;
// Q8_0 weights/matvecs on Vulkan, native host Q8 boundaries + SwiGLU; not CPU bit parity.
inline constexpr uint32_t kGpuNumericsVulkanHybridQ8 = 2;
inline constexpr size_t kGpuFrameMax = kGpuPlanMax + 64;
enum GpuMsg : uint16_t {
    GPU_HELLO = 1, GPU_HELLO_ACK, GPU_SETUP, GPU_SETUP_ACK, GPU_UPLOAD, GPU_UPLOAD_ACK,
    GPU_READY, GPU_READY_ACK, GPU_INFER, GPU_INFER_ACK, GPU_STATUS, GPU_STATUS_ACK,
    GPU_STOP, GPU_STOP_ACK, GPU_INFER_ROWS, GPU_INFER_ROWS_ACK
};
struct GpuCounters {
    uint64_t assigned_bytes = 0, cached_bytes = 0, transferred_bytes = 0;
    uint64_t persisted_bytes = 0, retained_disk_bytes = 0, verified_bytes = 0, reused_bytes = 0;
    uint64_t host_staging_peak = 0, gpu_allocated_bytes = 0, gpu_peak_bytes = 0;
    uint64_t gpu_weight_bytes = 0, gpu_scratch_bytes = 0, gpu_allocation_count = 0;
    uint64_t gpu_free_before = 0, gpu_free_after = 0, gpu_runtime_reserve = 0;
    uint64_t requests = 0, entries = 0, failures = 0, service_ns = 0;
};
namespace gpu_codec {
void put(std::vector<uint8_t>& p, uint64_t v, size_t n = 8);
struct Cursor {
    const std::vector<uint8_t>& p;
    size_t i = 0;
    bool ok = true;
    uint64_t get(size_t n = 8);
    bool done() const { return ok && i == p.size(); }
};
bool send(uintptr_t fd, uint16_t msg, uint16_t status, const std::vector<uint8_t>& p,
          int ms, const std::function<bool()>& cancel, std::string& err);
bool receive(uintptr_t fd, WireFrame& f, size_t cap, int ms,
             const std::function<bool()>& cancel, std::string& err, int response_wait_ms = 0);
std::vector<uint8_t> binding(uint64_t session, const GpuPlan& plan);
bool binding(Cursor& c, uint64_t session, const GpuPlan& plan);
bool encode_request(const GpuPlan& plan, const InferRequest& r, std::vector<uint8_t>& p);
bool decode_request(const GpuPlan& plan, const std::vector<uint8_t>& p, InferRequest& r);
std::vector<uint8_t> encode_reply(const GpuPlan& plan, Digest128 request, const InferReply& r);
bool decode_reply(const GpuPlan& plan, Digest128 request, const InferRequest& r,
                  const std::vector<uint8_t>& p, InferReply& out);
// Rows retain original frame-entry indices and raw unweighted F32 values.
bool encode_rows_reply(const GpuPlan&, Digest128, const InferRequest&, const EntryRowsReply&, std::vector<uint8_t>&);
bool decode_rows_reply(const GpuPlan&, Digest128, const InferRequest&, const std::vector<uint8_t>&, EntryRowsReply&);
std::vector<uint8_t> counters(const GpuCounters& c);
bool counters(const std::vector<uint8_t>& p, GpuCounters& c);
} // namespace gpu_codec
} // namespace strata::worker
