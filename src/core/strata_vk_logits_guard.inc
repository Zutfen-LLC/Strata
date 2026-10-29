// Included ONLY at the end of the exact pinned ggml-vulkan.cpp build-tree copy.
#include "strata_vk_logits_guard.hpp"
#include "ggml-alloc.h"
#include "strata_logits_guard_spv.hpp"
#include <array>
#include <memory>
#include <stdexcept>

namespace strata::core::portable_detail {
namespace {
void logits_guard_need(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
constexpr size_t LOGITS_STATUS_WORDS = 16; // native kNonfiniteBlocks
constexpr size_t LOGITS_METADATA = 4096;
}
struct StrataVkLogitsGuard {
    ggml_backend_t backend;
    vk_pipeline pipeline = std::make_shared<vk_pipeline_struct>();
    std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor *status = nullptr;
    size_t bytes = 0;
    explicit StrataVkLogitsGuard(ggml_backend_t b) : backend(b) {}
    ~StrataVkLogitsGuard() {
        auto *ctx = static_cast<ggml_backend_vk_context *>(backend->context);
        // A failed check may have queued work. Destruction must not unwind.
        try { ggml_backend_vk_synchronize(backend); } catch (...) {}
        if (pipeline) {
            // Remove the weak registration before destroying these owned handles.
            std::lock_guard<std::mutex> lock(ctx->device->compile_mutex);
            auto &all = ctx->device->all_pipelines;
            all.erase(std::remove_if(all.begin(), all.end(), [&](const vk_pipeline_ref &p) {
                return p.lock() == pipeline;
            }), all.end());
            if (pipeline->pipeline) ctx->device->device.destroyPipeline(pipeline->pipeline);
            if (pipeline->layout) ctx->device->device.destroyPipelineLayout(pipeline->layout);
            if (pipeline->shader_module) ctx->device->device.destroyShaderModule(pipeline->shader_module);
        }
    }
};
StrataVkLogitsGuard *strata_vk_logits_guard_create(ggml_backend_t backend, size_t budget) {
    logits_guard_need(backend && ggml_backend_is_vk(backend), "Vulkan logits guard: Vulkan backend required");
    auto guard = std::make_unique<StrataVkLogitsGuard>(backend);
    auto *ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    const auto &limits = ctx->device->properties.limits;
    logits_guard_need(limits.maxComputeWorkGroupInvocations >= 256 && limits.maxComputeWorkGroupSize[0] >= 256 &&
                      limits.maxComputeSharedMemorySize >= 256 * sizeof(uint32_t),
                      "Vulkan logits guard: device cannot execute integer guard geometry");
    guard->metadata.reset(ggml_init({LOGITS_METADATA, nullptr, true}));
    logits_guard_need(bool(guard->metadata), "Vulkan logits guard: metadata allocation failed");
    guard->status = ggml_new_tensor_1d(guard->metadata.get(), GGML_TYPE_I32, LOGITS_STATUS_WORDS);
    const size_t bound = ggml_backend_alloc_ctx_tensors_from_buft_size(
        guard->metadata.get(), ggml_backend_get_default_buffer_type(backend));
    logits_guard_need(bound <= budget, "Vulkan logits guard: aligned status budget exceeded");
    guard->buffer.reset(ggml_backend_alloc_ctx_tensors(guard->metadata.get(), backend));
    logits_guard_need(bool(guard->buffer), "Vulkan logits guard: status allocation failed");
    guard->bytes = ggml_backend_buffer_get_size(guard->buffer.get());
    logits_guard_need(guard->bytes <= budget, "Vulkan logits guard: actual aligned status budget exceeded");
    auto &p = guard->pipeline;
    p->name = "strata_logits_nonfinite_bits";
    p->parameter_count = 2;
    p->push_constant_size = 3 * sizeof(uint32_t);
    p->wg_denoms = {1, 1, 1};
    p->align = 1;
    p->initialized = true;
    ggml_vk_create_pipeline_func(ctx->device, p, sizeof(strata_logits_guard_spv), strata_logits_guard_spv,
                                "main", 2, {1, 1, 1}, {}, false, false, 0);
    return guard.release();
}
bool strata_vk_logits_guard_check(StrataVkLogitsGuard *g, const ggml_tensor *logits) {
    logits_guard_need(g && logits && logits->type == GGML_TYPE_F32 && ggml_is_contiguous(logits) &&
                      ggml_nelements(logits) > 0 && ggml_nelements(logits) <= 1048576,
                      "Vulkan logits guard: contiguous bounded F32 logits required");
    auto *ctx = static_cast<ggml_backend_vk_context *>(g->backend->context);
    logits_guard_need(logits->buffer && ggml_backend_buffer_is_vk(logits->buffer),
                      "Vulkan logits guard: device logits required");
    auto *source = static_cast<ggml_backend_vk_buffer_context *>(logits->buffer->context);
    logits_guard_need(source->device.lock() == ctx->device, "Vulkan logits guard: logits belong to another device");
    const uint32_t input_bias = get_misalign_bytes(ctx, logits);
    const uint32_t status_bias = get_misalign_bytes(ctx, g->status);
    logits_guard_need(input_bias % sizeof(uint32_t) == 0 && status_bias % sizeof(uint32_t) == 0,
                      "Vulkan logits guard: word-aligned tensor offsets required");
    const std::array<uint32_t, 3> params{uint32_t(ggml_nelements(logits)),
        input_bias / uint32_t(sizeof(uint32_t)), status_bias / uint32_t(sizeof(uint32_t))};
    // Complete the real head first, including an asynchronous caller's work.
    ggml_backend_vk_synchronize(g->backend);
    std::array<uint32_t, LOGITS_STATUS_WORDS> words;
    words.fill(UINT32_MAX);
    // Fail closed if any output word is not overwritten by this invocation.
    ggml_backend_tensor_set(g->status, words.data(), 0, sizeof(words));
    ggml_pipeline_request_descriptor_sets(ctx, g->pipeline, 1);
    auto compute = ggml_vk_get_compute_ctx(ctx);
    ggml_vk_sync_buffers(ctx, compute);
    const auto input = ggml_vk_tensor_subbuffer(ctx, logits, true);
    const auto output = ggml_vk_tensor_subbuffer(ctx, g->status, true);
    ggml_vk_dispatch_pipeline(ctx, compute, g->pipeline, {input, output}, params, {16, 1, 1});
    ggml_vk_sync_buffers(ctx, compute);
    compute.reset();
    ggml_backend_vk_synchronize(g->backend); // submit, fence wait and descriptor/command cleanup
    ggml_backend_tensor_get(g->status, words.data(), 0, sizeof(words));
    return std::all_of(words.begin(), words.end(), [](uint32_t word) { return word == 0; });
}
void strata_vk_logits_guard_destroy(StrataVkLogitsGuard *g) noexcept { delete g; }
size_t strata_vk_logits_guard_device_bytes(const StrataVkLogitsGuard *g) { return g->bytes; }
size_t strata_vk_logits_guard_host_bytes(const StrataVkLogitsGuard *) {
    return LOGITS_METADATA + sizeof(StrataVkLogitsGuard) + 2 * LOGITS_STATUS_WORDS * sizeof(uint32_t);
}
} // namespace strata::core::portable_detail
