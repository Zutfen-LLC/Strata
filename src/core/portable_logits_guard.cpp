#include "portable_logits_guard.hpp"
#include "ggml-cpu.h"
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
#include "ggml-vulkan.h"
#include "strata_vk_logits_guard.hpp"
#endif
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::core::portable_detail {
namespace {
void need(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
// Driver pipeline/descriptor allocations have no portable byte query. Keep a
// separate conservative allowance rather than presenting 64 status bytes as
// the whole resource cost. Existing backend context allowances also remain.
constexpr size_t RESOURCE_BOUND = 1u << 20;
}
struct PortableLogitsGuard::Impl {
    ggml_backend_t backend;
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
    std::unique_ptr<StrataVkLogitsGuard, decltype(&strata_vk_logits_guard_destroy)> vk{nullptr, strata_vk_logits_guard_destroy};
#endif
    explicit Impl(ggml_backend_t b, size_t budget) : backend(b) {
        (void)budget; // CPU reference has no device allocation.
        need(b != nullptr, "portable logits guard: backend required");
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
        if (ggml_backend_is_vk(b)) {
            need(budget >= RESOURCE_BOUND, "portable logits guard: resource budget exceeded");
            vk.reset(strata_vk_logits_guard_create(b, budget - RESOURCE_BOUND));
            return;
        }
#endif
        need(ggml_backend_is_cpu(b), "portable logits guard: unsupported backend; integer guard required");
    }
};
PortableLogitsGuard::PortableLogitsGuard(ggml_backend_t b, size_t budget) : impl_(std::make_unique<Impl>(b, budget)) {}
PortableLogitsGuard::~PortableLogitsGuard() = default;
bool PortableLogitsGuard::check(const ggml_tensor *t) {
    need(t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) > 0 &&
         ggml_nelements(t) <= 1048576, "portable logits guard: contiguous bounded F32 logits required");
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
    if (impl_->vk) return strata_vk_logits_guard_check(impl_->vk.get(), t);
#endif
    static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
    need(t->buffer && ggml_backend_buffer_is_host(t->buffer) && t->data,
         "portable logits guard: CPU reference requires host tensor");
    ggml_backend_synchronize(impl_->backend);
    uint32_t bad = 0;
    const auto *bytes = static_cast<const unsigned char *>(t->data);
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        uint32_t bits;
        std::memcpy(&bits, bytes + i * sizeof(bits), sizeof(bits));
        bad |= (bits & 0x7f800000u) == 0x7f800000u;
    }
    return bad == 0;
}
size_t PortableLogitsGuard::device_bytes() const {
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
    if (impl_->vk) return strata_vk_logits_guard_device_bytes(impl_->vk.get());
#endif
    return 0;
}
size_t PortableLogitsGuard::host_bound() const {
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
    if (impl_->vk) return sizeof(Impl) + strata_vk_logits_guard_host_bytes(impl_->vk.get()) + RESOURCE_BOUND;
#endif
    return sizeof(Impl);
}
size_t PortableLogitsGuard::device_resource_bound() const {
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
    if (impl_->vk) return RESOURCE_BOUND;
#endif
    return 0;
}
} // namespace strata::core::portable_detail
