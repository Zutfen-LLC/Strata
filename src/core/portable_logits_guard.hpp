#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <memory>

namespace strata::core::portable_detail {
// Owns a bounded integer status guard; borrows one serial backend. CPU is the
// exact-bit reference. Vulkan uses only the pin-checked local backend bridge.
class PortableLogitsGuard {
  public:
    explicit PortableLogitsGuard(ggml_backend_t backend, size_t device_budget);
    ~PortableLogitsGuard();
    PortableLogitsGuard(const PortableLogitsGuard &) = delete;
    PortableLogitsGuard &operator=(const PortableLogitsGuard &) = delete;
    bool check(const ggml_tensor *logits);
    size_t device_bytes() const; // actual aligned status allocation
    size_t host_bound() const;
    size_t device_resource_bound() const; // conservative driver/pipeline allowance
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core::portable_detail
