#pragma once
#include "portable_native_ple_host_bound.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::core::portable_detail {
// Arena requested by both component-owner and quant-stage ggml_init calls.
// This is an explicit source payload, not a general factory overhead allowance.
inline constexpr size_t native_component_metadata_arena_bytes = 4096;
// Payload only: excludes allocator/control-block overhead, driver and pipeline
// helper internals, borrowed input and any device allocation. The strict vector
// (input + 12 mode + 6 capability words) and its observation copy coexist;
// erasing the four-word mode and two-word capability does not release capacity.
inline size_t native_component_rewrite_host_bound(size_t words) {
    return ple_host_mul(ple_host_mul(ple_host_add(words, 18), sizeof(uint32_t)), 2);
}
template<class Factory>
auto native_component_admit(size_t bound, size_t budget, Factory&& factory) -> decltype(factory()) {
    if (bound > budget) throw std::length_error("native component host budget exceeded");
    return factory();
}
enum class NativeComponentHostKind { hc, hc_pending, hc_write, hc_mix, hc_head_mix, ple_conv, ple_gate, ple_norm, ple_key_quant, ple_key_mmvq, ple_value };
// Defined in the pinned Vulkan translation unit, using its embedded modules and
// concrete owner types. path_bytes is the receipt path length excluding NUL.
size_t native_component_factory_host_bound(NativeComponentHostKind kind, size_t path_bytes);
}
