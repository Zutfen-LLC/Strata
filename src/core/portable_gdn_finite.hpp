#pragma once
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::core::portable_detail {
// GDN's valid-buffer fast path. Integer classification permits a vectorized
// reduction without fast-math or floating-point operations on signaling NaNs.
// The caller retains its original scalar first-failure path.
inline bool gdn_all_finite(const std::vector<float> &x) {
    static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::radix == 2 && std::numeric_limits<float>::digits == 24 &&
                  std::numeric_limits<float>::max_exponent == 128);
    uint32_t bad = 0;
    for (float value : x)
        bad |= (std::bit_cast<uint32_t>(value) & 0x7f800000u) == 0x7f800000u;
    return bad == 0;
}
} // namespace strata::core::portable_detail
