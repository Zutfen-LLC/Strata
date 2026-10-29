#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::core::portable_detail {
inline size_t device_mib(int64_t value) {
    if (value <= 0 || uint64_t(value) > (SIZE_MAX >> 20))
        throw std::runtime_error("invalid or overflowing device MiB budget");
    return size_t(value) << 20;
}
// Policy ceiling P is independent of unrelated baseline usage. Reserve R stays
// outside the conservative model ledger L: L <= min(P, A-R), never P+R <= A.
inline size_t device_admission_limit(size_t policy, size_t available, size_t total, size_t reserve) {
    if (!policy || !total || available > total || policy > total || !reserve || available <= reserve)
        throw std::runtime_error("invalid device capacity/free evidence or reserve");
    return std::min(policy, available - reserve);
}
} // namespace strata::core::portable_detail
