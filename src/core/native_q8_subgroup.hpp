#pragma once
#include <cstdint>
namespace strata::core::portable_detail {
// Full compute subgroups are mandatory: subgroup-derived IDs then densely cover
// the 256 invocation workgroup even when local invocation IDs are unrelated.
inline bool native_q8_subgroup_supported(bool basic,bool shuffle,bool full,uint32_t size) {
    return basic && shuffle && full && (size==32 || size==64);
}
}
