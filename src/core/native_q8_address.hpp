#pragma once
#include "native_q8_gu_layout.hpp"
#include <algorithm>
#include <limits>
namespace strata::core::portable_detail {
inline void q8_require_address_features(bool bda,bool int64) {
    q8_gu_need(bda,"native Q8: logical bufferDeviceAddress is required");
    q8_gu_need(int64,"native Q8: logical shaderInt64 is required");
}
inline uint64_t q8_address_add(uint64_t a,uint64_t b) {
    q8_gu_need(b<=UINT64_MAX-a,"native Q8: address overflow");return a+b;
}
inline uint64_t q8_weight_address(uint64_t base,uint64_t owner_bytes,uint64_t tensor_offset,
    uint64_t view_offset,uint64_t tensor_bytes,uint64_t offset,uint64_t required) {
    q8_gu_need(base && owner_bytes && required,"native Q8: unavailable weight owner");
    const auto start=q8_address_add(tensor_offset,view_offset);
    q8_gu_need(start<=owner_bytes && tensor_bytes<=owner_bytes-start && offset<=tensor_bytes &&
        required<=tensor_bytes-offset,"native Q8: weight reference outside owner");
    const auto address=q8_address_add(q8_address_add(base,start),offset);
    q8_gu_need(address%4==0,"native Q8: unaligned weight reference");
    q8_address_add(address,required-1);return address;
}
inline size_t q8_resident_limit(size_t policy,uint64_t buffer,uint64_t allocation) {
    const auto limit=std::min(uint64_t(policy),std::min(buffer,allocation));
    q8_gu_need(limit>0,"native Q8: unavailable backend block limit");
    return size_t(limit);
}
}
