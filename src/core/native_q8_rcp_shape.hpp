#pragma once
#include <cstdint>
#include <stdexcept>
namespace strata::core::portable_detail {
inline void native_q8_rcp_require_mode(uint32_t mode) {
    if(mode!=0)throw std::runtime_error("native reciprocal observation: mode 0 only");
}
}
