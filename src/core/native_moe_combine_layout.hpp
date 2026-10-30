#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
namespace strata::core::portable_detail {
// This component implements only the selected plain-Q8 model's K10 combine.
// Parts are in original router order [token][10][2560], not grouped order.
inline void native_moe_combine_shape(uint32_t h,uint32_t tokens) {
    if(h!=2560 || tokens<1 || tokens>8)
        throw std::runtime_error("native MoE combine: H2560, K10 and T1..8 required");
}
inline std::array<uint32_t,6> native_moe_combine_parameters(uint32_t h,uint32_t tokens,
    const std::array<uint32_t,4>& byte_bias) {
    native_moe_combine_shape(h,tokens);
    if(byte_bias[0]%16 || byte_bias[1]%4 || byte_bias[2]%16 || byte_bias[3]%16)
        throw std::runtime_error("native MoE combine: aligned vec4 parts/shared/output and F32 weights required");
    return {h,tokens,byte_bias[0]/16,byte_bias[1]/4,byte_bias[2]/16,byte_bias[3]/16};
}
}
