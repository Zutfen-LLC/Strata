#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
namespace strata::core::portable_detail {
inline void native_router_shape(uint32_t experts,uint32_t k,uint32_t tokens) {
    if(experts!=512 || k!=10 || tokens<1 || tokens>8)
        throw std::runtime_error("native router: only 512 experts, top 10, T1..8 supported");
}
inline std::array<uint32_t,4> native_router_parameters(uint32_t tokens,const std::array<uint32_t,3>& bias) {
    native_router_shape(512,10,tokens);
    for(auto x:bias)if(x%4)throw std::runtime_error("native router: four-byte alignment required");
    return {tokens,bias[0]/4,bias[1]/4,bias[2]/4};
}
}
