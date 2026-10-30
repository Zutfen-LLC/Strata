#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace strata::core::portable_detail {
inline constexpr std::array<uint32_t,4> NATIVE_Q8_TRACE_ELEMENTS{36,118,1046,4479};
inline constexpr size_t NATIVE_Q8_TRACE_WORDS=16;
inline void native_q8_quant_trace_shape(uint32_t mode,int64_t elements,uint32_t n_ff,
    const int32_t* starts,size_t count) {
    if(mode!=0 || elements!=7680 || n_ff!=2560 || !starts || count!=2 || starts[0]!=0 || starts[1]!=3)
        throw std::runtime_error("Q8 quant trace: only the complete retained input fixture/mode0 is supported");
}
}
