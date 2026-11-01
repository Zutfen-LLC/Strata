#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
namespace strata_llvm_adapter {
using uint=uint32_t;
struct uvec2 {uint x,y;constexpr uvec2(uint a,uint b):x(a),y(b){}};
inline double packDouble2x32(uvec2 b){uint64_t u=(uint64_t(b.y)<<32)|b.x;double x;std::memcpy(&x,&u,8);return x;}
inline uvec2 unpackDouble2x32(double x){uint64_t u;std::memcpy(&u,&x,8);return {uint(u),uint(u>>32)};}
inline uint floatBitsToUint(float x){uint b;std::memcpy(&b,&x,4);return b;}
inline float uintBitsToFloat(uint b){float x;std::memcpy(&x,&b,4);return x;}
#define precise
#include "strata_llvm_log1p_adapter_body.hpp"
#undef precise
}
