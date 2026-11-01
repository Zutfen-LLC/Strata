#pragma once
#include "native_hc_layout.hpp"
namespace strata::core::portable_detail {
constexpr size_t native_gdn_log1p_count=256;
// exp(v), v<=20, is below 2^29 even with the declared exp error allowance.
// Zero and finite nonnegative normal inputs only; negative zero preserves sign.
inline void native_gdn_log1p_input(float x){const auto b=native_hc_bits(x);native_hc_normal(x);native_hc_need((b&0x7fffffffu)==0||(!(b>>31)&&b<=0x4e000000u),"GDN log1p: require normal/zero x in [0,2^29]");}
}
