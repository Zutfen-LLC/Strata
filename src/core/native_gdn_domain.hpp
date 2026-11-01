#pragma once
#include "native_hc_layout.hpp"
#include "native_hc_mix_limits.h"
namespace strata::core::portable_detail {
// Native gdn_step_norm_multi evaluates __expf(gate) without a minimum-log clamp.
// Input admission is separate from actual primitive/intermediate admission.
inline void native_gdn_decay_input(float x){native_hc_normal(x);native_hc_need(x<=0.f,"native GDN: nonpositive decay log required");}
inline void native_gdn_sigmoid_input(float x){native_hc_normal(x);const auto b=native_hc_bits(x)&0x7fffffffu;
 native_hc_need(b<=STRATA_HC_MIX_ACTIVATION_LIMIT_BITS||x>0.f||x<=-89.f,"native GDN: sigmoid transition/subnormal quotient not qualified");}
}
