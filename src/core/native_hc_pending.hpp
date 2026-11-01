#pragma once
#include "native_hc_layout.hpp"
#include "native_hc_mix_limits.h"
namespace strata::core::portable_detail {
inline float native_hc_pending_fma(float b,float g,float r){return native_hc_normal(std::fma(b,g,r));}
struct NativeHcPendingControl {std::vector<float> residual,gate,exponential;};
// Standalone T1 control only. Never changes the original residual/pending tuple.
inline NativeHcPendingControl native_hc_pending_control(const std::vector<float>& r,
 const std::vector<float>& bo,const std::vector<float>& inject){
 native_hc_need(r.size()==10240&&bo.size()==2560&&inject.size()==4,"HC pending: T1 spans required");
 NativeHcPendingControl out;out.residual.resize(10240);out.gate.resize(4);out.exponential.resize(4);
 for(float v:r)native_hc_normal(v);for(float v:bo)native_hc_normal(v);
 for(size_t c=0;c<4;++c){native_hc_normal(inject[c]);const float scaled=native_hc_normal(inject[c]/4.f);
  native_hc_need((native_hc_bits(scaled)&0x7fffffffu)<=STRATA_HC_MIX_ACTIVATION_LIMIT_BITS,"HC pending: sigmoid normal-domain limit");
  const float e=native_hc_normal(std::exp(-scaled)),den=native_hc_normal(1.f+e);
  native_hc_need(e>0.f&&den<=native_hc_value(STRATA_HC_MIX_MAX_DENOMINATOR_BITS),"HC pending: exp/denominator domain");
  const float sigma=native_hc_normal(1.f/den);native_hc_need(sigma!=0.f,"HC pending: quotient underflow");
  out.exponential[c]=e;out.gate[c]=native_hc_normal(2.f*sigma);
  for(size_t d=0;d<2560;++d)out.residual[c*2560+d]=native_hc_pending_fma(bo[d],out.gate[c],r[c*2560+d]);
 }return out;
}
}
