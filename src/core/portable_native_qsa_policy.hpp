#pragma once
#include "native_hc_layout.hpp"
namespace strata::core::portable_detail {
inline void native_qsa_model_environment(){
 // These native switches select a different method, not a Vulkan approximation.
 for(const char*key:{"STRATA_NO_NORM_ROPE","STRATA_ROPE_TABLE"}){
  const char*v=std::getenv(key);native_hc_need(!v||!*v||*v=='0',"native QSA norm/RoPE: analytic fused text method required; table/disabled fusion unsupported");
 }
}
}
