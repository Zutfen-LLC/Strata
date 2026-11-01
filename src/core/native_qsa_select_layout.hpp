#pragma once
#include "native_hc_layout.hpp"
#include <array>
#include <cstdint>
namespace strata::core::portable_detail {
constexpr uint32_t qsa_select_cap=2051, qsa_select_cells=4096;
inline uint32_t qsa_select_blocks(uint32_t cells){return cells/4+2;}
inline void native_qsa_select_geometry(uint32_t cells,const std::array<int32_t,4>&s){
 native_hc_need(cells>=1&&cells<=qsa_select_cells&&s[0]>=0&&uint32_t(s[0])<cells&&s[1]==s[0]+1&&s[2]==s[1]/4&&s[3]==(s[1]<int(qsa_select_cap)?s[1]:int(qsa_select_cap)),"native QSA select: bounded T1 step pos/nkv/nbid/width required");
}
inline void native_qsa_select_environment(){
 // This first port is the default CUDA Ampere T1 multi scorer/register radix
 // selector. Do not silently force another native dispatch for the oracle.
 for(const char*key:{"STRATA_EMULATE_CC","STRATA_SCORES_HALVE","STRATA_SCORES_MULTI","STRATA_QSA_ROWS","STRATA_QSA_EARLY_EXIT","STRATA_TOPK_OLD","STRATA_QSA_CLUSTER","STRATA_TOPK_CAPACITY_GUARD","STRATA_TOPK_ACTIVE_ANY","STRATA_TOPK_STREAM","STRATA_TOPK_SPLIT","STRATA_TOPK_SAMPLE"})native_hc_need(std::getenv(key)==nullptr,"native QSA select: explicit alternative native dispatch unsupported in this component");
}
}
