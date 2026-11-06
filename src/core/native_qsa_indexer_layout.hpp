#pragma once
#include "native_qsa_norm_rope_layout.hpp"
namespace strata::core::portable_detail {
// Native QsaIndexerBuffers, packed only for descriptor plumbing. No state
// ownership is transferred: before and candidate are always disjoint.
constexpr uint32_t qsa_idx_tail=0, qsa_idx_dead=384, qsa_idx_pool=512;
inline uint32_t qsa_idx_block(uint32_t cells){return qsa_idx_pool+(cells/4+1)*128;}
inline uint32_t qsa_idx_words(uint32_t cells){return qsa_idx_block(cells)+1;}
// Component geometry only: model admission and the other QSA owners retain
// their own limits until their capacity-dependent memory is accounted for.
inline void native_qsa_indexer_geometry(uint32_t cells,int32_t pos,int32_t base,uint32_t theta){
 native_hc_need(cells>=1&&cells<=4096&&pos>=0&&uint32_t(pos)<cells&&base>=0&&base%4==0&&uint64_t(base)+cells<=4096&&theta==native_qsa_theta_scale(),"native QSA indexer: T1 text/None scaling capacity/base/position within4096 required");
}
}
