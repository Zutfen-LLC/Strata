#pragma once
#include "native_qsa_select_layout.hpp"
namespace strata::core::portable_detail {
constexpr uint32_t qsa_attn_chunks=33,qsa_attn_heads=24,qsa_attn_head=256,qsa_attn_cap=2051;
constexpr uint32_t qsa_attn_acc=33*24*256,qsa_attn_meta=33*24;
constexpr uint32_t qsa_attn_scratch=qsa_attn_acc+2*qsa_attn_meta+64;
constexpr uint32_t qsa_attn_diag=24*33*64;
inline void native_qsa_attention_geometry(uint32_t cells,const std::array<int32_t,4>&s){native_qsa_select_geometry(cells,s);}
inline void native_qsa_attention_environment(){for(const char*k:{"STRATA_EMULATE_CC","STRATA_ATTN_DAILY","STRATA_ATTN_PRE75","STRATA_ATTN_MERGE_V2","STRATA_ATTN_LANECELL","STRATA_ATTN_CHUNK","STRATA_QSA_SPLIT_ATTN","STRATA_ATTN_V3","STRATA_KV_ROT"})native_hc_need(std::getenv(k)==nullptr,"native QSA attention: alternative native attention/KV dispatch unsupported");}
}
