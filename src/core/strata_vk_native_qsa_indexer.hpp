#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::core::portable_detail {
struct StrataVkNativeQsaIndexer;
StrataVkNativeQsaIndexer* strata_vk_native_qsa_indexer_create(ggml_backend_t,const char*,size_t budget=65536);
void strata_vk_native_qsa_indexer_destroy(StrataVkNativeQsaIndexer*)noexcept;
size_t strata_vk_native_qsa_indexer_bytes(const StrataVkNativeQsaIndexer*);
uint64_t strata_vk_native_qsa_indexer_calls(const StrataVkNativeQsaIndexer*)noexcept;
// raw/gamma F32[128], immutable before/candidate I32[qsa_idx_words(cells)],
// F32 diagnostic[386]. Candidate is never committed by this component.
// T1 contiguous text/None scaling, cells/base/position within256, eps1e-6.
uint32_t strata_vk_native_qsa_indexer_run(StrataVkNativeQsaIndexer*,ggml_tensor*const[5],uint32_t cells,int32_t pos,int32_t base,uint32_t theta);
}
