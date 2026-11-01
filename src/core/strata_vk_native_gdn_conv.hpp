#pragma once
#include "ggml-backend.h"
#include <cstddef>
namespace strata::core::portable_detail {
// Standalone ordinary T1/C10240/S128/qk32/eps1e-6; no product default change.
struct StrataVkNativeGdnConv;
StrataVkNativeGdnConv* strata_vk_native_gdn_conv_create(ggml_backend_t,const char*,size_t budget=65536);
void strata_vk_native_gdn_conv_destroy(StrataVkNativeGdnConv*) noexcept;
size_t strata_vk_native_gdn_conv_bytes(const StrataVkNativeGdnConv*);
// history[30720],qkv[10240],weights[40960]; separate candidate history/h,
// pre,exp,SiLU[10240] and rs/arg[32]. All exact disjoint contiguous F32 spans.
// commit=false copies history unchanged; commit=true produces [hist1,hist2,qkv].
// Neither option mutates live input history; caller owns candidate commit.
void strata_vk_native_gdn_conv_run(StrataVkNativeGdnConv*,ggml_tensor*const[10],bool commit);
}
