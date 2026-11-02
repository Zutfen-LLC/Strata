#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace strata::core::portable_detail {
constexpr size_t q8_wo_input_bytes=192u*36u;
constexpr size_t q8_wo_row_bytes=192u*34u;
inline void q8_wo_require_layout(size_t rows,size_t weights,size_t input,size_t output) {
    if(!rows || rows>2560 || weights!=rows*q8_wo_row_bytes || input!=q8_wo_input_bytes || output!=rows*sizeof(float))
        throw std::runtime_error("Q8 WO: unsupported shape or buffer size");
}
struct Q8WoSpan {const void* owner;uintptr_t address;size_t bytes;};
inline bool q8_wo_overlap(const Q8WoSpan& a,const Q8WoSpan& b){
    if(a.owner!=b.owner)return false;
    return a.address<=b.address?b.address-a.address<a.bytes:a.address-b.address<b.bytes;
}
inline void q8_wo_require_nonoverlap(const Q8WoSpan& w,const Q8WoSpan& x,const Q8WoSpan& y){
    if(q8_wo_overlap(w,x)||q8_wo_overlap(w,y)||q8_wo_overlap(x,y))throw std::runtime_error("Q8 WO: overlapping tensors");
}
inline void q8_wo_require_budget(size_t used,size_t budget){if(used>budget)throw std::runtime_error("Q8 WO: actual buffer budget exceeded");}
}
