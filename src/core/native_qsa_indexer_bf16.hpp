#pragma once
#include "native_shared_layout.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define STRATA_INDEXER_HOST_MXCSR 1
#endif
namespace strata::core::portable_detail {
constexpr size_t qsa_indexer_bf16_input=2560;
inline bool qsa_indexer_bf16_rows(size_t rows){return rows==128 || rows==512;}
inline bool qsa_indexer_bf16_normal_or_zero(float v){uint32_t bits;std::memcpy(&bits,&v,4);const uint32_t magnitude=bits&0x7fffffffu,exponent=magnitude&0x7f800000u;return magnitude==0 || (exponent!=0 && exponent!=0x7f800000u);}
inline bool qsa_indexer_bf16_host_preserves_denormals(){
#ifdef STRATA_INDEXER_HOST_MXCSR
    return (_mm_getcsr()&((1u<<15)|(1u<<6)))==0;
#else
    return false; // No unverified host FTZ policy for this bounded control.
#endif
}
inline float qsa_indexer_bf16_value(uint16_t v){uint32_t bits=uint32_t(v)<<16;float out;std::memcpy(&out,&bits,4);return out;}
inline void native_bf16_matrix_raw_weights(const void* raw,size_t count) {
    const auto* weights=static_cast<const uint8_t*>(raw);
    native_shared_need(weights && (count==qsa_indexer_bf16_input*128 || count==qsa_indexer_bf16_input*512 || count==qsa_indexer_bf16_input*2560),"BF16 matrix: weight span differs");
    for(size_t i=0;i<count;++i){uint16_t value=uint16_t(weights[2*i])|(uint16_t(weights[2*i+1])<<8);uint16_t e=value&0x7f80u,m=value&0x007fu;
        native_shared_need(e!=0x7f80u && (e!=0 || m==0),"QSA indexer BF16: nonfinite or subnormal weight");}
}
inline void qsa_indexer_bf16_raw_weights(const void* raw,size_t count){
    native_shared_need(count==qsa_indexer_bf16_input*128 || count==qsa_indexer_bf16_input*512,"QSA indexer BF16: weight span differs");
    native_bf16_matrix_raw_weights(raw,count);
}
inline void qsa_indexer_bf16_weights(const uint16_t* weights,size_t count){qsa_indexer_bf16_raw_weights(weights,count);}
// The input, both ordered FMA results and every native XOR reduction result
// must remain normal or zero. CUDA FTZ equivalence for subnormals is unproved.
// The resulting host values are controls only; the device remains the producer.
inline std::vector<float> native_bf16_matrix_control(const float* input,const uint16_t* weights,size_t rows) {
    native_shared_need(input && (qsa_indexer_bf16_rows(rows) || rows==2560) && std::fegetround()==FE_TONEAREST && qsa_indexer_bf16_host_preserves_denormals(),"BF16 matrix: input, rows or host rounding/FTZ mode");
    native_bf16_matrix_raw_weights(weights,qsa_indexer_bf16_input*rows);
    for(size_t i=0;i<qsa_indexer_bf16_input;++i)native_shared_need(qsa_indexer_bf16_normal_or_zero(input[i]),"BF16 matrix: nonfinite or subnormal input");
    std::vector<float> out(rows);
    for(size_t row=0;row<rows;++row){std::array<float,256> sums{};std::array<float,32> warps{};
        for(size_t t=0;t<256;++t)for(size_t pair=t;pair<1280;pair+=256){
            const size_t j=2*pair,base=row*qsa_indexer_bf16_input+j;
            sums[t]=std::fma(qsa_indexer_bf16_value(weights[base]),input[j],sums[t]);
            native_shared_need(qsa_indexer_bf16_normal_or_zero(sums[t]),"BF16 matrix: first FMA outside finite normal domain");
            sums[t]=std::fma(qsa_indexer_bf16_value(weights[base+1]),input[j+1],sums[t]);
            native_shared_need(qsa_indexer_bf16_normal_or_zero(sums[t]),"BF16 matrix: second FMA outside finite normal domain");
        }
        for(size_t off=16;off;off>>=1){auto previous=sums;for(size_t t=0;t<256;++t){sums[t]=previous[t]+previous[t^off];native_shared_need(qsa_indexer_bf16_normal_or_zero(sums[t]),"BF16 matrix: warp reduction outside finite normal domain");}}
        for(size_t warp=0;warp<8;++warp)warps[warp]=sums[warp*32];
        for(size_t off=16;off;off>>=1){auto previous=warps;for(size_t t=0;t<32;++t){warps[t]=previous[t]+previous[t^off];native_shared_need(qsa_indexer_bf16_normal_or_zero(warps[t]),"BF16 matrix: block reduction outside finite normal domain");}}
        out[row]=warps[0];
    }
    return out;
}
inline std::vector<float> qsa_indexer_bf16_control(const float* input,const uint16_t* weights,size_t rows) {
    native_shared_need(qsa_indexer_bf16_rows(rows), "QSA indexer BF16: unsupported rows");
    return native_bf16_matrix_control(input,weights,rows);
}
}
