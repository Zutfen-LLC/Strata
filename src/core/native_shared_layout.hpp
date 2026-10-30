#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace strata::core::portable_detail {
inline void native_shared_need(bool ok,const char* why) { if(!ok)throw std::invalid_argument(why); }
inline void native_shared_shape(uint32_t h,uint32_t f,uint32_t t,int type=8) {
    native_shared_need(type==8,"native shared: standalone projections support Q8_0 only (native CUDA supports additional formats)");
    native_shared_need(h>=32 && h<=8192 && f>=32 && f<=8192 && h%32==0 && f%32==0 && t>=1 && t<=8,
        "native shared: require aligned dimensions 32..8192 and T1..8");
}
// weights.cpp Conv::Bf16High recovers the high half of a promoted BF16.
// Do not apply it to an arbitrary F32 tensor. Raw BF16 is already native storage.
inline uint16_t native_shared_promoted_bf16(float x) {
    uint32_t b;std::memcpy(&b,&x,4);
    native_shared_need((b&65535u)==0,"native shared: scalar F32 is not an exact promoted BF16 value");
    native_shared_need((b&0x7f800000u)!=0x7f800000u,"native shared: nonfinite scalar weight");
    return uint16_t(b>>16);
}
inline uint32_t native_shared_mmvf_block(uint32_t h) {
    uint32_t best=32,iterations=(h+63)/64;
    for(uint32_t c=64;c<=256;c+=32)if((h+2*c-1)/(2*c)<iterations) {iterations=(h+2*c-1)/(2*c);best=c;}
    return best;
}
inline uint64_t native_shared_weight_bytes(uint32_t ni,uint32_t no) {return uint64_t(ni/32)*no*34;}
}
