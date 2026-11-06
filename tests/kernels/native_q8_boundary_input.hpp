#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
// Keep every scale normal in FP32 under exact division. A smaller nonzero
// maximum can make the CUDA FTZ division see 0/0; that domain is not silently
// assigned a quantized result by this observation tool. Bound all magnitudes
// at 127, keeping the 32-value sum representable as finite binary16.
inline void boundary_input_check(const std::array<unsigned char,512>& input) {
    for(size_t block=0;block<4;++block) {
        uint32_t max=0;
        for(size_t j=0;j<32;++j) {
            uint32_t bits; std::memcpy(&bits,input.data()+(block*32+j)*4,4);
            uint32_t mag=bits&0x7fffffffu;
            if(mag>0x42fe0000u) throw std::runtime_error("unsupported: non-finite or excessive input");
            if(mag>max) max=mag;
        }
        if(max<0x04000000u) throw std::runtime_error("unsupported: Q8 block scale may underflow FP32");
    }
}
