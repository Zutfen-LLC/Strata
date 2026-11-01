#pragma once
#include "native_hc_layout.hpp"
#include <cmath>
#include <cstring>
namespace strata::core::portable_detail {
inline uint32_t native_qsa_theta_scale(){float v=::powf(1e7f,-2.f/64.f);uint32_t b;std::memcpy(&b,&v,4);return b;}
inline void native_qsa_norm_rope_geometry(uint32_t cols,uint32_t rows,uint32_t stride,uint32_t theta){native_hc_need((cols==128||cols==256)&&rows>=1&&rows<=24&&(stride==cols||stride==2*cols)&&theta==native_qsa_theta_scale(),"native QSA norm/RoPE: text T1 widths128/256 rows1..24 stride and None-scaling theta required");}
}
