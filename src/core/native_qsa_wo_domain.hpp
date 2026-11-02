#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core::portable_detail {
// Admission for the bounded component domain. This does not quantize inputs or
// alter either native operation. Q8_0 and Q8_1 keep their original bytes.
struct QsaWoScaleClasses { size_t zero=0, normal=0, subnormal=0, nonfinite=0; };
inline QsaWoScaleClasses qsa_wo_scale_classes(const uint8_t* data,size_t blocks,size_t stride) {
    if(!data || stride<2) throw std::runtime_error("QSA WO: missing scale bytes");
    QsaWoScaleClasses c;
    for(size_t b=0;b<blocks;++b){const auto h=uint16_t(data[b*stride])|(uint16_t(data[b*stride+1])<<8);
        unsigned exponent=(h>>10)&31,mantissa=h&1023;
        if(exponent==31)++c.nonfinite;else if(exponent)++c.normal;else if(mantissa)++c.subnormal;else ++c.zero;
    }
    return c;
}
inline void qsa_wo_require_scales(const QsaWoScaleClasses& c,const std::string& role) {
    if(c.subnormal || c.nonfinite)throw std::runtime_error("QSA WO: "+role+" unsupported half scales: zero="+
        std::to_string(c.zero)+" normal="+std::to_string(c.normal)+" subnormal="+std::to_string(c.subnormal)+
        " nonfinite="+std::to_string(c.nonfinite));
}
inline void qsa_wo_weight_domain(const void* bytes,size_t size,const std::string& role) {
    constexpr size_t blocks=2560u*192u;
    if(size!=blocks*34 || !bytes)throw std::runtime_error("QSA WO: unsupported Q8_0 byte span for "+role);
    qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(bytes),blocks,34),role);
}
inline void qsa_wo_packed_domain(const void* bytes,size_t size) {
    if(size!=192u*36u || !bytes)throw std::runtime_error("QSA WO: unsupported Q8_1 byte span");
    qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(bytes),192,36),"producer Q8_1");
}
inline bool qsa_wo_normal_or_zero(float x) { return std::isfinite(x) && (x==0.f || std::isnormal(x)); }
inline void qsa_wo_input_domain(const std::vector<float>& input) {
    if(input.size()!=6144)throw std::runtime_error("QSA WO: gated input shape");
    for(size_t b=0;b<192;++b){std::array<float,32> sums{},maxima{};
        for(size_t lane=0;lane<32;++lane){float x=input[b*32+lane];
            if(!qsa_wo_normal_or_zero(x))throw std::runtime_error("QSA WO: nonfinite or subnormal gated input");
            sums[lane]=x;maxima[lane]=std::fabs(x);
        }
        for(size_t offset=16;offset;offset>>=1){const auto old=sums,oldmax=maxima;
            for(size_t lane=0;lane<32;++lane){sums[lane]=old[lane]+old[lane^offset];maxima[lane]=std::fmax(oldmax[lane],oldmax[lane^offset]);
                if(!qsa_wo_normal_or_zero(sums[lane]))throw std::runtime_error("QSA WO: unsupported reduction sum");
            }
        }
        if(maxima[0]){float d=std::min(maxima[0]/127.f,65504.f);
            if(!std::isnormal(d))throw std::runtime_error("QSA WO: unsupported full scale");
            for(size_t lane=0;lane<32;++lane)if(!qsa_wo_normal_or_zero(input[b*32+lane]/d))throw std::runtime_error("QSA WO: unsupported ratio");
        }
    }
}
}
