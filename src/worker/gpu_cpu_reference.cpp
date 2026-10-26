#include "gpu_cpu_reference.hpp"
#include <cmath>
#include "ggml-cpu.h"
#include <stdexcept>
namespace strata::worker {
namespace cpu = strata::kernels::cpu;
bool q8_cpu_reference_format(cpu::NativeFmt& f,std::string& err) {
    ggml_cpu_init();
    if(!ggml_cpu_has_avx2() || !ggml_cpu_has_fma()) {err="CPU reference requires the qualified GGML AVX2/FMA baseline";return false;}
    if (!cpu::native_fmt(8,8,2560,640,f,err)) return false;
    if (f.act_bytes!=2720 || f.h_bytes!=680 || f.gu_row!=2720 || f.d_row!=680 || f.bytes!=5222400 ||
        f.had_gu || f.had_d || f.tail_off) {err="CPU reference requires plain Q8_0 2560x640";return false;}
    return true;
}
void q8_cpu_reference_input(const cpu::NativeFmt& f,const float* x,int T,std::vector<uint8_t>& packed) {
    if(T<1 || T>8)throw std::runtime_error("CPU reference token bound");
    packed.resize((size_t)T*f.act_bytes);
    for(int t=0;t<T;++t)cpu::native_quant_act(f,x+(size_t)t*2560,packed.data()+(size_t)t*f.act_bytes);
}
void q8_cpu_reference_hidden(const cpu::NativeFmt& f,const float* gate,const float* up,int entries,
                             std::vector<uint8_t>& packed) {
    if(entries<1 || entries>80)throw std::runtime_error("CPU reference entry bound");
    packed.resize((size_t)entries*f.h_bytes);float hidden[640];
    for(int e=0;e<entries;++e) {
        for(int j=0;j<640;++j) {
            const size_t i=(size_t)e*640+j;const float g=gate[i];
            // Same expression and host math library as native_gu_rows. CUDA __expf
            // is intentionally not assumed bit-equivalent to CPU std::exp(float).
            hidden[j]=(g/(1.f+std::exp(-g)))*up[i];
        }
        cpu::native_quant_h(f,hidden,packed.data()+(size_t)e*f.h_bytes);
    }
}
}
