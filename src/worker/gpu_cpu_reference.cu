#include "gpu_cpu_reference.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace strata::worker {
namespace {
struct Q80 {__half scale;int8_t code[32];};
static_assert(sizeof(Q80)==34,"native Q8_0 layout");
__device__ int cpu_signed_product(int w,int x) {
    // The qualified AVX2 path uses unsigned abs(w) and sign(x,w),
    // including the int8 -128 edge; do not replace it with a float dot.
    const int ax=w<0?-w:w;
    const int sx=w<0?(x==-128?-128:-x):(w==0?0:x);
    return ax*sx;
}
__device__ int cpu_pair(int w0,int x0,int w1,int x1) {
    const int sum=cpu_signed_product(w0,x0)+cpu_signed_product(w1,x1);
    return min(32767,max(-32768,sum)); // _mm256_maddubs_epi16 saturation
}
__global__ void q8_avx2_dot_kernel(const unsigned long long* weights,const int32_t* input_row,
                                  const Q80* input,int blocks,int n_out,size_t weight_offset,float* out) {
    const int row=(int)blockIdx.x*32+(int)threadIdx.x/8,lane=(int)threadIdx.x&7;
    // Supported output sizes are multiples of 32: every live warp participates in shuffles.
    if(row>=n_out)return;
    const int entry=(int)blockIdx.y;
    const auto* w=(const Q80*)((const uint8_t*)weights[entry]+weight_offset)+(size_t)row*blocks;
    const auto* x=input+(size_t)input_row[entry]*blocks;
    float acc=0.f;
    for(int b=0;b<blocks;++b) {
        const int j=4*lane;
        const int dot=cpu_pair(w[b].code[j],x[b].code[j],w[b].code[j+1],x[b].code[j+1])+
                      cpu_pair(w[b].code[j+2],x[b].code[j+2],w[b].code[j+3],x[b].code[j+3]);
        const float scale=__fmul_rn(__half2float(w[b].scale),__half2float(x[b].scale));
        acc=__fmaf_rn(scale,(float)dot,acc);
    }
    const int base=((int)threadIdx.x&31)&~7;float a[8];
#pragma unroll
    for(int i=0;i<8;++i)a[i]=__shfl_sync(0xffffffffu,acc,base+i);
    if(lane==0) {
        const float even=__fadd_rn(__fadd_rn(a[4],a[0]),__fadd_rn(a[6],a[2]));
        const float odd=__fadd_rn(__fadd_rn(a[5],a[1]),__fadd_rn(a[7],a[3]));
        out[(size_t)entry*n_out+row]=__fadd_rn(even,odd);
    }
}
}
bool q8_avx2_reference_dot(const unsigned long long* weights,const int32_t* input_row,const uint8_t* input,
                           int n_in,int n_out,int entries,size_t offset,float* out,void* stream) {
    if(!((n_in==2560 && n_out==640)||(n_in==640 && n_out==2560)) || entries<1 || entries>80)return false;
    q8_avx2_dot_kernel<<<dim3((unsigned)n_out/32,(unsigned)entries),256,0,(cudaStream_t)stream>>>
        (weights,input_row,(const Q80*)input,n_in/32,n_out,offset,out);
    return true;
}
}
