// Boundaries call the existing native launchers. The final result is also
// checked against shared_expert_multi itself, with native BF16 explicitly on.
#include "native_shared_fixture.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/shared_expert.hpp"
#include <cuda_runtime.h>
using namespace native_shared_fixture;
namespace {
void cuda_need(cudaError_t s) {if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Allocation {void* data=nullptr;~Allocation(){if(data)cudaFree(data);}};
struct Stream {cudaStream_t value=nullptr;~Stream(){if(value)cudaStreamDestroy(value);}};
__global__ void capture_swiglu(const float* gate,const float* up,float* h,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)h[i]=__fdividef(gate[i],1.f+__expf(-gate[i]))*up[i];
}
__global__ void capture_scale(const float* raw,const float* down,float* out,float* sigmoid,int h) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    float s=__fdividef(1.f,1.f+__expf(-raw[t]));
    if(i<h)out[t*h+i]=down[t*h+i]*s;if(i==0)sigmoid[t]=s;
}
}
int main(int argc,char** argv) {try {
    arguments(argc,argv);const auto f=read_fixture(argv[6]);const std::string dir=argv[8];reserve(dir);
    const int device=std::stoi(argv[2]);cuda_need(cudaSetDevice(device));cudaDeviceProp properties{};cuda_need(cudaGetDeviceProperties(&properties,device));
    need(std::string(properties.name).find(argv[4])!=std::string::npos,"shared unexpected CUDA device");
    Stream stream;cuda_need(cudaStreamCreate(&stream.value));
    const auto before=padded(f);auto after=before;std::array<Allocation,N> memory;
    for(size_t i=0;i<N;++i) {
        cuda_need(cudaMalloc(&memory[i].data,before[i].size()*4));
        cuda_need(cudaMemcpyAsync(memory[i].data,before[i].data(),before[i].size()*4,cudaMemcpyHostToDevice,stream.value));
    }
    auto ptr=[&](size_t i){return static_cast<float*>(memory[i].data)+prefix(i);};
    using namespace strata::kernels;
    native_mmvq_set_multi_exact(true);
    native_quantize_q8_1(ptr(X),ptr(QX),f.h,f.t,stream.value);
    native_mmvq(8,ptr(WG),ptr(QX),ptr(GATE),f.h,f.f,f.t,stream.value);
    native_mmvq(8,ptr(WU),ptr(QX),ptr(UP),f.h,f.f,f.t,stream.value);
    capture_swiglu<<<(f.f*f.t+127)/128,128,0,stream.value>>>(ptr(GATE),ptr(UP),ptr(H),f.f*f.t);
    native_quantize_q8_1(ptr(H),ptr(QH),f.f,f.t,stream.value);
    native_mmvq(8,ptr(WD),ptr(QH),ptr(DOWN),f.f,f.h,f.t,stream.value);
    bf16_gemv_fp32_mmvf_multi(ptr(X),f.h,reinterpret_cast<uint16_t*>(ptr(WS)),ptr(RAW),1,f.h,1,f.t,stream.value);
    capture_scale<<<dim3((f.h+127)/128,f.t),128,0,stream.value>>>(ptr(RAW),ptr(DOWN),ptr(OUT),ptr(SIGMOID),f.h);
    cuda_need(cudaGetLastError());cuda_need(cudaStreamSynchronize(stream.value));
    for(size_t i=0;i<N;++i)cuda_need(cudaMemcpy(after[i].data(),memory[i].data,after[i].size()*4,cudaMemcpyDeviceToHost));
    // Whole native entry point, with independent scratch. Keep all boundary
    // captures intact. This detects drift in these two diagnostic unary kernels.
    std::array<Allocation,5> whole;
    const size_t sizes[]={size_t(f.f)*f.t*4,size_t(f.f)*f.t*4,size_t(f.t)*4,size_t(f.h)*f.t*4,
        size_t(std::max(f.h,f.f)/32)*36*f.t};
    for(size_t i=0;i<5;++i)cuda_need(cudaMalloc(&whole[i].data,sizes[i]));
    NativeSharedWeights nw;nw.gate_type=nw.up_type=nw.down_type=8;
    nw.gate_data=ptr(WG);nw.up_data=ptr(WU);nw.down_data=ptr(WD);nw.q8_1=whole[4].data;
    shared_expert_set_native_bf16(true);
    shared_expert_multi(f.t,ptr(X),nullptr,nw,reinterpret_cast<uint16_t*>(ptr(WS)),static_cast<float*>(whole[0].data),
        static_cast<float*>(whole[1].data),static_cast<float*>(whole[2].data),static_cast<float*>(whole[3].data),f.h,f.f,stream.value);
    cuda_need(cudaStreamSynchronize(stream.value));std::vector<uint32_t> final(size_t(f.h)*f.t);
    cuda_need(cudaMemcpy(final.data(),whole[3].data,sizes[3],cudaMemcpyDeviceToHost));
    need(std::equal(final.begin(),final.end(),after[OUT].begin()+prefix(OUT)),"shared diagnostic sequence differs from native shared_expert_multi");
    receipt(dir,f,before,after);printf("native_shared_reference=true whole_entrypoint_exact=true qualified=false device=%s\n",properties.name);return 0;
}catch(const std::exception& e){fprintf(stderr,"native shared reference: %s\n",e.what());return 1;}}
