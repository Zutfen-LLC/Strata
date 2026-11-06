#include "native_q8_boundary_input.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main(int argc,char** argv) { try {
    if(argc!=6) throw std::runtime_error("usage: DEVICE NAME-SUBSTRING expert|projection INPUT OUTPUT");
    if(std::filesystem::exists(argv[5])) throw std::runtime_error("output already exists");
    const int device=std::stoi(argv[1]); const std::string name=argv[2], mode=argv[3];
    if(name.empty() || (mode!="expert" && mode!="projection")) throw std::runtime_error("invalid selector");
    std::array<unsigned char,512> input{}; std::ifstream f(argv[4],std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(input.data()),input.size()) || f.peek()!=EOF) throw std::runtime_error("input must contain exactly 128 float32 values");
    boundary_input_check(input);
    ck(cudaSetDevice(device)); cudaDeviceProp props{}; ck(cudaGetDeviceProperties(&props,device));
    if(std::string(props.name).find(name)==std::string::npos) throw std::runtime_error("unexpected device");
    float* x=nullptr; void* y=nullptr; cudaStream_t stream=nullptr;
    ck(cudaStreamCreate(&stream)); ck(cudaMalloc(&x,input.size())); ck(cudaMalloc(&y,144));
    ck(cudaMemcpyAsync(x,input.data(),input.size(),cudaMemcpyHostToDevice,stream));
    if(mode=="expert") strata::kernels::quantize_q8_1_rows(x,1,128,y,stream);
    else strata::kernels::native_quantize_q8_1(x,y,128,1,stream);
    std::array<unsigned char,144> output{}; ck(cudaMemcpyAsync(output.data(),y,output.size(),cudaMemcpyDeviceToHost,stream));
    ck(cudaStreamSynchronize(stream)); ck(cudaFree(y)); ck(cudaFree(x)); ck(cudaStreamDestroy(stream));
    std::ofstream out(argv[5],std::ios::binary); out.write(reinterpret_cast<const char*>(output.data()),output.size());
    if(!out) throw std::runtime_error("output write failed");
    std::printf("producer=%s device=%s qualified=false bytes=%zu\n",mode.c_str(),props.name,output.size()); return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"CUDA boundary: %s\n",e.what()); return 1;} }
