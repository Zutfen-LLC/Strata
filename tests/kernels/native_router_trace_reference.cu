// Separate opt-in kernel generated from the exact hash-pinned native body.
#include "native_router_trace_fixture.hpp"
#include "native_router_trace_kernel.cuh"
#include <cuda_runtime.h>
#include <cfenv>
using namespace native_router_fixture;
namespace {
void cuda_need(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct Allocation {void* data=nullptr;~Allocation(){if(data)cudaFree(data);}};
struct Stream {cudaStream_t value=nullptr;~Stream(){if(value)cudaStreamDestroy(value);}};
}
int main(int argc,char** argv){try {
    need(argc==9 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--fixture" && std::string(argv[7])=="--output",
        "usage: native_router_trace_reference --device INDEX --expect-device DESCRIPTION --fixture FILE --output NEW_PREFIX");
    need(std::fesetround(FE_TONEAREST)==0,"router requires host RNE reference");
    const auto f=read_fixture(argv[6]);const std::string prefix=argv[8];validate_trace_fixture(f);reserve(prefix);
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"router invalid native device");
    const int device=std::stoi(index);cuda_need(cudaSetDevice(device));cudaDeviceProp properties{};cuda_need(cudaGetDeviceProperties(&properties,device));
    need(std::string(argv[4]).size() && std::string(properties.name).find(argv[4])!=std::string::npos,"router unexpected native device");
    Stream stream;cuda_need(cudaStreamCreate(&stream.value));
    const auto before=padded(f);auto after=before;std::array<Allocation,3> memory;
    for(size_t i=0;i<3;++i){cuda_need(cudaMalloc(&memory[i].data,before[i].size()*4));cuda_need(cudaMemcpyAsync(memory[i].data,before[i].data(),before[i].size()*4,cudaMemcpyHostToDevice,stream.value));}
    auto ptr=[&](size_t i){return static_cast<float*>(memory[i].data)+prefixes[i];};
    Allocation trace_memory;std::array<uint32_t,34> trace_before,trace_after;trace_before.fill(0xa5a5a5a5);
    cuda_need(cudaMalloc(&trace_memory.data,sizeof(trace_before)));
    cuda_need(cudaMemcpyAsync(trace_memory.data,trace_before.data(),sizeof(trace_before),cudaMemcpyHostToDevice,stream.value));
    strata_router_diagnostic::route_multi_diagnostic<<<1,dim3(32,8),0,stream.value>>>(
        ptr(0),reinterpret_cast<int32_t*>(ptr(1)),ptr(2),8,static_cast<float*>(trace_memory.data)+4);
    cuda_need(cudaGetLastError());
    cuda_need(cudaStreamSynchronize(stream.value));
    for(size_t i=0;i<3;++i)cuda_need(cudaMemcpy(after[i].data(),memory[i].data,after[i].size()*4,cudaMemcpyDeviceToHost));
    receipt(prefix,f,before,after);
    cuda_need(cudaMemcpy(trace_after.data(),trace_memory.data,sizeof(trace_after),cudaMemcpyDeviceToHost));
    std::array<uint32_t,26> trace{};std::copy(trace_after.begin()+4,trace_after.begin()+30,trace.begin());
    write_trace(prefix,trace);
    for(size_t i=0;i<34;++i)if(i<4 || i>=30)need(trace_after[i]==trace_before[i],"router trace guard mutation");
    complete_trace(prefix);
    printf("native_router_row2_diagnostic=true qualified=false device=%s tokens=%u\n",properties.name,f.tokens);
    return 0;
}catch(const std::exception& e){fprintf(stderr,"native router reference: %s\n",e.what());return 1;}}
