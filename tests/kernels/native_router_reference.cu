// Calls the existing native launcher; never duplicates the device kernel.
#include "native_router_fixture.hpp"
#include "strata/kernels/native_router.hpp"
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
        "usage: native_router_reference --device INDEX --expect-device DESCRIPTION --fixture FILE --output NEW_PREFIX");
    need(std::fesetround(FE_TONEAREST)==0,"router requires host RNE reference");
    const auto f=read_fixture(argv[6]);const std::string prefix=argv[8];reserve(prefix);
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"router invalid native device");
    const int device=std::stoi(index);cuda_need(cudaSetDevice(device));cudaDeviceProp properties{};cuda_need(cudaGetDeviceProperties(&properties,device));
    need(std::string(argv[4]).size() && std::string(properties.name).find(argv[4])!=std::string::npos,"router unexpected native device");
    Stream stream;cuda_need(cudaStreamCreate(&stream.value));
    const auto before=padded(f);auto after=before;std::array<Allocation,3> memory;
    for(size_t i=0;i<3;++i){cuda_need(cudaMalloc(&memory[i].data,before[i].size()*4));cuda_need(cudaMemcpyAsync(memory[i].data,before[i].data(),before[i].size()*4,cudaMemcpyHostToDevice,stream.value));}
    auto ptr=[&](size_t i){return static_cast<float*>(memory[i].data)+prefixes[i];};
    strata::kernels::native_router_top10_multi(ptr(0),reinterpret_cast<int32_t*>(ptr(1)),ptr(2),int(f.tokens),stream.value);
    cuda_need(cudaStreamSynchronize(stream.value));
    for(size_t i=0;i<3;++i)cuda_need(cudaMemcpy(after[i].data(),memory[i].data,after[i].size()*4,cudaMemcpyDeviceToHost));
    receipt(prefix,f,before,after);
    printf("native_router_reference=true qualified=false device=%s tokens=%u\n",properties.name,f.tokens);
    return 0;
}catch(const std::exception& e){fprintf(stderr,"native router reference: %s\n",e.what());return 1;}}
