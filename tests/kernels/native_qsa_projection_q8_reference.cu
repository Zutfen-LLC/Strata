#include "native_qsa_projection_q8_fixture.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
using namespace native_projection_q8;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Stream{cudaStream_t p=nullptr;Stream(){cu(cudaStreamCreate(&p));}~Stream(){if(p)cudaStreamDestroy(p);}Stream(const Stream&)=delete;};
struct Device{void*p=nullptr;size_t n;explicit Device(size_t size):n(size){cu(cudaMalloc(&p,n));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;};
int main(int argc,char**argv){try{
    if(argc==4&&std::string(argv[1])=="--contract"){contract(read(argv[2]),argv[3]);return 0;}
    need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","projection q8: invalid native arguments");
    auto input=read(argv[6]);std::string idx=argv[2];need(!idx.empty()&&idx.find_first_not_of("0123456789")==std::string::npos,"projection q8: invalid device");int device=std::stoi(idx);cu(cudaSetDevice(device));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,device));need(std::string(argv[4]).size()&&std::string(prop.name).find(argv[4])!=std::string::npos&&prop.major==8&&prop.minor==6,"projection q8: unexpected native device");
    std::filesystem::path dir=argv[8];need(std::filesystem::create_directory(dir),"projection q8: output exists");std::filesystem::copy_file(argv[6],dir/"fixture.bin");trace(input,dir/"host-rne-trace.csv");
    std::vector<uint32_t>xi(elements+16,canary),qi(bytes/4+16,canary);std::memcpy(xi.data()+8,input.data(),elements*4);Device dx(xi.size()*4),dq(qi.size()*4);Stream stream;Bytes first;
    for(int pass=0;pass<2;++pass){cu(cudaMemcpy(dx.p,xi.data(),xi.size()*4,cudaMemcpyHostToDevice));cu(cudaMemcpy(dq.p,qi.data(),qi.size()*4,cudaMemcpyHostToDevice));strata::kernels::native_quantize_q8_1((float*)dx.p+8,(uint32_t*)dq.p+8,elements,1,stream.p);cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));std::vector<uint32_t>a(xi.size()),b(qi.size());cu(cudaMemcpy(a.data(),dx.p,a.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(b.data(),dq.p,b.size()*4,cudaMemcpyDeviceToHost));Bytes out(bytes);std::memcpy(out.data(),b.data()+8,bytes);save(dir/(pass?"qx-repeat.bin":"qx.bin"),out);need(a==xi,"projection q8: native input mutated");for(size_t i=0;i<8;++i)need(b[i]==canary&&b[b.size()-1-i]==canary,"projection q8: native canary changed");if(pass)need(out==first,"projection q8: native repeat mismatch");first=out;}
#ifndef STRATA_PROJECTION_Q8_REPLAY
    pinned(first);
#endif
    std::ofstream(dir/"report.json")<<"{\"actual_unchanged_native_launcher\":true,\"immutable_canary_repeat\":true,"
#ifdef STRATA_PROJECTION_Q8_REPLAY
      "\"replay_fixture\":true,"
#endif
      "\"qualified\":false}\n";return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
