// Actual native_gr_rms_norm_weighted launcher, standalone PLE geometry.
#include "strata/kernels/native_gr_norm.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
constexpr size_t words=10240;constexpr uint32_t marker=0x4acaf00d;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<uint32_t> read(const std::filesystem::path&p){
 need(!std::filesystem::is_symlink(p),"symlink fixture");std::ifstream f(p,std::ios::binary);std::vector<uint32_t> v(words);
 f.read(reinterpret_cast<char*>(v.data()),words*4);need(f.gcount()==words*4&&f.peek()==EOF,"exact input fixture required");
 for(auto x:v){auto v=x&0x7fffffffu;need(v<0x7f800000u&&(v==0||v>=0x00800000u),"unsupported input class");}return v;
}
void save(const std::filesystem::path&p,const void* data,size_t bytes){need(!std::filesystem::exists(p),"output exists");std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),bytes);f.flush();need(bool(f),"output write");}
struct Buffer{uint32_t* p=nullptr;std::vector<uint32_t> initial;
 Buffer():initial(words+16,marker){cu(cudaMalloc(&p,initial.size()*4));}
 ~Buffer(){if(p)cudaFree(p);} float* data(){return reinterpret_cast<float*>(p+8);}
 void reset(const std::vector<uint32_t>* values=nullptr){if(values)std::copy(values->begin(),values->end(),initial.begin()+8);cu(cudaMemcpy(p,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));}
 std::vector<uint32_t> get(bool immutable){std::vector<uint32_t> v(initial.size());cu(cudaMemcpy(v.data(),p,v.size()*4,cudaMemcpyDeviceToHost));
  for(size_t i=0;i<8;++i)need(v[i]==marker&&v[8+words+i]==marker,"canary changed");if(immutable)need(v==initial,"input changed");return {v.begin()+8,v.end()-8};}
};
struct Stream{cudaStream_t value=nullptr;Stream(){cu(cudaStreamCreate(&value));}~Stream(){if(value)cudaStreamDestroy(value);}};
}
int main(int argc,char**argv){try{
 const uint16_t endian=1;need(*reinterpret_cast<const uint8_t*>(&endian)==1,"little-endian host required");
 if(argc==4&&std::string(argv[1])=="--contract"){
  read(std::filesystem::path(argv[2])/"input.f32le");read(std::filesystem::path(argv[2])/"gamma.f32le");
  const char out[]="{\"input_bytes\":40960,\"gamma_bytes\":40960,\"output_bytes\":40960,\"qualified\":false}\n";save(argv[3],out,sizeof(out)-1);return 0;}
 need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","usage: --device ID --expect-device NAME --fixture DIR --output NEW_DIR");
 std::string id=argv[2];need(!id.empty()&&id.find_first_not_of("0123456789")==std::string::npos,"invalid device index");
 const auto fixture=std::filesystem::path(argv[6]);auto x=read(fixture/"input.f32le"),gamma=read(fixture/"gamma.f32le");
 cu(cudaSetDevice(std::stoi(id)));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,std::stoi(id)));
 need(*argv[4]&&std::string(prop.name).find(argv[4])!=std::string::npos,"native device differs");
 std::filesystem::path out=argv[8];need(std::filesystem::create_directory(out),"fresh output required");
 Buffer input,weight,output;Stream stream;std::vector<uint32_t> first;
 for(int pass=0;pass<2;++pass){input.reset(&x);weight.reset(&gamma);output.reset();
  strata::kernels::native_gr_rms_norm_weighted(input.data(),weight.data(),output.data(),2560,4,1e-6f,stream.value);
  cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.value));input.get(true);weight.get(true);auto now=output.get(false);
  for(auto word:now){auto v=word&0x7fffffffu;need(v<0x7f800000u&&(v==0||v>=0x00800000u),"unsupported native output class");}
  if(pass)need(now==first,"native repeat differs");first=now;
  save(out/(pass?"output-repeat.f32le":"output.f32le"),now.data(),words*4);
 }
 const char report[]="{\"native_gr_rms_norm_weighted\":true,\"repeat_and_canaries\":true,\"inputs_unchanged\":true,\"qualified\":false}\n";save(out/"report.json",report,sizeof(report)-1);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"native PLE grouped norm: %s\n",e.what());return 1;}}
