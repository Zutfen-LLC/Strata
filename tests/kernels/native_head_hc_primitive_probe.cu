// Focused observation only. Compare generated instructions with the retained
// fused_gr kernels before interpreting these values as the same native method.
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t kInputs=4,kOutputs=10;
constexpr uint32_t kCanary=0x5a17c39du;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
struct Stream { cudaStream_t value=nullptr; Stream(){cu(cudaStreamCreate(&value));} ~Stream(){if(value)cudaStreamDestroy(value);} };
struct Bank {void* value=nullptr;explicit Bank(size_t n){cu(cudaMalloc(&value,n));}~Bank(){if(value)cudaFree(value);} };
}
extern "C" __global__ void native_head_hc_primitive_observation(const uint32_t* in,uint32_t* out){
 if(blockIdx.x||threadIdx.x)return;
 const float up=__uint_as_float(in[0]);
 const float up_exp=__expf(-up);
 const float up_den=1.0f+up_exp;
 const float up_gate=1.0f/up_den;
 const float low=__uint_as_float(in[1]);
 const float low_exp=__expf(-low);
 const float low_den=1.0f+low_exp;
 const float low_gate=low/low_den;
 const float rs_input=__uint_as_float(in[2]);
 const float rs=rsqrtf(rs_input);
 const float captured_den=__uint_as_float(in[3]);
 const float captured_low=low/captured_den;
 out[0]=__float_as_uint(up);out[1]=__float_as_uint(up_exp);out[2]=__float_as_uint(up_den);out[3]=__float_as_uint(up_gate);
 out[4]=__float_as_uint(low);out[5]=__float_as_uint(low_exp);out[6]=__float_as_uint(low_den);out[7]=__float_as_uint(low_gate);
 out[8]=__float_as_uint(rs);out[9]=__float_as_uint(captured_low);
}
int main(int argc,char** argv){try{
 if(argc==2 && std::string(argv[1])=="--contract"){puts("native_head_hc_primitive_probe_v1 fixed_inputs observation_only");return 0;}
 need(argc==7 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" && std::string(argv[5])=="--output","invalid arguments");
 const std::string expected=argv[4];need(!expected.empty(),"empty device identity");
 const std::string target=argv[6];need(std::filesystem::create_directory(target),"fresh output directory required");
 const int device=std::stoi(argv[2]);cu(cudaSetDevice(device));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,device));need(std::string(prop.name).find(expected)!=std::string::npos,"device mismatch");
 const std::array<uint32_t,kInputs> input{0xbf5544aau,0x33c013a7u,0x3e800022u,0x3fffffffu};
 std::array<uint32_t,kOutputs+2> first{},current{};Bank in(sizeof(input)),out(sizeof(current));Stream stream;
 for(int trial=0;trial<2;++trial){
  current.fill(kCanary);cu(cudaMemcpyAsync(in.value,input.data(),sizeof(input),cudaMemcpyHostToDevice,stream.value));cu(cudaMemcpyAsync(out.value,current.data(),sizeof(current),cudaMemcpyHostToDevice,stream.value));
  native_head_hc_primitive_observation<<<1,1,0,stream.value>>>((const uint32_t*)in.value,(uint32_t*)out.value+1);
  cu(cudaGetLastError());cu(cudaMemcpyAsync(current.data(),out.value,sizeof(current),cudaMemcpyDeviceToHost,stream.value));cu(cudaStreamSynchronize(stream.value));
  need(current.front()==kCanary && current.back()==kCanary,"probe canary changed");need(current[1]==input[0] && current[5]==input[1],"probe input echo differs");
  if(trial)need(current==first,"probe repeat differs");else first=current;
 }
 std::ofstream words(target+"/words.u32le",std::ios::binary);words.write(reinterpret_cast<const char*>(first.data()+1),kOutputs*sizeof(uint32_t));need(bool(words),"word write failed");words.close();
 std::ofstream out_json(target+"/report.json");out_json<<"{\"probe\":\"native_head_hc_primitive_v1\",\"inputs\":[\"bf5544aa\",\"33c013a7\",\"3e800022\",\"3fffffff\"],\"words\":[";
 char hex[9];for(size_t i=1;i<=kOutputs;++i){std::snprintf(hex,sizeof(hex),"%08x",first[i]);out_json<<(i==1?"":",")<<"\""<<hex<<"\"";}
 out_json<<"],\"repeat\":true,\"canaries\":true,\"compiled_instruction_review_required\":true,\"qualified\":false}\n";need(bool(out_json),"report write failed");return 0;
}catch(const std::exception& e){fprintf(stderr,"head primitive probe: %s\n",e.what());return 1;}}
