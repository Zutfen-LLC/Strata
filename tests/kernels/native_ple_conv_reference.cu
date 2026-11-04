// Thin standalone driver for the unchanged native_ple_postops launcher.
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ple.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
constexpr size_t N=2560,D=10240,HISTORY=9*D;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<uint32_t> read(const std::filesystem::path& p,size_t count,bool packed=false){
 need(!std::filesystem::is_symlink(p),"fixture is a symlink");std::ifstream f(p,std::ios::binary);std::vector<uint32_t> v(count);f.read(reinterpret_cast<char*>(v.data()),count*4);need(f.gcount()==std::streamsize(count*4)&&f.peek()==EOF,"fixture length differs");
 for(auto u:v){if(packed){need((u&0x7c00u)!=0x7c00u&&((u>>16)&0x7c00u)!=0x7c00u,"fixture nonfinite half weight");}else{u&=0x7fffffffu;need(u<0x7f800000u&&(u==0||u>=0x00800000u),"fixture finite normal/zero required");}}return v;
}
void save(const std::filesystem::path& p,const void* data,size_t bytes){need(!std::filesystem::exists(p),"output exists");std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),bytes);f.flush();need(bool(f),"output write failed");}
struct Fixture {std::vector<uint32_t> key,hidden,nk,nq,value,nc,history,conv;explicit Fixture(const std::filesystem::path& d):key(read(d/"projected-key.f32le",D)),hidden(read(d/"hidden.f32le",D)),nk(read(d/"norm-key.f32le",D)),nq(read(d/"norm-query.f32le",D)),value(read(d/"value.f32le",N)),nc(read(d/"norm-conv.f32le",D)),history(read(d/"history.f32le",HISTORY)),conv(read(d/"weights.f16le",D*2,true)) {}};
struct Buffer {
 static constexpr uint32_t marker=0xa5a5a5a5u;
 size_t words;std::vector<uint32_t> initial;void* device=nullptr;
 explicit Buffer(size_t n):words(n),initial(n+16,marker){std::fill(initial.begin()+8,initial.end()-8,0u);cu(cudaMalloc(&device,initial.size()*4));}
 ~Buffer(){if(device)cudaFree(device);} Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
 template<class T>T* data(){return reinterpret_cast<T*>(static_cast<uint32_t*>(device)+8);}
 void assign(const std::vector<uint32_t>& v){need(v.size()==words,"input size");std::copy(v.begin(),v.end(),initial.begin()+8);}
 void reset(){cu(cudaMemcpy(device,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));}
 std::vector<uint32_t> fetch(bool immutable){std::vector<uint32_t> v(initial.size());cu(cudaMemcpy(v.data(),device,v.size()*4,cudaMemcpyDeviceToHost));
  for(size_t j=0;j<8;++j)need(v[j]==marker&&v[v.size()-1-j]==marker,"canary changed");if(immutable)need(v==initial,"input or weights changed");return {v.begin()+8,v.end()-8};}
};
struct Stream{cudaStream_t s=nullptr;Stream(){cu(cudaStreamCreate(&s));}~Stream(){if(s)cudaStreamDestroy(s);}};
void finite(const std::vector<uint32_t>& data){for(auto x:data)need((x&0x7fffffffu)<0x7f800000u,"nonfinite native capture");}
}
int main(int argc,char**argv){try{
 const uint16_t endian=1;need(*reinterpret_cast<const uint8_t*>(&endian)==1,"little-endian host required");
 if(argc==4&&std::string(argv[1])=="--contract"){Fixture f(argv[2]);const char j[]="{\"history_bytes\":368640,\"channel_bytes\":40960,\"weights_bytes\":81920,\"candidate_history_bytes\":368640,\"qualified\":false}\n";save(argv[3],j,sizeof(j)-1);return 0;}
 need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","usage: --device ID --expect-device NAME --fixture DIR --output NEW_DIR");
 Fixture f(argv[6]);std::string dev=argv[2];need(!dev.empty()&&dev.find_first_not_of("0123456789")==std::string::npos,"invalid device index");cu(cudaSetDevice(std::stoi(dev)));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,std::stoi(dev)));need(*argv[4]&&std::string(prop.name).find(argv[4])!=std::string::npos&&prop.major==8&&prop.minor==6,"native device differs");
 std::filesystem::path output=argv[8];need(std::filesystem::create_directory(output),"fresh output required");
 Buffer key(D),hidden(D),nk(D),nq(D),value(N),history(HISTORY),history_candidate(HISTORY),nc(D),conv(D*2);
 key.assign(f.key);hidden.assign(f.hidden);nk.assign(f.nk);nq.assign(f.nq);
 value.assign(f.value);nc.assign(f.nc);history.assign(f.history);history_candidate.assign(f.history);conv.assign(f.conv);
 Buffer out_key(D),out_query(D),out_gate(4),out_gated(D),out_norm(D),out_conv(D),out_result(D);
 std::vector<Buffer*> inputs{&key,&hidden,&nk,&nq,&value,&history,&nc,&conv};
 std::vector<Buffer*> outputs{&out_key,&out_query,&out_gate,&out_gated,&out_norm,&out_conv,&out_result};
 strata::kernels::PleWeights w{};w.norm_key=nk.data<float>();w.norm_query=nq.data<float>();w.norm_conv=nc.data<float>();w.conv1d_f16=conv.data<uint16_t>();
 // The native post-op permits aliasing query and normalized, but the later norm
 // overwrites query. Keep them separate to retain the actual gate input.
 strata::kernels::NativePlePostopsBuffers b{out_key.data<float>(),out_query.data<float>(),out_gate.data<float>(),out_gated.data<float>(),out_norm.data<float>(),out_conv.data<float>(),out_result.data<float>()};
 Stream stream;std::vector<std::vector<uint32_t>> first;
 for(int pass=0;pass<2;++pass){for(auto* p:inputs)p->reset();history_candidate.reset();for(auto* p:outputs)p->reset();
  bool refused=false;try{strata::kernels::native_ple_postops(key.data<float>(),hidden.data<float>(),value.data<float>(),history.data<float>(),w,b,nullptr);}catch(...){refused=true;}need(refused,"null stream accepted");
  strata::kernels::native_ple_postops(key.data<float>(),hidden.data<float>(),value.data<float>(),history.data<float>(),w,b,stream.s);cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.s));
  strata::kernels::ple_history_advance(history_candidate.data<float>(),out_norm.data<float>(),stream.s);cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.s));
  for(auto* p:inputs)(void)p->fetch(true);auto next=history_candidate.fetch(false);std::vector<std::vector<uint32_t>> now;for(auto* p:outputs){auto v=p->fetch(false);finite(v);now.push_back(std::move(v));}
  now.push_back(std::move(next));if(pass)need(now==first,"native repeat differs");first=now;
  const char* suffix=pass?"-repeat.f32le":".f32le";save(output/(std::string("key")+suffix),now[0].data(),D*4);save(output/(std::string("query")+suffix),now[1].data(),D*4);save(output/(std::string("gate")+suffix),now[2].data(),4*4);
  save(output/(std::string("gated")+suffix),now[3].data(),D*4);save(output/(std::string("normalized")+suffix),now[4].data(),D*4);save(output/(std::string("conv")+suffix),now[5].data(),D*4);save(output/(std::string("result")+suffix),now[6].data(),D*4);save(output/(std::string("next-history")+suffix),now[7].data(),HISTORY*4);
 }
 const char report[]="{\"native_ple_postops\":true,\"query_normalized_distinct\":true,\"repeat_and_canaries\":true,\"inputs_unchanged\":true,\"qualified\":false}\n";save(output/"report.json",report,sizeof(report)-1);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"native PLE conv: %s\n",e.what());return 1;}}
