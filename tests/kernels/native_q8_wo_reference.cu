#include "native_q8_wo_fixture.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
using namespace native_q8_wo_fixture;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Stream{cudaStream_t p=nullptr;Stream(){cu(cudaStreamCreate(&p));}~Stream(){if(p)cudaStreamDestroy(p);}Stream(const Stream&)=delete;};
struct Device{void* p=nullptr;explicit Device(size_t n){cu(cudaMalloc(&p,n));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;};
int main(int argc,char**argv){try{
 if(argc==6&&std::string(argv[1])=="--contract"){auto f=read_fixture(argv[2],argv[3],std::stoull(argv[4]));contract(f,argv[5]);return 0;}
 need(argc==15&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--weights"&&std::string(argv[7])=="--input"&&std::string(argv[9])=="--rows"&&std::string(argv[11])=="--budget"&&std::string(argv[13])=="--output","WO: invalid native arguments");
 auto f=read_fixture(argv[6],argv[8],std::stoull(argv[10]));size_t budget=std::stoull(argv[12]);
 need(f.weights.size()+f.input.size()+f.rows*4+192<=budget,"WO: native allocation budget");
 std::string idx=argv[2];need(!idx.empty()&&idx.find_first_not_of("0123456789")==std::string::npos,"WO: invalid device");cu(cudaSetDevice(std::stoi(idx)));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,std::stoi(idx)));need(std::string(argv[4]).size()&&std::string(prop.name).find(argv[4])!=std::string::npos&&prop.major==8&&prop.minor==6,"WO: unexpected device");
 std::filesystem::path dir=argv[14];need(std::filesystem::create_directory(dir),"WO: output exists");
 std::vector<uint32_t>w(f.weights.size()/4+16,0xa5a5a5a5),x(f.input.size()/4+16,0xa5a5a5a5),y(f.rows+16,0xa5a5a5a5);
 std::memcpy(w.data()+8,f.weights.data(),f.weights.size());std::memcpy(x.data()+8,f.input.data(),f.input.size());
 Device dw(w.size()*4),dx(x.size()*4),dy(y.size()*4);Stream stream;std::vector<float> first;
 for(int pass=0;pass<2;++pass){cu(cudaMemcpy(dw.p,w.data(),w.size()*4,cudaMemcpyHostToDevice));cu(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));cu(cudaMemcpy(dy.p,y.data(),y.size()*4,cudaMemcpyHostToDevice));
  strata::kernels::native_mmvq(8,(uint32_t*)dw.p+8,(uint32_t*)dx.p+8,(float*)((uint32_t*)dy.p+8),6144,int(f.rows),1,stream.p);cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));
  std::vector<uint32_t>rw(w.size()),rx(x.size()),ry(y.size());cu(cudaMemcpy(rw.data(),dw.p,rw.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(rx.data(),dx.p,rx.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(ry.data(),dy.p,ry.size()*4,cudaMemcpyDeviceToHost));
  need(rw==w&&rx==x,"WO: native input mutated");for(size_t i=0;i<8;++i)need(ry[i]==0xa5a5a5a5&&ry[ry.size()-1-i]==0xa5a5a5a5,"WO: native output canary changed");
  std::vector<float> result(f.rows);std::memcpy(result.data(),ry.data()+8,f.rows*4);output_literals(result);if(pass)need(std::memcmp(result.data(),first.data(),result.size()*4)==0,"WO: native repeat differs");first=result;
  Bytes data(f.rows*4);std::memcpy(data.data(),result.data(),data.size());save(dir/(pass?"out-repeat.f32le":"out.f32le"),data);
 }
 std::ofstream(dir/"report.json")<<"{\"actual_native_mmvq_type8\":true,\"immutable_canary_repeat\":true,\"qualified\":false}\n";return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
