#include "native_qsa_indexer_bf16_fixture.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
using namespace native_qsa_indexer_bf16_fixture;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{void*p=nullptr;explicit Device(size_t n){cu(cudaMalloc(&p,n));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;};
struct Stream{cudaStream_t p=nullptr;Stream(){cu(cudaStreamCreate(&p));}~Stream(){if(p)cudaStreamDestroy(p);}Stream(const Stream&)=delete;};
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3]);return 0;}
 arguments(argc,argv);auto f=read(argv[6]);int dev=std::stoi(argv[2]);cu(cudaSetDevice(dev));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,dev));need(std::string(prop.name).find(argv[4])!=std::string::npos,"indexer BF16 native unexpected device");
 std::filesystem::path dir=argv[8];need(std::filesystem::create_directory(dir),"indexer BF16 output exists");Stream stream;constexpr uint32_t guard=0xa5a5a5a5u;
 std::vector<uint32_t>hx(2560+16,guard);std::memcpy(hx.data()+8,f.x.data(),10240);Device dx(hx.size()*4);cu(cudaMemcpy(dx.p,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
 std::array<std::vector<float>,2>first;for(size_t i=0;i<2;++i){const size_t bytes=f.w[i].size()*2;std::vector<uint32_t>hw(bytes/4+16,guard),hy(rows[i]+16,guard);std::memcpy(hw.data()+8,f.w[i].data(),bytes);Device dw(hw.size()*4),dy(hy.size()*4);cu(cudaMemcpy(dw.p,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));
  for(int pass=0;pass<2;++pass){cu(cudaMemcpy(dy.p,hy.data(),hy.size()*4,cudaMemcpyHostToDevice));strata::kernels::bf16_gemv_fp32_mmvf(static_cast<float*>(dx.p)+8,reinterpret_cast<uint16_t*>(static_cast<uint32_t*>(dw.p)+8),static_cast<float*>(dy.p)+8,2560,rows[i],stream.p);cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));std::vector<uint32_t>out(hy.size()),ww(hw.size()),xx(hx.size());cu(cudaMemcpy(out.data(),dy.p,out.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(ww.data(),dw.p,ww.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(xx.data(),dx.p,xx.size()*4,cudaMemcpyDeviceToHost));need(hw==ww&&hx==xx,"indexer BF16 native input changed");for(size_t j=0;j<8;++j)need(out[j]==guard&&out[out.size()-1-j]==guard,"indexer BF16 native output guard");std::vector<float>v(rows[i]);std::memcpy(v.data(),out.data()+8,v.size()*4);if(!pass)first[i]=v;else need(v==first[i],"indexer BF16 native repeat differs");const std::string name=i?"query":"key";save(dir/(name+(pass?"-repeat.f32le":".f32le")),v.data(),v.size()*4);}}
 literals(first);std::ofstream(dir/"report.json")<<"{\"actual_native_launcher\":true,\"immutable_canary_repeat\":true,\"qualified\":false}\n";return 0;
}catch(const std::exception&e){std::fprintf(stderr,"native indexer BF16: %s\n",e.what());return 1;}}
