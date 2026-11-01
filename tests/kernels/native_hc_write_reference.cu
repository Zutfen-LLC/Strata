// Exercise the actual selected native materialization launcher, not fused apply-read.
#include "native_hc_write_fixture.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include <cuda_runtime.h>
using namespace native_hc_write_fixture;
bool exact(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*4)==0;}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{float* p=nullptr;~Device(){if(p)cudaFree(p);}void alloc(size_t n){cu(cudaMalloc(reinterpret_cast<void**>(&p),n*4));}void upload(const std::vector<float>& a){cu(cudaMemcpy(p,a.data(),a.size()*4,cudaMemcpyHostToDevice));}std::vector<float> read(size_t n){std::vector<float>a(n);cu(cudaMemcpy(a.data(),p,n*4,cudaMemcpyDeviceToHost));return a;}};
int main(int argc,char** argv){try{
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","HC write native arguments");native_hc_environment();auto f=read(argv[6]);std::string dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"HC write native: fresh output");std::filesystem::copy_file(argv[6],dir+"/fixture.bin");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,std::stoi(argv[2])));native_hc_need(std::string(props.name).find(argv[4])!=std::string::npos,"HC write native: device identity");
 Device r,b,i,out,zero,one,gateout;r.alloc(10240);b.alloc(2560);i.alloc(4);out.alloc(10240);zero.alloc(10240);one.alloc(2560);gateout.alloc(10240);
 r.upload(f.residual);b.upload(f.block);i.upload(f.inject);zero.upload(std::vector<float>(10240,0.f));one.upload(std::vector<float>(2560,1.f));std::vector<float> first,gate;
 for(int repeat=0;repeat<2;++repeat){
  strata::kernels::native_gr_post(r.p,b.p,i.p,out.p,2560,4,nullptr);
  // Independent identical-injection invocation exposes actual weight exactly:
  // FMA(1,weight,+0) is weight in this strictly positive normal domain.
  strata::kernels::native_gr_post(zero.p,one.p,i.p,gateout.p,2560,4,nullptr);
  cu(cudaGetLastError());cu(cudaDeviceSynchronize());auto a=out.read(10240),g=gateout.read(10240);for(float x:a)native_hc_normal(x);for(float x:g)native_hc_normal(x);for(size_t c=0;c<4;++c)for(size_t d=0;d<2560;++d)native_hc_need(bits(g[c*2560+d])==bits(g[c*2560]),"HC write native: gate uniformity");
  if(repeat)native_hc_need(exact(a,first)&&exact(g,gate),"HC write native: reset/repeat");else{first=a;gate=g;}
  native_hc_need(exact(r.read(10240),f.residual)&&exact(b.read(2560),f.block)&&exact(i.read(4),f.inject),"HC write native: producer mutation");
 }
 std::vector<float> weights(4);for(size_t c=0;c<4;++c)weights[c]=gate[c*2560];floats(dir+"/residual.f32le",first);floats(dir+"/gate.f32le",weights);
 std::ofstream o(dir+"/report.json");o<<"{\"actual_native_gr_post\":true,\"tokens\":1,\"gate_from_identical_launcher_unit_block\":true,\"producer_immutable\":true,\"reset_repeat_exact\":true,\"qualified\":false,\"model_integration\":false}\n";native_hc_need(bool(o),"HC write native: report");return 0;
}catch(const std::exception& e){fprintf(stderr,"HC write native: %s\n",e.what());return 1;}}
