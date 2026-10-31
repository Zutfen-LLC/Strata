// Actual selected fused-GR launcher: independent of the Vulkan/host arithmetic implementations.
#include "native_hc_fixture.hpp"
#include "strata/kernels/fused_gr.hpp"
#include <cuda_runtime.h>
using namespace native_hc_fixture;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{void* p=nullptr;~Device(){if(p)cudaFree(p);}void alloc(size_t n){cu(cudaMalloc(&p,n));cu(cudaMemset(p,0,n));}};
int main(int argc,char** argv){try{
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","native HC oracle arguments");
 native_hc_environment();
 auto f=read_fixture(argv[6]);native_hc_need(strata::kernels::fused_gr_supported(2560,4,320),"native HC oracle: compiled geometry mismatch");std::string dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"native HC oracle fresh output");std::filesystem::copy_file(argv[6],dir+"/fixture.bin");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,std::stoi(argv[2])));native_hc_need(std::string(props.name).find(argv[4])!=std::string::npos,"native HC oracle device");
 Device x,w,wi,zeros,xn,rs,inject,lo,mixed;x.alloc(f.x.size()*4);w.alloc(f.w.size()*4);wi.alloc(f.wi.size()*2);zeros.alloc(10240*320*2);xn.alloc(f.x.size()*4);rs.alloc(f.t*16);inject.alloc(f.t*16);lo.alloc(f.t*320*4);mixed.alloc(f.t*2560*4);cu(cudaMemcpy(x.p,f.x.data(),f.x.size()*4,cudaMemcpyHostToDevice));cu(cudaMemcpy(w.p,f.w.data(),f.w.size()*4,cudaMemcpyHostToDevice));cu(cudaMemcpy(wi.p,f.wi.data(),f.wi.size()*2,cudaMemcpyHostToDevice));
 strata::kernels::fused_gr_check();std::vector<strata::kernels::FusedGrArgs> args(f.t);
 for(uint32_t t=0;t<f.t;++t){auto& a=args[t];a.R=static_cast<float*>(x.p)+t*10240;a.R_out=static_cast<float*>(x.p)+t*10240;a.apply=false;a.w_norm=(float*)w.p;a.w_down=a.w_up=(uint16_t*)zeros.p;a.w_inject=(uint16_t*)wi.p;a.lo=(float*)lo.p+t*320;a.rs=(float*)rs.p+t*4;a.inject_out=(float*)inject.p+t*4;a.mixed=(float*)mixed.p+t*2560;}
 const auto variant=strata::kernels::fused_gr_variant();native_hc_need(!strata::kernels::fused_gr_read_multi(args.data(),f.t,(float*)xn.p,nullptr),"native HC oracle unexpectedly quantized mixed");cu(cudaGetLastError());cu(cudaDeviceSynchronize());
 for(auto pair:{std::pair<Device*,std::string>{&xn,"normalized.f32le"},{&rs,"rs.f32le"},{&inject,"inject.f32le"}}){std::vector<float> v(pair.first==&xn?f.x.size():f.t*4);cu(cudaMemcpy(v.data(),pair.first->p,v.size()*4,cudaMemcpyDeviceToHost));floats(dir+"/"+pair.second,v);}
 std::vector<float> check(f.x.size());cu(cudaMemcpy(check.data(),x.p,check.size()*4,cudaMemcpyDeviceToHost));native_hc_need(std::memcmp(check.data(),f.x.data(),check.size()*4)==0,"native HC oracle changed no-pending residual");std::ofstream report(dir+"/report.json");report<<"{\"actual_fused_gr_launcher\":true,\"variant\":"<<variant<<",\"tokens\":"<<f.t<<",\"qualified\":false}\n";printf("native_hc_source_oracle=true variant=%d device=%s\n",variant,props.name);return 0;
}catch(const std::exception& e){fprintf(stderr,"native HC source oracle: %s\n",e.what());return 1;}}
