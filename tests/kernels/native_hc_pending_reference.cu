// Actual selected native fused read with apply=true, not a rewritten arithmetic oracle.
#include "native_hc_pending_fixture.hpp"
#include "strata/kernels/fused_gr.hpp"
#include <cuda_runtime.h>
using namespace native_hc_pending_fixture;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{void* p=nullptr;~Device(){if(p)cudaFree(p);}void alloc(size_t n){cu(cudaMalloc(&p,n));cu(cudaMemset(p,0,n));}void upload(const void* src,size_t n){cu(cudaMemcpy(p,src,n,cudaMemcpyHostToDevice));}};
int main(int argc,char** argv){try{
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","HC pending native arguments");native_hc_environment();auto f=read(argv[6]);std::string dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"HC pending native: fresh output");std::filesystem::copy_file(argv[6],dir+"/fixture.bin");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,std::stoi(argv[2])));native_hc_need(std::string(props.name).find(argv[4])!=std::string::npos,"HC pending native: device identity");native_hc_need(strata::kernels::fused_gr_supported(2560,4,320),"HC pending native: geometry");
 Device x,bo,previous,w,wi,down,up,xn,rs,inject,low,mixed,gate,exponential;
 x.alloc(40960);bo.alloc(10240);previous.alloc(16);w.alloc(40960);wi.alloc(81920);down.alloc(6553600);up.alloc(6553600);xn.alloc(40960);rs.alloc(16);inject.alloc(16);low.alloc(1280);mixed.alloc(10240);gate.alloc(16);exponential.alloc(16);
 bo.upload(f.bo.data(),10240);previous.upload(f.inj.data(),16);w.upload(f.norm.w.data(),40960);wi.upload(f.norm.wi.data(),81920);down.upload(f.down.data(),6553600);up.upload(f.up.data(),6553600);
 strata::kernels::fused_gr_check();const auto variant=strata::kernels::fused_gr_variant();native_hc_need(variant==3,"HC pending native: selected staged variant3 required");
 strata::kernels::FusedGrArgs a{};a.R=(float*)x.p;a.R_out=(float*)x.p;a.apply=true;a.bo_prev=(float*)bo.p;a.inj_prev=(float*)previous.p;a.w_norm=(float*)w.p;a.w_down=(uint16_t*)down.p;a.w_up=(uint16_t*)up.p;a.w_inject=(uint16_t*)wi.p;a.lo=(float*)low.p;a.rs=(float*)rs.p;a.inject_out=(float*)inject.p;a.mixed=(float*)mixed.p;
 const std::array<Device*,8> outputs{&x,&gate,&exponential,&xn,&rs,&inject,&low,&mixed};const std::array<size_t,8> count{10240,4,4,10240,4,4,320,2560};const char* names[]={"residual","pending-gate","pending-exp","normalized","rs","inject","low","mixed"};std::array<std::vector<float>,8> first;
 for(int repeat=0;repeat<2;++repeat){x.upload(f.norm.x.data(),40960);
  strata::kernels::fused_gr_pending_gate_probe((float*)previous.p,(float*)gate.p,(float*)exponential.p,nullptr);
  native_hc_need(!strata::kernels::fused_gr_read_multi(&a,1,(float*)xn.p,nullptr),"HC pending native: unexpected quantized path");cu(cudaGetLastError());cu(cudaDeviceSynchronize());
  for(size_t i=0;i<outputs.size();++i){std::vector<float> v(count[i]);cu(cudaMemcpy(v.data(),outputs[i]->p,count[i]*4,cudaMemcpyDeviceToHost));for(float x:v)native_hc_normal(x);if(repeat)native_hc_need(std::memcmp(v.data(),first[i].data(),v.size()*4)==0,"HC pending native: reset/repeat differs");else first[i]=std::move(v);}
 }
 // Neither pending injection nor block output is overwritten by a later read.
 std::vector<float> check(2560);cu(cudaMemcpy(check.data(),bo.p,10240,cudaMemcpyDeviceToHost));native_hc_need(std::memcmp(check.data(),f.bo.data(),10240)==0,"HC pending native: block output changed");cu(cudaMemcpy(check.data(),previous.p,16,cudaMemcpyDeviceToHost));native_hc_need(std::memcmp(check.data(),f.inj.data(),16)==0,"HC pending native: previous injection changed");
 for(size_t i=0;i<outputs.size();++i)floats(dir+"/"+names[i]+".f32le",first[i]);std::ofstream report(dir+"/report.json");report<<"{\"actual_fused_gr_launcher\":true,\"apply\":true,\"variant\":3,\"tokens\":1,\"reset_repeat_exact\":true,\"prior_injection_block_immutable\":true,\"qualified\":false,\"full_hc_parity\":false}\n";native_hc_need(bool(report),"HC pending native: report");return 0;
}catch(const std::exception& e){fprintf(stderr,"HC pending native: %s\n",e.what());return 1;}}
