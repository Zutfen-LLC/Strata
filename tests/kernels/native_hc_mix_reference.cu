#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>
#include "native_hc_mix_fixture.hpp"
#include "strata/kernels/fused_gr.hpp"
#include <cuda_runtime.h>
using namespace native_hc_mix_fixture;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{void* p=nullptr;~Device(){if(p)cudaFree(p);}void alloc(size_t n){cu(cudaMalloc(&p,n));cu(cudaMemset(p,0,n));}};
int sigmoid_probe(char** argv){
 std::string dir=argv[8];native_hc_need(std::filesystem::file_size(argv[6])==40960,"sigmoid probe: fixed F32 input");std::vector<float> pre(10240),scaled(10240);std::ifstream file(argv[6],std::ios::binary);file.read((char*)pre.data(),40960);native_hc_need(bool(file),"sigmoid probe: input read");
 for(size_t i=0;i<pre.size();++i){native_hc_normal(pre[i]);scaled[i]=pre[i]*4.f;native_hc_normal(scaled[i]);native_hc_need(bits(scaled[i]/4.f)==bits(pre[i]),"sigmoid probe: exact original power-two scaling");}
 native_hc_need(std::filesystem::create_directory(dir),"sigmoid probe: fresh output");std::filesystem::copy_file(argv[6],dir+"/pre.f32le");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,std::stoi(argv[2])));native_hc_need(std::string(props.name).find(argv[4])!=std::string::npos,"sigmoid probe: native device");
 Device x,e,g;x.alloc(41024);e.alloc(41024);g.alloc(41024);std::vector<uint32_t> guarded(10256,0xa5a5a5a5u);std::memcpy(guarded.data()+8,scaled.data(),40960);cu(cudaMemcpy(x.p,guarded.data(),41024,cudaMemcpyHostToDevice));std::vector<float> firste,firstg;
 for(int repeat=0;repeat<2;++repeat){
  std::vector<uint32_t> canary(10256,0xa5a5a5a5u);cu(cudaMemcpy(e.p,canary.data(),41024,cudaMemcpyHostToDevice));cu(cudaMemcpy(g.p,canary.data(),41024,cudaMemcpyHostToDevice));
  for(size_t i=0;i<10240;i+=4)strata::kernels::fused_gr_pending_gate_probe((float*)x.p+8+i,(float*)g.p+8+i,(float*)e.p+8+i,nullptr);
  cu(cudaGetLastError());cu(cudaDeviceSynchronize());std::vector<float> ev(10240),gv(10240);
  for(auto*d:{&x,&e,&g}){std::vector<uint32_t> words(10256);cu(cudaMemcpy(words.data(),d->p,41024,cudaMemcpyDeviceToHost));for(size_t j=0;j<words.size();++j)if(d==&x||j<8||j>=10248)native_hc_need(words[j]==(d==&x?guarded[j]:0xa5a5a5a5u),"sigmoid probe: canary/input mutation");if(d!=&x)std::memcpy((d==&e?ev:gv).data(),words.data()+8,40960);}
  for(float&v:gv){native_hc_normal(v);float half=v*.5f;native_hc_need(bits(half*2.f)==bits(v),"sigmoid probe: exact undo of native gate scale");v=half;}
  if(repeat)native_hc_need(std::memcmp(ev.data(),firste.data(),40960)==0&&std::memcmp(gv.data(),firstg.data(),40960)==0,"sigmoid probe: repeat differs");else{firste=ev;firstg=gv;}
 }
 floats(dir+"/exp.f32le",firste);floats(dir+"/gate.f32le",firstg);std::ofstream report(dir+"/report.json");report<<"{\"actual_native_sigmoid_function\":true,\"exact_power_two_input_output_scaling\":true,\"canaries_input_repeat\":true,\"qualified\":false}\n";native_hc_need(bool(report),"sigmoid probe: report write");return 0;
}
int main(int argc,char** argv){try{
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&(std::string(argv[5])=="--fixture"||std::string(argv[5])=="--sigmoid-pre")&&std::string(argv[7])=="--output","HC mix native arguments");native_hc_environment();if(std::string(argv[5])=="--sigmoid-pre")return sigmoid_probe(argv);auto f=read(argv[6]);native_hc_need(strata::kernels::fused_gr_supported(2560,4,320),"HC mix native shape");std::string dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"HC mix native: fresh output");std::filesystem::copy_file(argv[6],dir+"/fixture.bin");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,std::stoi(argv[2])));native_hc_need(std::string(props.name).find(argv[4])!=std::string::npos,"HC mix native device");
 Device x,down,up,one,rs,low,mixed;x.alloc(40960);down.alloc(6553600);up.alloc(6553600);one.alloc(40960);rs.alloc(16);low.alloc(1280);mixed.alloc(10240);cu(cudaMemcpy(x.p,f.x.data(),40960,cudaMemcpyHostToDevice));cu(cudaMemcpy(down.p,f.down.data(),6553600,cudaMemcpyHostToDevice));cu(cudaMemcpy(up.p,f.up.data(),6553600,cudaMemcpyHostToDevice));std::vector<float> ones(10240,1.f);cu(cudaMemcpy(one.p,ones.data(),40960,cudaMemcpyHostToDevice));cu(cudaMemcpy(rs.p,ones.data(),16,cudaMemcpyHostToDevice));strata::kernels::fused_gr_check();auto variant=strata::kernels::fused_gr_variant();native_hc_need(variant==3,"HC mix native: selected staged variant3 required");
 std::vector<float> firstlow,firstmixed;
 // Compare actual existing plain/staged kernels on identical normalized input,
 // then repeat staged; no new numerical implementation replaces those kernels.
 for(int which:{1,3,3}){strata::kernels::fused_gr_normalized_probe((float*)x.p,(uint16_t*)down.p,(uint16_t*)up.p,(float*)one.p,(float*)rs.p,(float*)low.p,(float*)mixed.p,which,nullptr);cu(cudaGetLastError());cu(cudaDeviceSynchronize());std::vector<float> l(320),m(2560);cu(cudaMemcpy(l.data(),low.p,1280,cudaMemcpyDeviceToHost));cu(cudaMemcpy(m.data(),mixed.p,10240,cudaMemcpyDeviceToHost));for(float v:l)native_hc_normal(v);for(float v:m)native_hc_normal(v);if(firstlow.empty()){firstlow=l;firstmixed=m;}else native_hc_need(std::memcmp(l.data(),firstlow.data(),1280)==0&&std::memcmp(m.data(),firstmixed.data(),10240)==0,"HC mix native: plain/staged/repeat differs");}
 std::vector<float> check(10240);cu(cudaMemcpy(check.data(),x.p,40960,cudaMemcpyDeviceToHost));native_hc_need(std::memcmp(check.data(),f.x.data(),40960)==0,"HC mix native: input changed");floats(dir+"/low.f32le",firstlow);floats(dir+"/mixed.f32le",firstmixed);std::ofstream report(dir+"/report.json");report<<"{\"existing_down_up_kernels\":true,\"variant\":3,\"tokens\":1,\"plain_staged_repeat_exact\":true,\"qualified\":false}\n";native_hc_need(bool(report),"HC mix native report");return 0;
}catch(const std::exception& e){fprintf(stderr,"HC mix native: %s\n",e.what());return 1;}}
