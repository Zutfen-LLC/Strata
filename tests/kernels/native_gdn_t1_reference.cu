#include "native_gdn_t1_fixture.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
using namespace native_gdn_t1;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{float*p=nullptr;~Device(){if(p)cudaFree(p);}void alloc(size_t n){cu(cudaMalloc((void**)&p,n*4));}void set(const std::vector<float>&v){cu(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));}std::vector<float>get(size_t n){std::vector<float>v(n);cu(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}};
// Diagnostic intrinsic controls, not replacement numerical execution. The actual
// reference output/state always come from the unchanged gdn_step_norm_multi launcher.
__global__ void primitives(const float*g,const float*z,const float*a,float*eg,float*rs,float*se,float*sig){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<48){eg[i]=__expf(g[i]);rs[i]=rsqrtf(a[i]);}if(i==0)rs[48]=rsqrtf(128.f);if(i<6144){float e=__expf(-z[i]);se[i]=e;sig[i]=1.f/(1.f+e);}}
int main(int argc,char**argv){try{
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","GDN native arguments");auto f=read(argv[6]);std::filesystem::path dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"GDN native fresh output");std::filesystem::copy_file(argv[6],dir/"fixture.bin");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,std::stoi(argv[2])));native_hc_need(std::string(props.name).find(argv[4])!=std::string::npos,"GDN native actual device");
 Device inputs[6],y,eg,rs,se,sig,args,keep;for(size_t i=0;i<6;++i){inputs[i].alloc(counts[i]);inputs[i].set(f[i]);}y.alloc(6144);eg.alloc(48);rs.alloc(49);se.alloc(6144);sig.alloc(6144);args.alloc(48);args.set(std::vector<float>(48,1));keep.alloc(1);int one=1;cu(cudaMemcpy(keep.p,&one,4,cudaMemcpyHostToDevice));
 std::vector<float>state,output;
 for(int repeat=0;repeat<2;++repeat){inputs[0].set(f[0]);strata::kernels::gdn_step_norm_multi(inputs[0].p,inputs[1].p,10240,inputs[2].p,inputs[3].p,inputs[4].p,inputs[5].p,1e-6f,y.p,16,48,1,reinterpret_cast<int*>(keep.p),nullptr,0,nullptr);cu(cudaGetLastError());cu(cudaDeviceSynchronize());auto s=inputs[0].get(786432),o=y.get(6144);if(repeat){exact(state,s,"GDN native reset state");exact(output,o,"GDN native reset output");}else{state=s;output=o;}}
 for(size_t i=1;i<6;++i)exact(inputs[i].get(counts[i]),f[i],"GDN native input changed");
 primitives<<<24,256>>>(inputs[2].p,inputs[4].p,args.p,eg.p,rs.p,se.p,sig.p);cu(cudaGetLastError());cu(cudaDeviceSynchronize());auto ge=eg.get(48),r=rs.get(49),ss=sig.get(6144);auto control1=control(f,ge,r,ss);args.set(control1.arg);
 primitives<<<24,256>>>(inputs[2].p,inputs[4].p,args.p,eg.p,rs.p,se.p,sig.p);cu(cudaGetLastError());cu(cudaDeviceSynchronize());r=rs.get(49);auto ref=control(f,ge,r,sig.get(6144));
 // Persist actual outputs even if a compiler contraction assumption is rejected.
 floats((dir/"next-state.f32le").string(),state);floats((dir/"y.f32le").string(),output);floats((dir/"decay-exp.f32le").string(),ge);floats((dir/"oc.f32le").string(),ref.oc);floats((dir/"norm-arg.f32le").string(),ref.arg);floats((dir/"rs.f32le").string(),r);floats((dir/"sigmoid-exp.f32le").string(),se.get(6144));floats((dir/"sigmoid.f32le").string(),sig.get(6144));
 exact(state,ref.state,"GDN native delta/FMA control differs: inspect compiled lowering, do not relax tolerance");exact(output,ref.y,"GDN native normalization/gate control differs: inspect lowering");
 std::ofstream report(dir/"report.json");report<<"{\"actual_native_launcher\":true,\"T\":1,\"keep\":1,\"qfuse\":false,\"reset_repeat_exact\":true,\"exact_own_primitive_state_and_output\":true,\"qualified\":false}\n";native_hc_need(bool(report),"GDN native report");return 0;
 }catch(const std::exception&e){fprintf(stderr,"native GDN T1 reference: %s\n",e.what());return 1;}}
