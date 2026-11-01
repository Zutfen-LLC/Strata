#include "native_gdn_conv_fixture.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
using namespace native_gdn_conv;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{float*p=nullptr;~Device(){if(p)cudaFree(p);}void alloc(size_t n){cu(cudaMalloc((void**)&p,n*4));}void set(const std::vector<float>&v){cu(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));}std::vector<float>get(size_t n){std::vector<float>v(n);cu(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}};
// Explicit contraction candidate/primitive controls, not replacement execution.
// h/history below always come from the unchanged production launcher.
__global__ void primitives(const float*hist,const float*x,const float*w,const float*args,float*pre,float*ex,float*silu,float*rs){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<10240){float second=hist[c*3+1]*w[c*4+1];float v=fmaf(hist[c*3],w[c*4],second);v=fmaf(hist[c*3+2],w[c*4+2],v);v=fmaf(x[c],w[c*4+3],v);pre[c]=v;float e=__expf(-v);ex[c]=e;silu[c]=v/(1.f+e);}if(c<32)rs[c]=rsqrtf(args[c]);}
int main(int argc,char**argv){try{
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","GDN conv native arguments");auto f=read(argv[6]);std::filesystem::path dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"GDN conv fresh output");std::filesystem::copy_file(argv[6],dir/"fixture.bin");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,std::stoi(argv[2])));native_hc_need(std::string(props.name).find(argv[4])!=std::string::npos,"GDN conv actual native device");
 Device in[3],out,pre,ex,silu,args,rs;for(size_t i=0;i<3;++i){in[i].alloc(counts[i]);in[i].set(f[i]);}out.alloc(10240);pre.alloc(10240);ex.alloc(10240);silu.alloc(10240);args.alloc(32);args.set(std::vector<float>(32,1));rs.alloc(32);
 std::vector<float>actual,history;
 for(bool commit:{false,true})for(int repeat=0;repeat<2;++repeat){in[0].set(f[0]);strata::kernels::gdn_conv_l2_multi(in[0].p,in[1].p,in[2].p,out.p,10240,32,1e-6f,1,nullptr,0,commit);cu(cudaGetLastError());cu(cudaDeviceSynchronize());auto h=out.get(10240),hist=in[0].get(30720);if(!commit)exact(hist,f[0],"native verify changed history");if(actual.empty())actual=h;else exact(h,actual,"native verify/commit/reset output differs");if(commit){if(repeat)exact(hist,history,"native reset history differs");else history=hist;}}
 for(size_t i=1;i<3;++i)exact(in[i].get(counts[i]),f[i],"native conv input changed");in[0].set(f[0]);
 primitives<<<40,256>>>(in[0].p,in[1].p,in[2].p,args.p,pre.p,ex.p,silu.p,rs.p);cu(cudaGetLastError());cu(cudaDeviceSynchronize());auto actual_silu=silu.get(10240);auto ref=control(f,ex.get(10240),rs.get(32),true,&actual_silu);args.set(ref.arg);
 primitives<<<40,256>>>(in[0].p,in[1].p,in[2].p,args.p,pre.p,ex.p,silu.p,rs.p);cu(cudaGetLastError());cu(cudaDeviceSynchronize());actual_silu=silu.get(10240);ref=control(f,ex.get(10240),rs.get(32),true,&actual_silu);
 // Retain actual production output even if the contraction candidate is wrong.
 floats((dir/"next-history.f32le").string(),history);floats((dir/"h.f32le").string(),actual);floats((dir/"pre.f32le").string(),pre.get(10240));floats((dir/"exp.f32le").string(),ex.get(10240));floats((dir/"silu.f32le").string(),ref.silu);floats((dir/"rs.f32le").string(),rs.get(32));floats((dir/"norm-arg.f32le").string(),ref.arg);
 exact(pre.get(10240),ref.pre,"native diagnostic contraction differs from host source control");exact(actual,ref.h,"native launcher differs from convolution contraction/reduction candidate; inspect lowering, do not relax tolerance");exact(history,ref.history,"native commit history association");
 std::ofstream report(dir/"report.json");report<<"{\"actual_native_launcher\":true,\"T\":1,\"commit_and_verify\":true,\"verify_history_unchanged\":true,\"reset_repeat_exact\":true,\"exact_own_primitive_state_and_output\":true,\"pre_is_source_control_candidate\":true,\"qualified\":false}\n";native_hc_need(bool(report),"native conv report");return 0;
}catch(const std::exception&e){fprintf(stderr,"native GDN conv reference: %s\n",e.what());return 1;}}
