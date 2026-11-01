#include "native_gdn_log1p_fixture.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
using namespace native_gdn_log1p;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void probe(const float*x,float*out){int i=threadIdx.x;float a=x[i],u=__fadd_rn(1.f,a),l=0,n=0,d=0,c=a;
 if(u!=1.f){l=logf(u);n=__fmul_rn(l,a);d=__fsub_rn(u,1.f);c=n/d;}
 out[i]=log1pf(a);out[256+i]=c;out[512+i]=u;out[768+i]=l;out[1024+i]=n;out[1280+i]=d;
}
struct Device{float*p=nullptr;~Device(){if(p)cudaFree(p);}};
int main(int argc,char**argv){try{native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","log1p native arguments");auto x=read(argv[6]);std::filesystem::path dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"log1p fresh output");std::filesystem::copy_file(argv[6],dir/"fixture.bin");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,std::stoi(argv[2])));native_hc_need(std::string(props.name).find(argv[4])!=std::string::npos,"log1p actual native device");Device in,out;cu(cudaMalloc((void**)&in.p,1024));cu(cudaMalloc((void**)&out.p,6*1024));std::vector<float>v(1536),first,after(256);for(int r=0;r<2;++r){cu(cudaMemcpy(in.p,x.data(),1024,cudaMemcpyHostToDevice));probe<<<1,256>>>(in.p,out.p);cu(cudaGetLastError());cu(cudaDeviceSynchronize());cu(cudaMemcpy(v.data(),out.p,6*1024,cudaMemcpyDeviceToHost));cu(cudaMemcpy(after.data(),in.p,1024,cudaMemcpyDeviceToHost));exact(x,after,"log1p native input changed");if(r)exact(v,first,"log1p native reset/repeat");else first=v;}
 for(size_t j=0;j<6;++j)floats((dir/(std::string(fields[j])+".f32le")).string(),std::vector<float>(v.begin()+j*256,v.begin()+(j+1)*256));std::ofstream(dir/"report.json")<<"{\"actual_cuda_log1pf\":true,\"compensated_is_diagnostic\":true,\"repeat_exact\":true,\"qualified\":false}\n";return 0;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
