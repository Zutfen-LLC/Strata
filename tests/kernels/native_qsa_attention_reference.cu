#include "native_qsa_attention_fixture.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/native_qsa.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
using namespace native_qsa_attn;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{uint32_t*p=nullptr;size_t n=0;~Device(){if(p)cudaFree(p);}void alloc(const std::vector<uint32_t>&v){n=v.size();cu(cudaMalloc((void**)&p,n*4));set(v);}void set(const std::vector<uint32_t>&v){native_hc_need(v.size()==n,"native QSA span");cu(cudaMemcpy(p,v.data(),n*4,cudaMemcpyHostToDevice));}std::vector<uint32_t>get(){std::vector<uint32_t>v(n);cu(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}};
struct Stream{cudaStream_t p=nullptr;Stream(){cu(cudaStreamCreate(&p));}~Stream(){if(p)cudaStreamDestroy(p);}};
// Primitive controls only: original attention/gate kernels are not edited.
__global__ void primitive_exp(const float*x,float*y,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=__expf(x[i]);}
std::vector<float> exponent(const std::vector<float>&x,cudaStream_t stream){std::vector<uint32_t>w(x.size());std::memcpy(w.data(),x.data(),w.size()*4);Device a,b;a.alloc(w);b.alloc(std::vector<uint32_t>(w.size(),canary));primitive_exp<<<unsigned((w.size()+255)/256),256,0,stream>>>((float*)a.p,(float*)b.p,w.size());cu(cudaStreamSynchronize(stream));auto bits=b.get();std::vector<float>y(bits.size());std::memcpy(y.data(),bits.data(),bits.size()*4);return y;}
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--contract"){contract(read(argv[2]),argv[3]);return 0;}
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","native QSA attention arguments");native_qsa_attention_environment();auto f=read(argv[6]);std::filesystem::path dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"fresh native QSA attention output");std::filesystem::copy_file(argv[6],dir/"fixture.bin");cu(cudaSetDevice(std::stoi(argv[2])));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,std::stoi(argv[2])));native_hc_need(std::string(prop.name).find(argv[4])!=std::string::npos&&prop.major==8,"default Ampere FP16 chunk/merge oracle required");
 auto n=counts(f.cells);std::array<Device,13>d;std::array<std::vector<uint32_t>,13>initial,first;for(size_t i=0;i<13;++i){initial[i].assign(n[i]+16,canary);if(i<7)std::copy(f.in[i].begin(),f.in[i].end(),initial[i].begin()+8);d[i].alloc(initial[i]);}
 Stream stream;auto shapes=strata::kernels::qsa_real_shapes();native_hc_need(shapes.page_size==4&&strata::kernels::qsa_decode_attn_scratch_floats(2051,shapes)==204400,"unchanged native attention geometry/scratch");strata::kernels::QsaAttnPools pools{};pools.k_pool=(uint16_t*)(d[0].p+8);pools.v_pool=(uint16_t*)(d[1].p+8);pools.page_table=(int32_t*)(d[3].p+8);
 Device ones,sig;ones.alloc(std::vector<uint32_t>(6144,0x3f800000));sig.alloc(std::vector<uint32_t>(6144,canary));
 for(unsigned repeat=0;repeat<2;++repeat){for(size_t i=0;i<13;++i)d[i].set(initial[i]);strata::kernels::qsa_decode_attn_batch((float*)(d[2].p+8),pools,(int32_t*)(d[4].p+8),(int32_t*)(d[5].p+8),2051,shapes,(float*)(d[7].p+8),(float*)(d[10].p+8),1,stream.p);strata::kernels::native_qsa_gate_apply((float*)(d[10].p+8),(float*)(d[6].p+8),(float*)(d[12].p+8),24,256,stream.p);strata::kernels::native_qsa_gate_apply((float*)ones.p,(float*)(d[6].p+8),(float*)sig.p,24,256,stream.p);cu(cudaStreamSynchronize(stream.p));
  std::array<std::vector<uint32_t>,13>actual;for(size_t i=0;i<13;++i){auto v=d[i].get();for(size_t j=0;j<8;++j)native_hc_need(v[j]==canary&&v[n[i]+8+j]==canary,"native attention canary changed");if(i<7)native_hc_need(v==initial[i],"native attention immutable input changed");actual[i].assign(v.begin()+8,v.end()-8);}
  if(repeat==0){for(size_t i:{7u,10u,12u})native_qsa_ss::put(dir/names[i-7],actual[i]);native_qsa_ss::put(dir/"control-sigmoid.f32le",sig.get());}
  auto host=control(f);auto ec=exponent(host.chunk_args,stream.p);auto partial=control(f,ec);native_hc_need(actual[7]==partial.out[0],"actual native partials differ from independent source-order/own-exp control");auto em=exponent(partial.merge_args,stream.p),eg=exponent(partial.gate_args,stream.p);auto sw=sig.get();std::vector<float>sv(sw.size());std::memcpy(sv.data(),sw.data(),sw.size()*4);auto ref=control(f,ec,em,eg,sv);native_hc_need(actual[10]==ref.out[3]&&actual[12]==ref.out[5],"native RNE merge or own-sigmoid gate order differs");
  // Native score/exp diagnostics are independently derived controls. Actual
  // original-kernel partials, attention and gated output remain distinct.
  actual[8]=ref.out[1];actual[9]=ref.out[2];actual[11]=ref.out[4];for(size_t i=7;i<13;++i){if(repeat)native_hc_need(actual[i]==first[i],"native reset/repeat differs");first[i]=actual[i];}
 }
 for(size_t i:{8u,9u,11u})native_qsa_ss::put(dir/names[i-7],first[i]);check(f,dir);std::ofstream(dir/"report.json")<<"{\"actual_native_launchers\":true,\"native_logits_derived_control\":true,\"source_order_own_primitive_exact\":true,\"immutable_inputs_canaries\":true,\"reset_repeat_exact\":true,\"attention_calls\":2,\"gate_calls\":2,\"control_only_gate_calls\":2,\"qualified\":false}\n";return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
