#include "native_gdn_t1_fixture.hpp"
#include "strata_vk_native_gdn.hpp"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
using namespace native_gdn_t1;
using namespace strata::core::portable_detail;
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],native_gdn_t1::synthetic(unsigned(std::stoul(argv[2]))));return 0;}
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","GDN component arguments");auto f=read(argv[6]);std::filesystem::path dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"GDN fresh output");std::filesystem::copy_file(argv[6],dir/"fixture.bin");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_vk_init(std::stoi(argv[2])),ggml_backend_free);native_hc_need(bool(backend),"GDN Vulkan backend");const char*name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));native_hc_need(std::string(name).find(argv[4])!=std::string::npos,"GDN actual device");
 std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({65536,nullptr,true}),ggml_free);native_hc_need(bool(ctx),"GDN tensor metadata");std::array<ggml_tensor*,13>base{},t{};std::array<std::vector<float>,13>before,values,first;
 auto*aux=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,48+49+6144+6144+64);size_t auxoff=0;
 for(size_t i=0;i<13;++i){before[i].assign(counts[i]+16,value(0x4acaf00d));if(i<6)std::copy(f[i].begin(),f[i].end(),before[i].begin()+8);if(i==8||i==10||i==11||i==12){base[i]=ggml_view_1d(ctx.get(),aux,int64_t(before[i].size()),auxoff*4);auxoff+=before[i].size();}else base[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,int64_t(before[i].size()));t[i]=ggml_view_1d(ctx.get(),base[i],int64_t(counts[i]),32);}
 size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));native_hc_need(bound<=8u<<20,"GDN tensors exceed 8MiB bound");std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buf(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);native_hc_need(bool(buf)&&ggml_backend_buffer_get_size(buf.get())<=8u<<20,"GDN actual tensor bound");
 std::unique_ptr<StrataVkNativeGdn,decltype(&strata_vk_native_gdn_destroy)>kernel(strata_vk_native_gdn_create(backend.get(),(dir/"effective.spv").c_str()),strata_vk_native_gdn_destroy);
 for(int repeat=0;repeat<2;++repeat){for(size_t i=0;i<13;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);strata_vk_native_gdn_run(kernel.get(),t.data(),aux);for(size_t i=0;i<13;++i){std::vector<float>out(before[i].size());ggml_backend_tensor_get(base[i],out.data(),0,out.size()*4);for(size_t j=0;j<out.size();++j)if(i<6||j<8||j>=counts[i]+8)native_hc_need(bits(out[j])==bits(before[i][j]),"GDN immutable state/input/canary");values[i].assign(out.begin()+8,out.end()-8);if(repeat)exact(values[i],first[i],"GDN reset/repeat differs");else first[i]=values[i];}}
 auto ref=control(f,values[8],values[10],values[12]);exact(values[6],ref.state,"GDN own-exp next state order");exact(values[9],ref.oc,"GDN own-state output reduction");exact(values[7],ref.y,"GDN own-rsqrt/gate output association");
 // A rejected call cannot mutate input state. Candidate scratch is never committed by this API.
 float bad=INFINITY;ggml_backend_tensor_set(t[2],&bad,0,4);bool refused=false;try{strata_vk_native_gdn_run(kernel.get(),t.data(),aux);}catch(...){refused=true;}native_hc_need(refused,"GDN invalid input accepted");std::vector<float>state(counts[0]);ggml_backend_tensor_get(t[0],state.data(),0,state.size()*4);exact(state,f[0],"GDN refusal committed state");
 const char*names[]={"next-state","y","decay-exp","oc","rs","sigmoid-exp","sigmoid"};for(size_t i=6;i<13;++i)floats((dir/(std::string(names[i-6])+".f32le")).string(),values[i]);floats((dir/"norm-arg.f32le").string(),ref.arg);
 std::ofstream report(dir/"report.json");report<<"{\"actual_vulkan\":true,\"immutable_inputs_canaries\":true,\"reset_repeat_exact\":true,\"failure_noncommit\":true,\"exact_own_primitive_state_and_output\":true,\"qualified\":false,\"tensor_bytes\":"<<ggml_backend_buffer_get_size(buf.get())<<"}\n";native_hc_need(bool(report),"GDN report");return 0;
 }catch(const std::exception&e){fprintf(stderr,"native GDN T1: %s\n",e.what());return 1;}}
