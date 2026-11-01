#include "native_gdn_conv_fixture.hpp"
#include "strata_vk_native_gdn_conv.hpp"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
using namespace native_gdn_conv;
using namespace strata::core::portable_detail;
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],native_gdn_conv::synthetic(unsigned(std::stoul(argv[2]))));return 0;}
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","GDN conv arguments");auto f=read(argv[6]);std::filesystem::path dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"GDN conv fresh output");std::filesystem::copy_file(argv[6],dir/"fixture.bin");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_vk_init(std::stoi(argv[2])),ggml_backend_free);native_hc_need(bool(backend),"GDN conv Vulkan required");const char*name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));native_hc_need(std::string(name).find(argv[4])!=std::string::npos,"GDN conv actual Vulkan device");
 std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({32768,nullptr,true}),ggml_free);native_hc_need(bool(ctx),"GDN conv metadata");std::array<ggml_tensor*,10>base{},t{};std::array<std::vector<float>,10>before,values,first;
 for(size_t i=0;i<10;++i){before[i].assign(counts[i]+16,value(0x4acaf00d));if(i<3)std::copy(f[i].begin(),f[i].end(),before[i].begin()+8);base[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,before[i].size());t[i]=ggml_view_1d(ctx.get(),base[i],counts[i],32);}
 size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));native_hc_need(bound<=2u<<20,"GDN conv aligned 2MiB admission");std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buf(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);native_hc_need(bool(buf)&&ggml_backend_buffer_get_size(buf.get())<=2u<<20,"GDN conv actual tensor budget");
 std::unique_ptr<StrataVkNativeGdnConv,decltype(&strata_vk_native_gdn_conv_destroy)>kernel(strata_vk_native_gdn_conv_create(backend.get(),(dir/"effective.spv").c_str()),strata_vk_native_gdn_conv_destroy);
 for(size_t i=0;i<10;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);auto*saved=t[3];t[3]=t[0];bool refused=false;try{strata_vk_native_gdn_conv_run(kernel.get(),t.data(),true);}catch(...){refused=true;}t[3]=saved;native_hc_need(refused,"GDN conv live/candidate alias accepted");
 for(bool commit:{false,true})for(int repeat=0;repeat<2;++repeat){for(size_t i=0;i<10;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);strata_vk_native_gdn_conv_run(kernel.get(),t.data(),commit);
  for(size_t i=0;i<10;++i){std::vector<float>v(before[i].size());ggml_backend_tensor_get(base[i],v.data(),0,v.size()*4);for(size_t j=0;j<v.size();++j)if(i<3||j<8||j>=counts[i]+8)native_hc_need(bits(v[j])==bits(before[i][j]),"GDN conv input/canary changed");values[i].assign(v.begin()+8,v.end()-8);if(repeat)exact(values[i],first[i],"GDN conv repeat differs");else first[i]=values[i];}
  auto ref=control(f,values[6],values[8],commit);exact(values[3],ref.history,"GDN conv candidate history differs");exact(values[5],ref.pre,"GDN conv FMA association");exact(values[7],ref.silu,"GDN conv original own-exp SiLU");exact(values[9],ref.arg,"GDN conv XOR reduction/epsilon");exact(values[4],ref.h,"GDN conv own-rsqrt output");
 }
 float bad=INFINITY;ggml_backend_tensor_set(t[2],&bad,0,4);refused=false;try{strata_vk_native_gdn_conv_run(kernel.get(),t.data(),true);}catch(...){refused=true;}native_hc_need(refused,"GDN conv invalid weights accepted");std::vector<float>unchanged(30720);ggml_backend_tensor_get(t[0],unchanged.data(),0,unchanged.size()*4);exact(unchanged,f[0],"GDN conv failure committed live history");
 ggml_backend_tensor_set(t[2],f[2].data(),0,f[2].size()*4);strata_vk_native_gdn_conv_run(kernel.get(),t.data(),true);std::vector<float>retry(10240);ggml_backend_tensor_get(t[4],retry.data(),0,retry.size()*4);exact(retry,values[4],"GDN conv retry differs");
 const char*names[]={"next-history","h","pre","exp","silu","rs","norm-arg"};for(size_t i=3;i<10;++i)floats((dir/(std::string(names[i-3])+".f32le")).string(),values[i]);
 std::ofstream report(dir/"report.json");report<<"{\"actual_vulkan\":true,\"immutable_inputs_canaries\":true,\"commit_and_verify\":true,\"verify_history_unchanged\":true,\"reset_repeat_exact\":true,\"failure_noncommit\":true,\"retry_exact\":true,\"state_candidate_alias_refused\":true,\"exact_own_primitive_state_and_output\":true,\"qualified\":false,\"tensor_bytes\":"<<ggml_backend_buffer_get_size(buf.get())<<"}\n";native_hc_need(bool(report),"GDN conv report");return 0;
}catch(const std::exception&e){fprintf(stderr,"native GDN conv: %s\n",e.what());return 1;}}
