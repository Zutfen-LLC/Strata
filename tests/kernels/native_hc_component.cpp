#include "native_hc_fixture.hpp"
#include "strata_vk_native_hc.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_hc_fixture;
int main(int argc,char** argv){try{
 if(argc==6&&std::string(argv[1])=="--from-q8"){
  Fixture f;const auto bytes=std::filesystem::file_size(argv[2]);native_hc_need(bytes>0&&bytes%40960==0&&bytes<=8*40960,"native HC: bounded residual file");f.t=uint32_t(bytes/40960);native_hc_shape(2560,4,f.t);native_hc_need(std::filesystem::file_size(argv[3])==40960&&std::filesystem::file_size(argv[4])==43520,"native HC: bounded parameter files");f.x.resize(bytes/4);f.w.resize(10240);std::vector<uint8_t> q(43520);std::ifstream x(argv[2],std::ios::binary),w(argv[3],std::ios::binary),wi(argv[4],std::ios::binary);for(float& v:f.x)v=value(word(x));for(float& v:f.w)v=value(word(w));native_hc_need(bool(wi.read(reinterpret_cast<char*>(q.data()),q.size())),"native HC: Q8 source read");f.wi=native_hc_q8_bf16(q.data(),q.size(),40960);write_fixture(argv[5],f);return 0;
 }
 if(argc==3&&std::string(argv[1])=="--synthetic"){write_fixture(argv[2],synthetic());return 0;}
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","usage: --device N --expect-device NAME --fixture FILE --output FRESH_DIR");
 auto f=read_fixture(argv[6]);const std::string dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"native HC: exclusive output");std::filesystem::copy_file(argv[6],dir+"/fixture.bin");auto device=std::stoul(argv[2]);native_hc_need(device<size_t(ggml_backend_vk_get_device_count()),"native HC: Vulkan device absent");std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);native_hc_need(bool(backend),"native HC: Vulkan init");const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));native_hc_need(name&&std::string(name).find(argv[4])!=std::string::npos,"native HC: unexpected device");
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({65536,nullptr,true}),ggml_free);native_hc_need(bool(ctx),"native HC: metadata");
 const std::array<size_t,6> count{f.x.size(),f.w.size(),f.wi.size()/2,f.x.size(),f.t*4,f.t*4};std::array<ggml_tensor*,6> base{},v{};std::array<std::vector<uint32_t>,6> before,after;
 for(size_t i=0;i<6;++i){size_t prefix=2+i;before[i].assign(prefix+count[i]+9,0xa5a5a5a5);if(i==0)std::memcpy(before[i].data()+prefix,f.x.data(),count[i]*4);if(i==1)std::memcpy(before[i].data()+prefix,f.w.data(),count[i]*4);if(i==2)std::memcpy(before[i].data()+prefix,f.wi.data(),count[i]*4);base[i]=ggml_new_tensor_1d(ctx.get(),i==2?GGML_TYPE_I32:GGML_TYPE_F32,before[i].size());v[i]=ggml_view_1d(ctx.get(),base[i],count[i],prefix*4);}
 auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));native_hc_need(bound<=16u<<20,"native HC: tensor budget");std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);native_hc_need(bool(buffer)&&ggml_backend_buffer_get_size(buffer.get())<=16u<<20,"native HC: actual tensor budget");
 for(size_t i=0;i<6;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);
 std::unique_ptr<StrataVkNativeHc,decltype(&strata_vk_native_hc_destroy)> component(strata_vk_native_hc_create(backend.get(),(dir+"/effective.spv").c_str()),strata_vk_native_hc_destroy);
 std::array<std::vector<uint32_t>,3> first;
 for(int repeat=0;repeat<2;++repeat){strata_vk_native_hc_run(component.get(),v[0],v[1],v[2],v[3],v[4],v[5],f.t);
  for(size_t i=0;i<6;++i){after[i].resize(before[i].size());ggml_backend_tensor_get(base[i],after[i].data(),0,after[i].size()*4);for(size_t j=0;j<after[i].size();++j)if(i<3||j<2+i||j>=2+i+count[i])native_hc_need(before[i][j]==after[i][j],"native HC: immutable input/canary changed");
   if(i>=3){std::vector<uint32_t> view(after[i].begin()+2+i,after[i].begin()+2+i+count[i]);if(repeat)native_hc_need(first[i-3]==view,"native HC: repeated output changed");else first[i-3]=std::move(view);}}
 }
 for(size_t i=3;i<6;++i){std::vector<float> out(count[i]);std::memcpy(out.data(),first[i-3].data(),count[i]*4);for(float x:out)native_hc_normal(x);floats(dir+"/"+std::array<const char*,3>{"normalized.f32le","rs.f32le","inject.f32le"}[i-3],out);}
 std::ofstream report(dir+"/report.json");report<<"{\"guards\":true,\"finite_normal_domain\":true,\"repeat_exact\":true,\"qualified\":false,\"cuda_rsqrt_equivalence_established\":false,\"subnormal_qualification_supported\":false,\"tokens\":"<<f.t<<",\"tensor_bytes\":"<<ggml_backend_buffer_get_size(buffer.get())<<",\"metadata_bytes\":"<<strata_vk_native_hc_bytes(component.get())<<"}\n";native_hc_need(bool(report),"native HC: report write");printf("native_hc_observation=true qualified=false device=%s\n",name);return 0;
}catch(const std::exception& e){fprintf(stderr,"native HC component: %s\n",e.what());return 1;}}
