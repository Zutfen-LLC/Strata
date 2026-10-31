#include "native_hc_pending_fixture.hpp"
#include "strata_vk_native_hc.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_hc_pending_fixture;
int main(int argc,char** argv){try{
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],synthetic(argv[2]));return 0;}
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","HC pending: --synthetic CASE FILE or --device N --expect-device NAME --fixture FILE --output FRESH_DIR");
 auto f=read(argv[6]);std::string dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"HC pending: fresh output");std::filesystem::copy_file(argv[6],dir+"/fixture.bin");auto device=std::stoul(argv[2]);native_hc_need(device<size_t(ggml_backend_vk_get_device_count()),"HC pending: missing device");std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);native_hc_need(bool(backend),"HC pending: Vulkan init");const auto* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));native_hc_need(name&&std::string(name).find(argv[4])!=std::string::npos,"HC pending: device identity");
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({65536,nullptr,true}),ggml_free);native_hc_need(bool(ctx),"HC pending: metadata");
 const std::array<size_t,20> count{10240,2560,4,10240,20480,1638400,1638400,10240,4,4,10240,4,4,320,320,320,10240,10240,10240,2560};
 std::array<ggml_tensor*,20> base{},v{};std::array<std::vector<uint32_t>,20> before,first;
 const void* inputs[]={f.norm.x.data(),f.bo.data(),f.inj.data(),f.norm.w.data(),f.norm.wi.data(),f.down.data(),f.up.data()};
 for(size_t i=0;i<count.size();++i){const auto prefix=i+2;before[i].assign(prefix+count[i]+9,0xa5a5a5a5);if(i<7)std::memcpy(before[i].data()+prefix,inputs[i],count[i]*4);base[i]=ggml_new_tensor_1d(ctx.get(),i>=4&&i<=6?GGML_TYPE_I32:GGML_TYPE_F32,before[i].size());v[i]=ggml_view_1d(ctx.get(),base[i],count[i],prefix*4);}
 auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));native_hc_need(bound<=16u<<20,"HC pending: tensor bound");std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);native_hc_need(buffer&&ggml_backend_buffer_get_size(buffer.get())<=16u<<20,"HC pending: actual tensor budget");for(size_t i=0;i<count.size();++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);
 std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> pending(strata_vk_native_hc_pending_create(backend.get(),(dir+"/pending-effective.spv").c_str()),strata_vk_native_hc_pending_destroy);
 std::unique_ptr<StrataVkNativeHc,decltype(&strata_vk_native_hc_destroy)> norm(strata_vk_native_hc_create(backend.get(),(dir+"/norm-effective.spv").c_str()),strata_vk_native_hc_destroy);
 std::unique_ptr<StrataVkNativeHcMix,decltype(&strata_vk_native_hc_mix_destroy)> mix(strata_vk_native_hc_mix_create(backend.get(),(dir+"/mix-effective.spv").c_str()),strata_vk_native_hc_mix_destroy);
 bool refused=false;try{strata_vk_native_hc_pending_run(pending.get(),v[0],v[1],v[2],v[0],v[8],v[9]);}catch(const std::invalid_argument&){refused=true;}native_hc_need(refused,"HC pending: original R alias accepted");
 for(int repeat=0;repeat<2;++repeat){
  strata_vk_native_hc_pending_run(pending.get(),v[0],v[1],v[2],v[7],v[8],v[9]);
  strata_vk_native_hc_run(norm.get(),v[7],v[3],v[4],v[10],v[11],v[12],1);
  strata_vk_native_hc_mix_run(mix.get(),v[10],v[5],v[6],v[13],v[14],v[15],v[16],v[17],v[18],v[19]);
  for(size_t i=0;i<count.size();++i){std::vector<uint32_t> actual(before[i].size());ggml_backend_tensor_get(base[i],actual.data(),0,actual.size()*4);for(size_t j=0;j<actual.size();++j)if(i<7||j<i+2||j>=i+2+count[i])native_hc_need(actual[j]==before[i][j],"HC pending: original tuple/weight/canary changed");if(repeat)native_hc_need(actual==first[i],"HC pending: repeated composition changed");else first[i]=std::move(actual);}
 }
 const char* names[]={"residual","pending-gate","pending-exp","normalized","rs","inject","low-pre","low","low-exp","up-pre","gates","up-exp","mixed"};
 for(size_t i=7;i<count.size();++i){std::vector<float> values(count[i]);std::memcpy(values.data(),first[i].data()+i+2,count[i]*4);for(float x:values)native_hc_normal(x);floats(dir+"/"+names[i-7]+".f32le",values);}
 std::ofstream report(dir+"/report.json");report<<"{\"tokens\":1,\"original_tuple_immutable\":true,\"candidate_only\":true,\"canaries\":true,\"repeat_exact\":true,\"alias_refused\":true,\"qualified\":false,\"full_hc_parity\":false,\"tensor_bytes\":"<<ggml_backend_buffer_get_size(buffer.get())<<"}\n";native_hc_need(bool(report),"HC pending: report");return 0;
}catch(const std::exception& e){fprintf(stderr,"HC pending component: %s\n",e.what());return 1;}}
