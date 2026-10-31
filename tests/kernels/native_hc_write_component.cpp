#include "native_hc_write_fixture.hpp"
#include "strata_vk_native_hc.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_hc_write_fixture;
int main(int argc,char** argv){try{
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],synthetic(argv[2]));return 0;}
 native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","HC write: --synthetic CASE FILE or --device N --expect-device NAME --fixture FILE --output FRESH_DIR");
 auto f=read(argv[6]);std::string dir=argv[8];native_hc_need(std::filesystem::create_directory(dir),"HC write: fresh output");std::filesystem::copy_file(argv[6],dir+"/fixture.bin");auto d=std::stoul(argv[2]);native_hc_need(d<size_t(ggml_backend_vk_get_device_count()),"HC write: device");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(d),ggml_backend_free);native_hc_need(bool(backend),"HC write: Vulkan init");const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));native_hc_need(name&&std::string(name).find(argv[4])!=std::string::npos,"HC write: device identity");
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({65536,nullptr,true}),ggml_free);native_hc_need(bool(ctx),"HC write: metadata");
 const std::array<size_t,6> count{10240,2560,4,10240,4,4};std::array<ggml_tensor*,6> base{},v{};std::array<std::vector<uint32_t>,6> before,first;
 const void* inputs[]={f.residual.data(),f.block.data(),f.inject.data()};
 for(size_t i=0;i<6;++i){before[i].assign(i+2+count[i]+9,0xa5a5a5a5);if(i<3)std::memcpy(before[i].data()+i+2,inputs[i],count[i]*4);base[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,before[i].size());v[i]=ggml_view_1d(ctx.get(),base[i],count[i],(i+2)*4);}
 const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));native_hc_need(bound<=128u<<10,"HC write: tensor admission");std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);native_hc_need(buffer&&ggml_backend_buffer_get_size(buffer.get())<=128u<<10,"HC write: allocation bound");for(size_t i=0;i<6;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);
 bool short_refused=false;try{auto* bad=strata_vk_native_hc_write_create(backend.get(),nullptr,1);strata_vk_native_hc_pending_destroy(bad);}catch(const std::exception&){short_refused=true;}native_hc_need(short_refused,"HC write: short budget admitted");
 std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> owner(strata_vk_native_hc_write_create(backend.get(),(dir+"/write-effective.spv").c_str()),strata_vk_native_hc_pending_destroy);native_hc_need(strata_vk_native_hc_pending_bytes(owner.get())<=65536,"HC write: owner bound");
 bool alias=false;try{strata_vk_native_hc_pending_run(owner.get(),v[0],v[1],v[2],v[0],v[4],v[5]);}catch(const std::invalid_argument&){alias=true;}native_hc_need(alias,"HC write: producer alias admitted");
 for(int repeat=0;repeat<2;++repeat){strata_vk_native_hc_pending_run(owner.get(),v[0],v[1],v[2],v[3],v[4],v[5]);
  for(size_t i=0;i<6;++i){std::vector<uint32_t> a(before[i].size());ggml_backend_tensor_get(base[i],a.data(),0,a.size()*4);for(size_t j=0;j<a.size();++j)if(i<3||j<i+2||j>=i+2+count[i])native_hc_need(a[j]==before[i][j],"HC write: producer/canary mutation");if(repeat)native_hc_need(a==first[i],"HC write: rejected consumer/reset/replay changed candidate");else first[i]=a;}
  // A rejecting consumer does not update any producer. Accept only after it succeeds.
  std::vector<uint32_t> committed=before[0];if(repeat)std::copy(first[3].begin()+5,first[3].begin()+5+10240,committed.begin()+2);
  if(!repeat)native_hc_need(committed==before[0],"HC write: rollback committed");else native_hc_need(std::equal(committed.begin()+2,committed.begin()+2+10240,first[3].begin()+5),"HC write: accepted materialization mismatch");
 }
 const char* names[]={"residual","gate","exp"};for(size_t i=3;i<6;++i){std::vector<float> a(count[i]);std::memcpy(a.data(),first[i].data()+i+2,count[i]*4);for(float x:a)native_hc_normal(x);floats(dir+"/"+names[i-3]+".f32le",a);}
 std::ofstream o(dir+"/report.json");o<<"{\"tokens\":1,\"candidate_only\":true,\"producer_immutable\":true,\"consumer_rollback\":true,\"explicit_materialization\":true,\"reset_repeat_exact\":true,\"alias_refused\":true,\"short_budget_refused\":true,\"tensor_bytes\":"<<ggml_backend_buffer_get_size(buffer.get())<<",\"owner_bytes\":"<<strata_vk_native_hc_pending_bytes(owner.get())<<",\"qualified\":false,\"model_integration\":false}\n";native_hc_need(bool(o),"HC write: report");return 0;
}catch(const std::exception& e){fprintf(stderr,"HC write: %s\n",e.what());return 1;}}
