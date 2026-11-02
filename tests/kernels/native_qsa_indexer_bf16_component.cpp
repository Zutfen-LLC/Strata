#include "native_qsa_indexer_bf16_fixture.hpp"
#include "strata_vk_native_shared.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_qsa_indexer_bf16_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3]);return 0;}
 arguments(argc,argv);auto f=read(argv[6]);std::filesystem::path dir=argv[8];need(std::filesystem::create_directory(dir),"indexer BF16 output exists");size_t device=std::stoull(argv[2]);need(device<size_t(ggml_backend_vk_get_device_count()),"indexer BF16 Vulkan device absent");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_vk_init(device),ggml_backend_free);need(bool(backend),"indexer BF16 Vulkan init");const char*name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));need(name&&std::string(name).find(argv[4])!=std::string::npos,"indexer BF16 unexpected device");
 std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({32768,nullptr,true}),ggml_free);need(bool(ctx),"indexer BF16 metadata");constexpr uint32_t guard=0xa5a5a5a5u;std::array<size_t,6>sizes{{2560,128*2560/2,512*2560/2,128,512,1}};std::array<std::vector<uint32_t>,6>before,after;std::array<ggml_tensor*,6>base{},view{};
 for(size_t i=0;i<6;++i){before[i].assign(sizes[i]+16,guard);base[i]=ggml_new_tensor_1d(ctx.get(),i==1||i==2?GGML_TYPE_I32:GGML_TYPE_F32,before[i].size());view[i]=ggml_view_1d(ctx.get(),base[i],sizes[i],8*4);}
 std::memcpy(before[0].data()+8,f.x.data(),10240);for(size_t i=0;i<2;++i)std::memcpy(before[i+1].data()+8,f.w[i].data(),f.w[i].size()*2);
 size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));need(bound<=16u<<20,"indexer BF16 tensor budget");std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer)&&ggml_backend_buffer_get_size(buffer.get())<=bound,"indexer BF16 allocation");for(size_t i=0;i<6;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);
 std::unique_ptr<StrataVkNativeShared,decltype(&strata_vk_native_shared_destroy)>owner(strata_vk_qsa_indexer_bf16_create(backend.get(),(dir/"effective.spv").c_str(),bound+(1u<<20)),strata_vk_native_shared_destroy);std::array<std::vector<float>,2>first;
 for(int pass=0;pass<2;++pass)for(size_t i=0;i<2;++i){strata_vk_native_shared_run(owner.get(),1,view[i+1],view[0],view[i+3],view[5],2560,rows[i],1);after[i+3].resize(sizes[i+3]+16);ggml_backend_tensor_get(base[i+3],after[i+3].data(),0,after[i+3].size()*4);for(size_t j=0;j<8;++j)need(after[i+3][j]==guard&&after[i+3][after[i+3].size()-1-j]==guard,"indexer BF16 output guard");std::vector<float>v(rows[i]);std::memcpy(v.data(),after[i+3].data()+8,v.size()*4);if(!pass)first[i]=v;else need(v==first[i],"indexer BF16 repeat differs");const std::string role=i?"query":"key";save(dir/(role+(pass?"-repeat.f32le":".f32le")),v.data(),v.size()*4);}
 for(size_t i=0;i<3;++i){after[i].resize(before[i].size());ggml_backend_tensor_get(base[i],after[i].data(),0,after[i].size()*4);need(after[i]==before[i],"indexer BF16 input changed");}literals(first);std::ofstream(dir/"report.json")<<"{\"actual_vulkan_producer\":true,\"immutable_canary_repeat\":true,\"qualified\":false}\n";return 0;
}catch(const std::exception&e){std::fprintf(stderr,"Vulkan indexer BF16: %s\n",e.what());return 1;}}
