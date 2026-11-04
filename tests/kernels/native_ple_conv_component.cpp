#include "strata_vk_native_ple_conv.hpp"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::core::portable_detail;
namespace {
void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
const std::array<size_t,10> counts{92160,10240,20480,10240,10240,92160,10240,10240,10240,10240};
const char* files[]={"history.f32le","normalized.f32le","weights.f16le","hidden.f32le","gated.f32le","next-history.f32le","conv.f32le","result.f32le","pre.f32le","exp.f32le"};
std::vector<uint32_t> read(const std::filesystem::path&p,size_t n){need(!std::filesystem::is_symlink(p),"fixture symlink");std::vector<uint32_t>v(n);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),n*4);need(f.gcount()==std::streamsize(n*4)&&f.peek()==EOF,"fixture span differs");return v;}
void save(const std::filesystem::path&p,const void*data,size_t n){need(!std::filesystem::exists(p),"output exists");std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),n);f.flush();need(bool(f),"output write");}
std::array<std::vector<uint32_t>,5> fixture(const std::filesystem::path&dir){std::array<std::vector<uint32_t>,5>v;need(!std::filesystem::is_symlink(dir)&&std::filesystem::is_directory(dir),"fixture directory");for(size_t i=0;i<5;++i){v[i]=read(dir/files[i],counts[i]);for(auto x:v[i])if(i==2)need((x&0x7c00u)!=0x7c00u&&((x>>16)&0x7c00u)!=0x7c00u,"nonfinite half weight");else{x&=0x7fffffffu;need(x<0x7f800000u&&(x==0||x>=0x00800000u),"input must be finite normal or zero");}}return v;}
}
int main(int argc,char**argv){try{
 const uint16_t little=1;need(*reinterpret_cast<const uint8_t*>(&little)==1,"little endian required");
 if(argc==4&&std::string(argv[1])=="--contract"){(void)fixture(argv[2]);const char j[]="{\"history_bytes\":368640,\"channel_bytes\":40960,\"weights_bytes\":81920,\"candidate_history_bytes\":368640,\"qualified\":false}\n";save(argv[3],j,sizeof(j)-1);return 0;}
 need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","invalid component arguments");auto f=fixture(argv[6]);std::string device=argv[2];need(!device.empty()&&device.find_first_not_of("0123456789")==std::string::npos,"device index");
 std::filesystem::path out=argv[8];need(std::filesystem::create_directory(out),"fresh output required");std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>b(ggml_backend_vk_init(std::stoi(device)),ggml_backend_free);need(bool(b),"Vulkan device required");const char*name=ggml_backend_dev_description(ggml_backend_get_device(b.get()));need(*argv[4]&&std::string(name).find(argv[4])!=std::string::npos,"device identity differs");
 std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({32768,nullptr,true}),ggml_free);need(bool(ctx),"metadata");std::array<ggml_tensor*,10>base{},view{};std::array<std::vector<uint32_t>,10>before,first,values;constexpr uint32_t guard=0xa5a5a5a5u;
 for(size_t i=0;i<10;++i){before[i].assign(counts[i]+16,guard);if(i<5)std::copy(f[i].begin(),f[i].end(),before[i].begin()+8);base[i]=ggml_new_tensor_1d(ctx.get(),i==2?GGML_TYPE_I32:GGML_TYPE_F32,before[i].size());view[i]=ggml_view_1d(ctx.get(),base[i],counts[i],32);}
 auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(b.get()));need(bound<=2u<<20,"tensor admission");std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buf(ggml_backend_alloc_ctx_tensors(ctx.get(),b.get()),ggml_backend_buffer_free);need(bool(buf)&&ggml_backend_buffer_get_size(buf.get())<=2u<<20,"tensor allocation");
 std::unique_ptr<StrataVkNativePleConv,decltype(&strata_vk_native_ple_conv_destroy)>k(strata_vk_native_ple_conv_create(b.get(),(out/"effective.spv").c_str(),2u<<20),strata_vk_native_ple_conv_destroy);need(bool(k),"component");
 for(size_t i=0;i<10;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);
 auto*saved=view[5];view[5]=view[0];bool refused=false;try{strata_vk_native_ple_conv_run(k.get(),view.data());}catch(...){refused=true;}view[5]=saved;need(refused,"live and candidate history alias accepted");
 for(int repeat=0;repeat<2;++repeat){for(size_t i=0;i<10;++i)ggml_backend_tensor_set(base[i],before[i].data(),0,before[i].size()*4);strata_vk_native_ple_conv_run(k.get(),view.data());
  for(size_t i=0;i<10;++i){std::vector<uint32_t>v(before[i].size());ggml_backend_tensor_get(base[i],v.data(),0,v.size()*4);for(size_t j=0;j<v.size();++j)if(i<5||j<8||j>=v.size()-8)need(v[j]==before[i][j],"input or canary changed");values[i].assign(v.begin()+8,v.end()-8);if(repeat)need(values[i]==first[i],"repeat differs");else first[i]=values[i];}
  for(size_t i=5;i<10;++i){std::string name=files[i];if(repeat)name.replace(name.size()-6,6,"-repeat.f32le");save(out/name,values[i].data(),values[i].size()*4);}
 }
 const char report[]="{\"actual_vulkan\":true,\"input_canaries_unchanged\":true,\"history_alias_refused\":true,\"repeat_exact\":true,\"qualified\":false}\n";save(out/"report.json",report,sizeof(report)-1);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"native PLE conv: %s\n",e.what());return 1;}}
