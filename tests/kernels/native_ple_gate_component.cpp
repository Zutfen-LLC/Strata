// Standalone Vulkan observation of the T1 native PLE post-projection gate.
#include <cfenv>
#include "native_hc_layout.hpp"
#include "strata_vk_native_ple_gate.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
using namespace strata::core::portable_detail;
namespace {
std::vector<uint32_t> read(const std::filesystem::path& p){
    native_hc_need(!std::filesystem::is_symlink(p),"PLE gate: symlink fixture");
    std::ifstream f(p,std::ios::binary);std::vector<uint32_t> v(10240);f.read(reinterpret_cast<char*>(v.data()),40960);
    native_hc_need(f.gcount()==40960&&f.peek()==EOF,"PLE gate: exact input fixture required");
    for(auto bits:v){float x;std::memcpy(&x,&bits,4);native_hc_normal(x);}return v;
}
void save(const std::filesystem::path& p,const void* data,size_t n){native_hc_need(!std::filesystem::exists(p),"PLE gate: output exists");std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),n);f.flush();native_hc_need(bool(f),"PLE gate: output write");}
}
int main(int argc,char**argv){try{
    const uint16_t endian=1;native_hc_need(*reinterpret_cast<const uint8_t*>(&endian)==1,"PLE gate: little-endian host required");
    if(argc==4&&std::string(argv[1])=="--contract"){
        read(std::filesystem::path(argv[2])/"key.f32le");read(std::filesystem::path(argv[2])/"query.f32le");
        const char result[]="{\"key_bytes\":40960,\"query_bytes\":40960,\"gate_bytes\":16,\"diagnostic_bytes\":64,\"qualified\":false}\n";save(argv[3],result,sizeof(result)-1);return 0;
    }
    native_hc_need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","PLE gate arguments");
    const auto fixture=std::filesystem::path(argv[6]);auto key=read(fixture/"key.f32le"),query=read(fixture/"query.f32le");
    std::string index=argv[2];native_hc_need(!index.empty()&&index.find_first_not_of("0123456789")==std::string::npos,"PLE gate: device index");
    auto device=std::stoull(index);native_hc_need(device<size_t(ggml_backend_vk_get_device_count()),"PLE gate: device missing");
    const std::filesystem::path out=argv[8];native_hc_need(std::filesystem::create_directory(out),"PLE gate: fresh output required");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    native_hc_need(backend&&*argv[4]&&std::string(ggml_backend_dev_description(ggml_backend_get_device(backend.get()))).find(argv[4])!=std::string::npos,"PLE gate: device identity differs");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({32768,nullptr,true}),ggml_free);native_hc_need(bool(ctx),"PLE gate: metadata");
    std::array<ggml_tensor*,4> bank{},t{};const size_t words[]={10240,10240,4,16};
    for(size_t i=0;i<t.size();++i){bank[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,words[i]+16);t[i]=ggml_view_1d(ctx.get(),bank[i],words[i],32);}
    const size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));native_hc_need(bound<=1u<<20,"PLE gate: bank bound");
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
    native_hc_need(bool(buffer)&&ggml_backend_buffer_get_size(buffer.get())<=1u<<20,"PLE gate: bank allocation");
    std::unique_ptr<StrataVkNativePleGate,decltype(&strata_vk_native_ple_gate_destroy)> owner(strata_vk_native_ple_gate_create(backend.get(),(out/"effective.spv").c_str(),2u<<20),strata_vk_native_ple_gate_destroy);
    std::array<std::vector<uint32_t>,2> first;
    for(int pass=0;pass<2;++pass){
        for(size_t i=0;i<4;++i){std::vector<uint32_t> initial(words[i]+16,0x4acaf00d);ggml_backend_tensor_set(bank[i],initial.data(),0,initial.size()*4);}
        ggml_backend_tensor_set(t[0],key.data(),0,40960);ggml_backend_tensor_set(t[1],query.data(),0,40960);
        native_hc_need(strata_vk_native_ple_gate_run(owner.get(),t[0],t[1],t[2],t[3])==0,"PLE gate: bounded domain refused");
        for(size_t i=0;i<4;++i){std::vector<uint32_t> actual(words[i]+16);ggml_backend_tensor_get(bank[i],actual.data(),0,actual.size()*4);
            for(size_t j=0;j<8;++j)native_hc_need(actual[j]==0x4acaf00d&&actual[8+words[i]+j]==0x4acaf00d,"PLE gate: canary changed");
            if(i<2)native_hc_need(std::memcmp(actual.data()+8,(i?query:key).data(),40960)==0,"PLE gate: input changed");
            else {std::vector<uint32_t> value(actual.begin()+8,actual.begin()+8+words[i]);if(pass)native_hc_need(value==first[i-2],"PLE gate: repeat differs");first[i-2]=std::move(value);}
        }
    }
    native_hc_need(strata_vk_native_ple_gate_calls(owner.get())==2,"PLE gate: dispatch count");
    save(out/"gate.f32le",first[0].data(),16);save(out/"diagnostic.f32le",first[1].data(),64);
    const char report[]="{\"actual_vulkan\":true,\"immutable_inputs_canaries_repeat\":true,\"qualified\":false}\n";save(out/"report.json",report,sizeof(report)-1);
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"PLE gate: %s\n",e.what());return 1;}}
