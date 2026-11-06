#include "native_q8_boundary_input.hpp"
#include "native_q8_boundary_domain.hpp"
#include "strata_vk_native_stages.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
using namespace strata::core::portable_detail;
static void need(bool p,const char* s){if(!p)throw std::runtime_error(s);}
int main(int argc,char** argv) {try {
    need(argc==6,"usage: DEVICE NAME-SUBSTRING expert|gdn-projection INPUT OUTPUT");
    need(!std::filesystem::exists(argv[5]) && !std::filesystem::exists(std::string(argv[5])+".domain.json") && !std::filesystem::exists(std::string(argv[5])+".effective.spv"),"output already exists");
    const size_t device=std::stoull(argv[1]); const std::string expected=argv[2],mode=argv[3];
    need(!expected.empty() && (mode=="expert" || mode=="gdn-projection"),"invalid selector");
    std::array<unsigned char,512> input{}; std::ifstream f(argv[4],std::ios::binary);
    need(bool(f.read(reinterpret_cast<char*>(input.data()),input.size())) && f.peek()==EOF,"input must contain exactly 128 float32 values");
    boundary_input_check(input);
    need(device<size_t(ggml_backend_vk_get_device_count()),"device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free); need(bool(backend),"init failed");
    const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get())); need(std::string(name).find(expected)!=std::string::npos,"unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({8192,nullptr,true}),ggml_free); need(bool(ctx),"context failed");
    auto* x=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,128); auto* u=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,128);
    auto* h=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,128); auto* q=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,36);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free); need(bool(buffer),"allocation failed");
    ggml_backend_tensor_set(x,input.data(),0,input.size());
    std::string spv=std::string(argv[5])+".effective.spv";
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> stage(
        mode=="expert" ? strata_vk_native_stages_create_quant_rn_observation(backend.get(),spv.c_str()) : strata_vk_native_stages_create_quant_rcp_observation(backend.get(),spv.c_str()),strata_vk_native_stages_destroy);
    const int32_t starts[]={0,1}; strata_vk_native_stages_run(stage.get(),0,x,u,h,q,128,starts,2);
    std::array<unsigned char,144> result{}; ggml_backend_tensor_get(q,result.data(),0,result.size());
    std::ofstream out(argv[5],std::ios::binary);out.write(reinterpret_cast<const char*>(result.data()),result.size());need(bool(out),"output write failed");
    if(mode=="gdn-projection") {
        std::ofstream domain(std::string(argv[5])+".domain.json");need(bool(domain),"domain output create failed");
        domain << "{\"scope\":\"individual GDN input/packed-scale checks; isolated shader only\",\"model_execution\":false,\"blocks\":[";
        for(size_t i=0;i<4;++i){
            const auto input_error=boundary_gdn_input_error(input.data()+i*128);
            const auto scale_error=input_error.empty()?boundary_gdn_scale_error(result.data()+i*36):std::string();
            const char* state=!input_error.empty()?"unsupported_input":!scale_error.empty()?"unsupported_packed_scale":"block_checks_passed";
            domain << (i?",":"") << "{\"state\":" << std::quoted(state) << ",\"reason\":" << std::quoted(input_error.empty()?scale_error:input_error) << "}";
        }
        domain << "]}\n";need(bool(domain),"domain output write failed");
    }
    std::printf("producer=%s device=%s isolated_shader=true model_execution=false qualified=false fp32_denorm=unspecified bytes=%zu\n",mode.c_str(),name,result.size());return 0;
} catch(const std::exception& e){std::fprintf(stderr,"Vulkan boundary: %s\n",e.what());return 1;} }
