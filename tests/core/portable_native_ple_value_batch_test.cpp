#include "portable_native_ple_value.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <chrono>
using namespace strata::core::portable_detail;
namespace { unsigned calls=0; }
namespace strata::core::portable_detail {
struct StrataVkNativeShared {};
StrataVkNativeShared* strata_vk_ple_value_bf16_create(ggml_backend_t,const char*,size_t) { return new StrataVkNativeShared; }
void strata_vk_native_shared_destroy(StrataVkNativeShared* p) noexcept { delete p; }
size_t strata_vk_native_shared_bytes(const StrataVkNativeShared*) { return 0; }
void strata_vk_native_shared_run(StrataVkNativeShared*,uint32_t mode,const ggml_tensor* w,const ggml_tensor* x,ggml_tensor* out,ggml_tensor*,uint32_t ni,uint32_t no,uint32_t tokens) {
    native_shared_need(mode==1 && ni==2560 && no==2560 && (tokens==1 || tokens==2),"mock grid");
    native_shared_need(ggml_nbytes(x)==size_t(tokens)*ni*4 && ggml_nbytes(out)==size_t(tokens)*no*4,"mock spans");
    std::vector<uint16_t> weights(size_t(ni)*no); ggml_backend_tensor_get(w,weights.data(),0,weights.size()*2);
    std::vector<float> input(size_t(tokens)*ni), result; ggml_backend_tensor_get(x,input.data(),0,input.size()*4);
    for(size_t t=0;t<tokens;++t) { auto row=native_bf16_matrix_control(input.data()+t*ni,weights.data(),ni); result.insert(result.end(),row.begin(),row.end()); }
    ggml_backend_tensor_set(out,result.data(),0,result.size()*4); ++calls;
}
}
int main(int argc,char** argv) { try {
    native_shared_need(argc==2,"scratch root");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> b(ggml_backend_cpu_init(),ggml_backend_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({16384,nullptr,true}),ggml_free);
    auto* w=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,2560*2560/2);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> wb(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);
    std::vector<uint16_t> weights(2560*2560,0); weights[0]=0x3f80; weights[2560]=0x4000;
    ggml_backend_tensor_set(w,weights.data(),0,weights.size()*2);
    auto dir=std::filesystem::path(argv[1])/("ple-value-batch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    PortableNativePleValue owner(b.get(),w,weights,dir.string(),PortableNativePleValue::estimate(b.get(),2),2);
    std::vector<float> input(5120,0); input[0]=3; input[2560]=5;
    owner.run(input); std::vector<float> out(5120); ggml_backend_tensor_get(owner.output(),out.data(),0,out.size()*4);
    native_shared_need(calls==1 && out[0]==3 && out[1]==6 && out[2560]==5 && out[2561]==10,"full T2 row offsets");
    std::vector<float> tail(2560,0); tail[0]=7;
    owner.run(tail); ggml_backend_tensor_get(owner.output(),out.data(),0,out.size()*4);
    native_shared_need(calls==2 && out[0]==7 && out[1]==14 && out[2560]==5 && out[2561]==10,
                       "T1 tail must preserve inactive row");
    bool rejected=false; try { owner.run(std::vector<float>(2561)); } catch(...) { rejected=true; }
    native_shared_need(rejected && calls==2,"bad span dispatched");
    rejected=false; try { PortableNativePleValue::estimate(b.get(),3); } catch(...) { rejected=true; }
    native_shared_need(rejected,"T3 accepted");
    std::puts("PLE value T2 mock passed"); return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; } }
