#include "strata/core/portable_ffn.hpp"
#include "ggml-cpu.h"
#include <memory>
#include <cstdio>
int main(){
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    strata::core::PortableFfnOptions options;options.accelerator=backend.get();options.request_accelerator=true;
    options.native_router=true;options.native_router_receipt_prefix="unused-no-receipt-created";
    std::string error;
    auto result=strata::core::PortableFfn::create({2560,4,1,32,512,10,1e-6f},0,{},options,error);
    if(result || error.find("native router requires Vulkan native-stage build")==std::string::npos){fprintf(stderr,"unsupported router build not refused: %s\n",error.c_str());return 1;}
    options.native_router=false;options.native_shared=true;options.native_shared_receipt_prefix="unused-shared";
    result=strata::core::PortableFfn::create({2560,4,1,32,512,10,1e-6f},0,{},options,error);
    if(result || error.find("native shared requires strict-FMA native-stage build")==std::string::npos){fprintf(stderr,"unsupported shared build not refused: %s\n",error.c_str());return 1;}
    puts("native router unavailable-build refusal precedes allocation/dispatch");return 0;
}
