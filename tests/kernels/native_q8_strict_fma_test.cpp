#include "native_q8_strict_fma.hpp"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <vector>
using strata::core::portable_detail::NativeQ8StrictFmaState;
struct FmaReceipt {NativeQ8StrictFmaState state;bool chained,admitted;size_t extensions;VkBool32 fp32,fp16,fp64;};
#include "strata_native_q8_fma_test.hpp"
void need(bool v,const char* message){if(!v)throw std::runtime_error(message);}
int main(){try {
    for(bool extension:{false,true})for(bool support:{false,true})for(bool created:{false,true}) {
        const auto r=fma_feature_chain(extension,support,created);
        need(r.chained==extension && r.extensions==size_t(extension),"extension/feature query chain mismatch");
        need(r.admitted==(extension && support),"unsupported FP32 FMA admitted");
        need(r.state.enabled==(extension && support && created),"enablement ignored logical creation failure");
        if(r.admitted)need(r.fp32 && !r.fp16 && !r.fp64,"request enabled unrequested widths");
        bool accepted=true;try{r.state.require_enabled();}catch(...){accepted=false;}
        need(accepted==r.state.enabled,"shader admission ignored logical enablement");
    }
    NativeQ8StrictFmaState query_only;query_only.queried(true,true);
    bool rejected=false;try{query_only.require_enabled();}catch(...){rejected=true;}
    need(rejected,"physical support alone admitted strict shader");
    puts("strict FMA: actual query/request chain, unsupported feature/extension, FP32-only request and logical creation receipt checked; no GPU execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
