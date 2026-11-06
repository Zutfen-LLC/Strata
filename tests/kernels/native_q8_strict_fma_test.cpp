#include "native_q8_strict_fma.hpp"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
using strata::core::portable_detail::NativeQ8StrictFmaState;
struct FmaReceipt {NativeQ8StrictFmaState state;bool chained,admitted;size_t extensions;VkBool32 fp32,fp16,fp64;};
#include "strata_native_q8_fma_test.hpp"
void need(bool v,const char* message){if(!v)throw std::runtime_error(message);}
int main(){try {
    for(bool extension:{false,true}) {
        std::FILE* output=std::tmpfile();need(output!=nullptr,"identity output unavailable");
        const uint8_t uuid[16]={0x00,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,1,2,3,4,5,6,0xff};
        strata::core::portable_detail::log_native_q8_fma_device(output,2,3,"test device",0x1002,0x1234,0x4567,0x89ab,uuid,extension,false);
        std::rewind(output);char line[512]{};need(std::fgets(line,sizeof line,output)!=nullptr,"identity log missing");std::fclose(output);
        const std::string text=line;
        need(text.find("index=2 physical_index=3 name=\"test device\" vendor=0x1002 device=0x1234")!=std::string::npos &&
             text.find("driver_version=0x00004567 api_version=0x000089ab")!=std::string::npos &&
             text.find("uuid=00123456789abcdef0010203040506ff")!=std::string::npos &&
             text.find(extension?"extension=1 fp32_queried=1 fp32=0":"extension=0 fp32_queried=0 fp32=0")!=std::string::npos,"identity log fields mismatch");
    }
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
