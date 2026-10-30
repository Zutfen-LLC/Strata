#include "native_q8_quant_trace.hpp"
#include "native_q8_observation.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_stages_spv.hpp"
#include "strata_native_quant_trace_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include <map>
#include <set>
#include <cstdio>
using namespace strata::core::portable_detail;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::map<uint32_t,size_t> arithmetic(const std::vector<uint32_t>& code) {
    std::map<uint32_t,size_t> result;
    for(size_t p=5;p<code.size();p+=code[p]>>16){const uint32_t op=code[p]&65535;
        if(op==spv::OpDecorate && code[p+2]==spv::DecorationNoContraction)++result[0x20000u];
        if(op==spv::OpExtInst)++result[0x10000u+code[p+4]];
        if(op==spv::OpFAdd || op==spv::OpFSub || op==spv::OpFMul || op==spv::OpFDiv || op==spv::OpFNegate ||
            op==spv::OpFOrdGreaterThan || op==spv::OpFOrdGreaterThanEqual || op==spv::OpFOrdLessThan ||
            op==spv::OpConvertFToS || op==spv::OpIsNan)++result[op];}
    return result;
}
int main(int argc,char** argv){try {
    const int32_t starts[]={0,3};native_q8_quant_trace_shape(0,7680,2560,starts,2);
    auto reject=[&](uint32_t mode,int64_t n,uint32_t f,const int32_t* s,size_t count){bool bad=false;
        try{native_q8_quant_trace_shape(mode,n,f,s,count);}catch(...){bad=true;}need(bad,"unsafe trace shape accepted");};
    reject(1,7680,2560,starts,2);reject(2,7680,2560,starts,2);reject(0,4480,2560,starts,2);
    reject(0,7679,2560,starts,2);reject(0,7681,2560,starts,2);reject(0,7680,640,starts,2);
    reject(0,7680,2560,nullptr,2);reject(0,7680,2560,starts,1);reject(0,7680,2560,starts,3);
    const int32_t wrong[]={1,3};reject(0,7680,2560,wrong,2);
    need(NATIVE_Q8_TRACE_ELEMENTS==std::array<uint32_t,4>{36,118,1046,4479} && NATIVE_Q8_TRACE_WORDS==16,"trace element/word map changed");
    const auto base=native_stage_float_controls(strata_native_stages_spv,sizeof(strata_native_stages_spv)/4);
    const auto strict=native_stage_float_controls(strata_native_quant_trace_spv,sizeof(strata_native_quant_trace_spv)/4);
    need(arithmetic(base)==arithmetic(strict),"instrumentation added/removed arithmetic instructions");
    std::set<uint32_t> precise,divisions;
    for(size_t p=5;p<strict.size();p+=strict[p]>>16){const auto op=strict[p]&65535;
        if(op==spv::OpDecorate && strict[p+2]==spv::DecorationNoContraction)precise.insert(strict[p+1]);
        if(op==spv::OpFDiv)divisions.insert(strict[p+2]);
        need(!(op>=spv::OpGroupNonUniformElect && op<=spv::OpGroupNonUniformQuadSwap),"physical subgroup introduced");}
    need(divisions.size()==3,"division mapping changed");for(auto id:divisions)need(precise.count(id),"division lost NoContraction");
    const auto observation=native_q8_observation_code(strict);
    for(int mask=0;mask<4;++mask){const auto effective=native_q8_test_helper_effective(observation,mask&1,mask&2);
        const std::string path=argc==2?std::string(argv[1])+"."+std::to_string(mask)+".spv":"";
        native_q8_validate_observation_effective(observation.data(),observation.size(),effective.data(),effective.size(),mask&1,mask&2,argc==2?path.c_str():nullptr);}
    puts("quant trace: fixed bounds, unchanged arithmetic profile/division controls and effective modules checked; no GPU execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
