#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_shared_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include "native_shared_fixture.hpp"
#include <set>
using namespace strata::core::portable_detail;
using native_shared_fixture::need;
int main(int argc,char** argv) {try {
    auto code=native_component_float_controls(strata_native_shared_spv,sizeof(strata_native_shared_spv)/4,NativeComponentAddressing::descriptor);
    size_t dots=0,fmas=0,modes=0;std::set<uint32_t> precise;
    for(size_t p=5;p<code.size();p+=code[p]>>16) {
        const auto op=code[p]&65535;
        need(code[p]>>16 && p+(code[p]>>16)<=code.size(),"shared malformed module");
        if(op==spv::OpDecorate && code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
        dots+=op==spv::OpSDot;fmas+=op==4427; // registry SPV_KHR_fma OpFmaKHR
        if(op==spv::OpFAdd)need(precise.count(code[p+2]),"shared addition can contract");
        if(op==spv::OpExecutionMode && code[p+3]==32) {
            need(code[p+2]!=spv::ExecutionModeDenormPreserve && code[p+2]!=spv::ExecutionModeDenormFlushToZero,"shared denorm policy introduced");
            modes+=code[p+2]==spv::ExecutionModeRoundingModeRTE || code[p+2]==spv::ExecutionModeSignedZeroInfNanPreserve;
        }
    }
    need(dots==2 && fmas==3 && modes==2,"shared dense/scalar mapping changed");
    for(unsigned flags=0;flags<4;++flags) {
        auto effective=native_q8_test_helper_effective(code,flags&1,flags&2);
        const std::string path=argc==2?std::string(argv[1])+"."+std::to_string(flags)+".spv":"";
        native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),flags&1,flags&2,argc==2?path.c_str():nullptr);
        auto changed=effective;
        for(size_t p=5;p<changed.size();p+=changed[p]>>16)if((changed[p]&65535)==4427){changed[p+3]^=1;break;}
        bool refused=false;try{native_q8_validate_observation_effective(code.data(),code.size(),changed.data(),changed.size(),flags&1,flags&2,nullptr);}catch(...){refused=true;}
        need(refused,"shared effective FMA mutation accepted");
    }
    puts("native shared effective modules and exact helper transformations passed; no GPU execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
