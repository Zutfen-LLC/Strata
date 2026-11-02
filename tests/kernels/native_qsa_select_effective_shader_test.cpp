#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_qsa_scores_t1_spv.hpp"
#include "strata_native_qsa_select_t1_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include "native_hc_fixture.hpp"
#include <set>
using namespace strata::core::portable_detail;
inline void need(bool ok,const char* msg){native_hc_need(ok,msg);}
void audit(const uint32_t* input,size_t count,const std::string& prefix,size_t expected_fma,size_t expected_basic) {
    auto code=native_component_float_controls(input,count,NativeComponentAddressing::descriptor);
    bool abi_refused=false;try{q8_gu_float_controls(input,count);}catch(...){abi_refused=true;}need(abi_refused,"QSA score/select descriptor module accepted as physical64");
    size_t dots=0,fmas=0,modes=0,basic=0,storage_barriers=0;std::set<uint32_t> precise;
    for(size_t p=5;p<code.size();p+=code[p]>>16) {
        const auto op=code[p]&65535;
        need(code[p]>>16 && p+(code[p]>>16)<=code.size(),"QSA score/select malformed module");
        if(op==spv::OpDecorate && code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
        storage_barriers+=op==spv::OpMemoryBarrier;dots+=op==spv::OpSDot;fmas+=op==4427; // registry SPV_KHR_fma OpFmaKHR
        if(op==spv::OpFAdd||op==spv::OpFMul||op==spv::OpFDiv||op==spv::OpFSub){++basic;need(precise.count(code[p+2]),"QSA score/select basic operation can contract");}
        if(op==spv::OpExecutionMode && code[p+3]==32) {
            need(code[p+2]!=spv::ExecutionModeDenormPreserve && code[p+2]!=spv::ExecutionModeDenormFlushToZero,"QSA score/select denorm policy introduced");
            modes+=code[p+2]==spv::ExecutionModeRoundingModeRTE || code[p+2]==spv::ExecutionModeSignedZeroInfNanPreserve;
        }
    }
    need(dots==0 && fmas==expected_fma && modes==2 && basic==expected_basic,"QSA score/select dense/scalar mapping changed");
    for(unsigned flags=0;flags<4;++flags) {
        auto effective=native_q8_test_helper_effective(code,flags&1,flags&2);
        const std::string path=prefix.empty()?"":prefix+"."+std::to_string(flags)+".spv";
        native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),flags&1,flags&2,path.empty()?nullptr:path.c_str());
        auto changed=effective;
        for(size_t p=5;p<changed.size();p+=changed[p]>>16)if((changed[p]&65535)==spv::OpFAdd){changed[p+3]^=1;break;}
        bool refused=false;try{native_q8_validate_observation_effective(code.data(),code.size(),changed.data(),changed.size(),flags&1,flags&2,nullptr);}catch(...){refused=true;}
        need(refused,"QSA score/select effective arithmetic mutation accepted");
    }
}
int main(int argc,char**argv){try{
    native_hc_need(argc<=2,"effective module arguments");
    std::string prefix=argc==2?argv[1]:"";
    audit(strata_native_qsa_scores_t1_spv,sizeof(strata_native_qsa_scores_t1_spv)/4,prefix.empty()?"":prefix+".scores",3,4);
    audit(strata_native_qsa_select_t1_spv,sizeof(strata_native_qsa_select_t1_spv)/4,prefix.empty()?"":prefix+".select",0,1);
    puts("native QSA score/select effective modules, protections and exact helper transformations pass; no GPU execution");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
