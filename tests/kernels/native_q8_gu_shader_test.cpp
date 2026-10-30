#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#ifdef STRATA_TEST_STRICT_FMA
#include "strata_q8_gu_strict_fma_spv.hpp"
#define strata_q8_gu_spv strata_q8_gu_strict_fma_spv
#else
#include "strata_q8_gu_spv.hpp"
#endif
#include "strata_native_observe_test_helper.hpp"
#include "native_q8_fixture.hpp"
#include <set>
using namespace strata::core::portable_detail;
using native_q8_fixture::need;
int main(int argc,char** argv){try {
    auto code=q8_gu_float_controls(strata_q8_gu_spv,sizeof(strata_q8_gu_spv)/4);
    // Missing BDA capability must fail before pipeline creation.
    auto no_bda=code;
    for(size_t p=5;p<no_bda.size();p+=no_bda[p]>>16)
        if((no_bda[p]&65535)==spv::OpCapability && no_bda[p+1]==spv::CapabilityPhysicalStorageBufferAddresses) {
            no_bda.erase(no_bda.begin()+p,no_bda.begin()+p+2);break;
        }
    bool refused_bda=false;try{q8_gu_float_controls(no_bda.data(),no_bda.size());}catch(...){refused_bda=true;}
    need(refused_bda,"missing physical address capability accepted");
    size_t dots=0,fmas=0,muls=0,adds=0,barriers=0,modes=0;std::set<uint32_t> precise;
    for(size_t p=5;p<code.size();p+=code[p]>>16){const auto op=code[p]&65535;
        need(code[p]>>16 && p+(code[p]>>16)<=code.size(),"GU malformed shader");
        if(op==spv::OpDecorate && code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
        if(op==spv::OpSDot)++dots;
#ifdef STRATA_TEST_STRICT_FMA
        // Read the registry opcode; no binary instruction is hand-encoded.
        if(op==4427)++fmas; // SPV_KHR_fma OpFmaKHR
        need(!(op==spv::OpExtInst && code[p+4]==50),"strict shader contains legacy Fma");
#else
        if(op==spv::OpExtInst && code[p+4]==50)++fmas; // GLSL.std.450 Fma
#endif
        if(op==spv::OpFMul){++muls;need(precise.count(code[p+2]),"GU scale multiply can contract");}
        if(op==spv::OpFAdd){++adds;need(precise.count(code[p+2]),"GU reduction can contract");}
        if(op==spv::OpControlBarrier)++barriers;
        need(op!=spv::OpFDiv,"GU division introduced");
        need(!(op>=spv::OpGroupNonUniformElect && op<=spv::OpGroupNonUniformQuadSwap),"GU physical subgroup op");
        if(op==spv::OpExecutionMode && code[p+3]==32){
            need(code[p+2]!=spv::ExecutionModeDenormPreserve && code[p+2]!=spv::ExecutionModeDenormFlushToZero,"GU FP32 denorm policy introduced");
            modes+=code[p+2]==spv::ExecutionModeRoundingModeRTE || code[p+2]==spv::ExecutionModeSignedZeroInfNanPreserve;
        }
    }
    need(dots==2 && fmas==1 && muls==2 && adds==2 && barriers>=4 && modes==2,"GU arithmetic mapping changed");
    for(unsigned mask=0;mask<8;++mask){vk::PhysicalDeviceVulkan12Properties fp;
        fp.shaderSignedZeroInfNanPreserveFloat32=mask&1?VK_TRUE:VK_FALSE;
        fp.shaderRoundingModeRTEFloat32=mask&2?VK_TRUE:VK_FALSE;
        fp.shaderDenormPreserveFloat32=mask&4?VK_TRUE:VK_FALSE;
        bool pass=true;try{q8_gu_require_controls(fp);}catch(...){pass=false;}
        need(pass==((mask&3)==3),"GU incorrectly requires FP32 denorm preservation");
    }
    {vk::PhysicalDeviceVulkan12Properties fp;
        fp.shaderSignedZeroInfNanPreserveFloat32=fp.shaderRoundingModeRTEFloat32=VK_TRUE;
        fp.shaderDenormPreserveFloat16=VK_TRUE;fp.denormBehaviorIndependence=vk::ShaderFloatControlsIndependence::eNone;
        bool rejected=false;try{q8_gu_require_controls(fp);}catch(...){rejected=true;}
        need(rejected,"GU incompatible helper denorm independence accepted");
    }
    for(int bits=0;bits<4;++bits){auto effective=native_q8_test_helper_effective(code,bits&1,bits&2);
        std::string path=argc==2?std::string(argv[1])+"."+std::to_string(bits)+".spv":"";
        native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),bits&1,bits&2,argc==2?path.c_str():nullptr);
        auto altered=effective;for(size_t p=5;p<altered.size();p+=altered[p]>>16)
            if((altered[p]&65535)==spv::OpFMul){altered[p+3]^=1;break;}
        bool rejected=false;try{native_q8_validate_observation_effective(code.data(),code.size(),altered.data(),altered.size(),bits&1,bits&2,nullptr);}catch(...){rejected=true;}
        need(rejected,"GU final helper arithmetic mutation accepted");
    }
    puts("GU shader arithmetic, logical warps and effective helper modules checked; no GPU execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
