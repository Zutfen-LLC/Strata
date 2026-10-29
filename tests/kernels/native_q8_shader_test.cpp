#include "native_q8_spirv.hpp"
#include "native_q8_float_controls.hpp"
#include "native_q8_observation.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_stages_spv.hpp"
#include "native_q8_fixture.hpp"
#include "strata_native_observe_test_helper.hpp"
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
namespace {
uint32_t property_mask=0;
bool query_chain_valid=false;
VKAPI_ATTR void VKAPI_CALL probe_properties(VkPhysicalDevice device,VkPhysicalDeviceProperties2* p) {
    query_chain_valid=device==reinterpret_cast<VkPhysicalDevice>(uintptr_t(0x1234)) &&
        p->sType==VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 && p->pNext;
    if(!query_chain_valid)return;
    auto* fp=static_cast<VkPhysicalDeviceVulkan12Properties*>(p->pNext);
    query_chain_valid=fp->sType==VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES && !fp->pNext;
    if(!query_chain_valid)return;
    fp->shaderDenormPreserveFloat32=property_mask&1?VK_TRUE:VK_FALSE;
    fp->shaderSignedZeroInfNanPreserveFloat32=property_mask&2?VK_TRUE:VK_FALSE;
    fp->shaderRoundingModeRTEFloat32=property_mask&4?VK_TRUE:VK_FALSE;
    fp->shaderDenormFlushToZeroFloat32=VK_TRUE;fp->shaderRoundingModeRTZFloat32=VK_FALSE;
    fp->denormBehaviorIndependence=VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY;
    fp->roundingModeIndependence=VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL;
    fp->shaderDenormPreserveFloat16=VK_FALSE;fp->shaderRoundingModeRTEFloat16=VK_TRUE;
    p->properties.apiVersion=VK_API_VERSION_1_2;p->properties.driverVersion=123;
    p->properties.vendorID=456;p->properties.deviceID=789;
    std::strcpy(p->properties.deviceName,"mock property query; no GPU");
}
struct ProbeDispatch {
    uint32_t getVkHeaderVersion()const{return VK_HEADER_VERSION;}
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2=probe_properties;
};
void property_diagnostics() {
    vk::PhysicalDevice device(reinterpret_cast<VkPhysicalDevice>(uintptr_t(0x1234)));
    for(property_mask=0;property_mask<8;++property_mask){
        // Exercise the same real Vulkan-Hpp overload and chain as the bridge,
        // with a mock dispatcher. This checks plumbing, not a driver's values.
        const auto properties=device.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceVulkan12Properties>(ProbeDispatch{});
        need(query_chain_valid,"Vulkan-Hpp property chain/physical handle incorrect");
        const auto& fp=properties.get<vk::PhysicalDeviceVulkan12Properties>();
        const auto& base=properties.get<vk::PhysicalDeviceProperties2>();
        need(base.pNext==&fp,"returned Vulkan-Hpp chain pointer incorrect");
        need(fp.shaderDenormPreserveFloat32==uint32_t(bool(property_mask&1)) &&
             fp.shaderSignedZeroInfNanPreserveFloat32==uint32_t(bool(property_mask&2)) &&
             fp.shaderRoundingModeRTEFloat32==uint32_t(bool(property_mask&4)),"queried property values lost");
        bool accepted=true;std::string error;
        try{native_q8_require_float_controls(fp,base.properties);}catch(const std::runtime_error& e){accepted=false;error=e.what();}
        need(accepted==(property_mask==7),"diagnostic changed original three-property decision");
        if(!accepted){
            for(const auto& field:{std::pair{"shaderDenormPreserveFloat32=",bool(property_mask&1)},
                                   std::pair{"shaderSignedZeroInfNanPreserveFloat32=",bool(property_mask&2)},
                                   std::pair{"shaderRoundingModeRTEFloat32=",bool(property_mask&4)}})
                need(error.find(std::string(field.first)+(field.second?"1":"0"))!=std::string::npos,"missing raw required property");
            for(const char* field:{"shaderDenormFlushToZeroFloat32=1","shaderRoundingModeRTZFloat32=0",
                "denormBehaviorIndependence=0","roundingModeIndependence=1","shaderDenormPreserveFloat16=0",
                "shaderRoundingModeRTEFloat16=1","apiVersion=4202496","driverVersion=123","vendorID=456",
                "deviceID=789","deviceName=mock property query; no GPU"})
                need(error.find(field)!=std::string::npos,"missing raw context property");
        }
    }
}
}
namespace {
template<class F> void reject_observation(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}need(failed,"observation accepted invalid module/controls");}
void observation_tests(const std::vector<uint32_t>& strict,const char* prefix) {
    const auto observed=native_q8_observation_code(strict);
    need(observed.size()+4==strict.size(),"observation delta must be one four-word mode");
    reject_observation([&]{native_q8_observation_code(observed);});
    const char* tokens[]={"program","--observe-unspecified-fp32-denorm","--device","1","--expect-device","mock","--fixture","f","--output","o"};
    auto argv=const_cast<char**>(tokens);
    need(native_q8_observation_requested(10,argv) && !native_q8_observation_requested(9,argv),"observation must be explicit and complete");
    need(!native_q8_observation_requested(9,argv+1),"strict argv must not opt in");
    vk::PhysicalDeviceVulkan12Properties fp;
    fp.shaderSignedZeroInfNanPreserveFloat32=fp.shaderRoundingModeRTEFloat32=VK_TRUE;
    fp.shaderDenormPreserveFloat16=VK_TRUE;fp.denormBehaviorIndependence=vk::ShaderFloatControlsIndependence::eAll;
    native_q8_require_observation_controls(fp);
    auto invalid=fp;invalid.shaderSignedZeroInfNanPreserveFloat32=VK_FALSE;reject_observation([&]{native_q8_require_observation_controls(invalid);});
    invalid=fp;invalid.shaderRoundingModeRTEFloat32=VK_FALSE;reject_observation([&]{native_q8_require_observation_controls(invalid);});
    invalid=fp;invalid.shaderDenormPreserveFloat16=VK_FALSE;reject_observation([&]{native_q8_require_observation_controls(invalid);});
    invalid=fp;invalid.denormBehaviorIndependence=vk::ShaderFloatControlsIndependence::eNone;
    reject_observation([&]{native_q8_require_observation_controls(invalid);});
    for(int bits=0;bits<4;++bits){const bool rte=bits&1,preserve=bits&2;
        const auto effective=native_q8_test_helper_effective(observed,rte,preserve);
        const auto strict_effective=native_q8_test_helper_effective(strict,rte,preserve);
        need(native_q8_observation_code(strict_effective)==effective,"effective strict/observation delta changed instructions");
        std::string path=prefix?std::string(prefix)+".effective-"+std::to_string(bits)+".spv":"";
        native_q8_validate_observation_effective(observed.data(),observed.size(),effective.data(),effective.size(),rte,preserve,prefix?path.c_str():nullptr);
        auto altered=effective;
        for(size_t pos=5;pos<altered.size();pos+=altered[pos]>>16)if((altered[pos]&65535)==spv::OpFDiv){altered[pos+3]^=1;break;}
        reject_observation([&]{native_q8_validate_observation_effective(observed.data(),observed.size(),altered.data(),altered.size(),rte,preserve,nullptr);});
        reject_observation([&]{native_q8_validate_observation_effective(observed.data(),observed.size(),strict_effective.data(),strict_effective.size(),rte,preserve,nullptr);});
    }
}
}
int main(int argc,char** argv){try {
    property_diagnostics();
    auto words=native_stage_float_controls(strata_native_stages_spv,sizeof(strata_native_stages_spv)/4);
    size_t divides=0,contractions=0,controls=0,barriers=0;
    for(size_t pos=5;pos<words.size();){uint32_t len=words[pos]>>16,op=words[pos]&65535;
        need(len && pos+len<=words.size(),"bad instruction");
        if(op==spv::OpFDiv)++divides;
        if(op==spv::OpControlBarrier)++barriers;
        if(op==spv::OpDecorate && words[pos+2]==spv::DecorationNoContraction)++contractions;
        if(op==spv::OpExecutionMode && len==4 && words[pos+3]==32 &&
            (words[pos+2]==spv::ExecutionModeDenormPreserve || words[pos+2]==spv::ExecutionModeSignedZeroInfNanPreserve ||
             words[pos+2]==spv::ExecutionModeRoundingModeRTE))++controls;
        // Logical warp must not silently become a physical subgroup operation.
        need(!(op>=spv::OpGroupNonUniformElect && op<=spv::OpGroupNonUniformQuadSwap),"physical subgroup instruction");
        pos+=len;}
    need(divides==3 && contractions>=7 && controls==3 && barriers>=3,"native shader arithmetic/controls missing");
    const bool dump_observation=argc==3 && std::string(argv[1])=="--observation-modules";
    observation_tests(words,dump_observation?argv[2]:nullptr);
    if(dump_observation){} // effective module receipts were written by validation
    else if(argc==2){std::ofstream o(argv[1],std::ios::binary);for(uint32_t w:words)word(o,w);need(bool(o),"SPIRV write failed");}
    else need(argc==1,"optional argument: controlled SPIRV output path");
    std::puts("native shader: division, NoContraction, FP32 controls and logical-warp barriers checked; not GPU proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
