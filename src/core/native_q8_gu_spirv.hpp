#pragma once
#include "native_q8_observation.hpp"
namespace strata::core::portable_detail {
inline std::vector<uint32_t> q8_gu_float_controls(const uint32_t* code,size_t count) {
    bool int64=false,physical=false,memory_model=false;
    for(size_t p=5;p<count;) {
        const uint32_t words=code[p]>>16,op=code[p]&65535;
        native_spirv_need(words && p+words<=count,"Q8 BDA: malformed module");
        if(op==spv::OpCapability && words==2) {
            int64|=code[p+1]==spv::CapabilityInt64;
            physical|=code[p+1]==spv::CapabilityPhysicalStorageBufferAddresses;
        }
        if(op==spv::OpMemoryModel && words==3)memory_model=code[p+1]==spv::AddressingModelPhysicalStorageBuffer64;
        p+=words;
    }
    native_spirv_need(int64 && physical && memory_model,"Q8 BDA: 64-bit physical address ABI required");
    // Reuse mode insertion but remove both the unused FP32 denorm mode and its
    // capability. This is a separate consumer, never a quantizer fallback.
    auto words=native_q8_observation_code(native_stage_float_controls(code,count));
    for(size_t p=5;p<words.size();p+=words[p]>>16)
        if((words[p]&65535)==spv::OpCapability && words[p+1]==spv::CapabilityDenormPreserve){
            words.erase(words.begin()+p,words.begin()+p+2);break;}
    return words;
}
template<class Controls> void q8_gu_require_controls(const Controls& fp) {
    native_spirv_need(fp.shaderSignedZeroInfNanPreserveFloat32 && fp.shaderRoundingModeRTEFloat32,
        "Q8 GU: FP32 signed-zero/Inf/NaN and RTE required");
    native_spirv_need(!fp.shaderDenormPreserveFloat16 || uint32_t(fp.denormBehaviorIndependence)!=2,
        "Q8 GU: helper FP16 preservation must be independent of unspecified FP32");
}
}
