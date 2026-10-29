#pragma once
#include "native_q8_observation.hpp"
namespace strata::core::portable_detail {
inline std::vector<uint32_t> q8_gu_float_controls(const uint32_t* code,size_t count) {
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
