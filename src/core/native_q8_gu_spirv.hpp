#pragma once
#include "native_q8_observation.hpp"
namespace strata::core::portable_detail {
// The common pipeline owner also serves descriptor-only router/shared/combine
// modules. Addressing is a caller contract, separate from shared FP32 controls.
enum class NativeComponentAddressing { descriptor, physical64 };
inline std::vector<uint32_t> native_component_float_controls(const uint32_t* code,size_t count,
    NativeComponentAddressing addressing) {
    bool int64=false,physical=false;
    size_t memory_models=0;uint32_t memory_model=~uint32_t(0);
    for(size_t p=5;p<count;) {
        const uint32_t words=code[p]>>16,op=code[p]&65535;
        native_spirv_need(words && p+words<=count,"Q8 BDA: malformed module");
        if(op==spv::OpCapability && words==2) {
            int64|=code[p+1]==spv::CapabilityInt64;
            physical|=code[p+1]==spv::CapabilityPhysicalStorageBufferAddresses;
        }
        if(op==spv::OpMemoryModel && words==3){++memory_models;memory_model=code[p+1];}
        p+=words;
    }
    if(addressing==NativeComponentAddressing::physical64)
        native_spirv_need(int64 && physical && memory_models==1 && memory_model==spv::AddressingModelPhysicalStorageBuffer64,
            "Q8 BDA: 64-bit physical address ABI required");
    else {
        native_spirv_need(addressing==NativeComponentAddressing::descriptor,"native component: unknown addressing ABI");
        native_spirv_need(!physical && memory_models==1 && memory_model==spv::AddressingModelLogical,
            "native component: logical descriptor address ABI required");
    }
    // Reuse mode insertion but remove both the unused FP32 denorm mode and its
    // capability. This is a separate consumer, never a quantizer fallback.
    auto words=native_q8_observation_code(native_stage_float_controls(code,count));
    for(size_t p=5;p<words.size();p+=words[p]>>16)
        if((words[p]&65535)==spv::OpCapability && words[p+1]==spv::CapabilityDenormPreserve){
            words.erase(words.begin()+p,words.begin()+p+2);break;}
    return words;
}
inline std::vector<uint32_t> q8_gu_float_controls(const uint32_t* code,size_t count) {
    return native_component_float_controls(code,count,NativeComponentAddressing::physical64);
}
template<class Controls> void q8_gu_require_controls(const Controls& fp) {
    native_spirv_need(fp.shaderSignedZeroInfNanPreserveFloat32 && fp.shaderRoundingModeRTEFloat32,
        "Q8 GU: FP32 signed-zero/Inf/NaN and RTE required");
    native_spirv_need(!fp.shaderDenormPreserveFloat16 || uint32_t(fp.denormBehaviorIndependence)!=2,
        "Q8 GU: helper FP16 preservation must be independent of unspecified FP32");
}
}
