#pragma once
#include "native_q8_spirv.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
namespace strata::core::portable_detail {
inline bool native_q8_observation_requested(int argc,char** argv) {
    return argc==10 && std::string(argv[1])=="--observe-unspecified-fp32-denorm";
}
inline std::vector<uint32_t> native_q8_observation_code(const std::vector<uint32_t>& strict) {
    auto result=strict;size_t removed=0;
    for(size_t pos=5;pos<result.size();) {
        const uint32_t size=result[pos]>>16,op=result[pos]&65535;
        native_spirv_need(size && pos+size<=result.size(),"observation: malformed module");
        if(op==spv::OpExecutionMode && size==4 && result[pos+2]==spv::ExecutionModeDenormPreserve && result[pos+3]==32) {
            result.erase(result.begin()+pos,result.begin()+pos+size);++removed;
        } else pos+=size;
    }
    native_spirv_need(removed==1,"observation: expected exactly one DenormPreserve32 mode");
    return result;
}
template<class FloatControls>
void native_q8_require_observation_controls(const FloatControls& fp) {
    native_spirv_need(fp.shaderSignedZeroInfNanPreserveFloat32 && fp.shaderRoundingModeRTEFloat32,
        "observation: remaining FP32 signed-zero/Inf/NaN and RTE controls required");
    // The unchanged module still declares the DenormPreserve capability. At
    // least one supported width is required even though its FP32 mode is gone.
    native_spirv_need(fp.shaderDenormPreserveFloat16 || fp.shaderDenormPreserveFloat32 || fp.shaderDenormPreserveFloat64,
        "observation: remaining DenormPreserve capability unsupported at every width");
    // The pinned helper adds DenormPreserve16 when supported. Without FP32's
    // mode, those controls must be independently selectable.
    native_spirv_need(!fp.shaderDenormPreserveFloat16 || uint32_t(fp.denormBehaviorIndependence)!=2,
        "observation: helper FP16 preservation cannot be independent of unspecified FP32");
}
// Called by an observation-only copy of the exact pinned pipeline helper,
// immediately before vkCreateShaderModule, AFTER all helper transformations.
// Every input instruction must survive byte-for-byte in order. Only the pinned
// helper's documented FP16 metadata insertions may be added, with exact counts.
inline void native_q8_validate_observation_effective(const uint32_t* input,size_t input_count,
    const uint32_t* effective,size_t effective_count,bool rte16,bool preserve16,const char* path) {
    native_spirv_need(input_count>=5 && effective_count>=5 &&
        std::equal(input,input+5,effective),"observation: effective shader header changed");
    // Validate the observation mode set before inspecting helper insertions.
    uint32_t entry=0;size_t signed32=0,rte32=0;
    for(size_t p=5;p<input_count;){uint32_t n=input[p]>>16,op=input[p]&65535;
        native_spirv_need(n && p+n<=input_count,"observation: malformed input");
        if(op==spv::OpEntryPoint){native_spirv_need(!entry && n>=4,"observation: invalid entry");entry=input[p+2];}
        if(op==spv::OpExecutionMode && n==4 && input[p+3]==32){
            native_spirv_need(input[p+2]!=spv::ExecutionModeDenormPreserve &&
                input[p+2]!=spv::ExecutionModeDenormFlushToZero,"observation: FP32 denorm mode must be unspecified");
            signed32+=input[p+2]==spv::ExecutionModeSignedZeroInfNanPreserve;
            rte32+=input[p+2]==spv::ExecutionModeRoundingModeRTE;
        }p+=n;}
    native_spirv_need(entry && signed32==1 && rte32==1,"observation: required FP32 modes missing");
    std::vector<std::vector<uint32_t>> additions;
    if(rte16)additions.push_back({(4u<<16)|spv::OpExecutionMode,entry,spv::ExecutionModeRoundingModeRTE,16});
    if(preserve16)additions.push_back({(4u<<16)|spv::OpExecutionMode,entry,spv::ExecutionModeDenormPreserve,16});
    if(rte16 || preserve16){
        const char ext[]="SPV_KHR_float_controls";
        std::vector<uint32_t> e(1+(sizeof(ext)+3)/4,0);e[0]=(uint32_t(e.size())<<16)|spv::OpExtension;
        std::memcpy(e.data()+1,ext,sizeof(ext));additions.push_back(e);
    }
    if(rte16)additions.push_back({(2u<<16)|spv::OpCapability,spv::CapabilityRoundingModeRTE});
    if(preserve16)additions.push_back({(2u<<16)|spv::OpCapability,spv::CapabilityDenormPreserve});
    size_t at=5;
    for(size_t p=5;p<effective_count;){uint32_t n=effective[p]>>16;
        native_spirv_need(n && p+n<=effective_count,"observation: malformed effective module");
        if(at+n<=input_count && std::equal(effective+p,effective+p+n,input+at))at+=n;
        else {
            auto extra=std::find(additions.begin(),additions.end(),std::vector<uint32_t>(effective+p,effective+p+n));
            native_spirv_need(extra!=additions.end(),"observation: helper changed an instruction outside allowed FP16 metadata");
            additions.erase(extra);
        }
        p+=n;
    }
    native_spirv_need(at==input_count && additions.empty(),"observation: effective module lost instructions or metadata");
    if(path){std::ofstream out(path,std::ios::binary);native_spirv_need(bool(out),"observation: cannot create effective SPIR-V receipt");
        out.write(reinterpret_cast<const char*>(effective),effective_count*4);native_spirv_need(bool(out),"observation: effective SPIR-V write failed");}
}
}
