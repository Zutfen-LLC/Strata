#pragma once
#include <spirv/unified1/spirv.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strata::core::portable_detail {
inline void native_spirv_need(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
// Add mandatory FP32 execution modes. The pinned helper may additionally enable
// supported FP16 modes; this shader has no FP16 arithmetic or storage types.
inline std::vector<uint32_t> native_stage_float_controls(const uint32_t* code,size_t count) {
    std::vector<uint32_t> words(code,code+count);
    size_t caps=5,exec=0;uint32_t entry=0;
    for(size_t pos=5;pos<words.size();) {
        const uint32_t op=words[pos]&spv::OpCodeMask,len=words[pos]>>spv::WordCountShift;
        native_spirv_need(len>0 && pos+len<=words.size(),"native stages: malformed shader");
        if(op==spv::OpCapability)caps=pos+len;
        if(op==spv::OpEntryPoint){native_spirv_need(!entry,"native stages: multiple shader entries");entry=words[pos+2];exec=pos+len;}
        if(op==spv::OpExecutionMode || op==spv::OpExecutionModeId)exec=pos+len;
        pos+=len;
    }
    native_spirv_need(entry && exec>=caps,"native stages: missing shader entry");
    const uint32_t modes[]={spv::ExecutionModeDenormPreserve,spv::ExecutionModeSignedZeroInfNanPreserve,
                            spv::ExecutionModeRoundingModeRTE};
    for(uint32_t mode:modes){const uint32_t instruction[]={(4u<<16)|spv::OpExecutionMode,entry,mode,32};
        words.insert(words.begin()+exec,std::begin(instruction),std::end(instruction));}
    const uint32_t capabilities[]={spv::CapabilityDenormPreserve,spv::CapabilitySignedZeroInfNanPreserve,
                                   spv::CapabilityRoundingModeRTE};
    for(uint32_t cap:capabilities){const uint32_t instruction[]={(2u<<16)|spv::OpCapability,cap};
        words.insert(words.begin()+caps,std::begin(instruction),std::end(instruction));}
    return words;
}
}
