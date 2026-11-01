#pragma once
#include "native_q8_gu_spirv.hpp"
namespace strata::core::portable_detail {
inline std::vector<uint32_t> llvm_log1p_modes(const uint32_t*code,size_t count){std::vector<uint32_t>v(code,code+count);uint32_t entry=0;size_t at=0;bool fp64=false;
 for(size_t p=5;p<v.size();p+=v[p]>>16){native_spirv_need((v[p]>>16)&&p+(v[p]>>16)<=v.size(),"LLVM log1p malformed SPIR-V");if((v[p]&65535)==spv::OpCapability)fp64|=v[p+1]==spv::CapabilityFloat64;if((v[p]&65535)==spv::OpEntryPoint){entry=v[p+2];at=p+(v[p]>>16);}if((v[p]&65535)==spv::OpExecutionMode)at=p+(v[p]>>16);}
 native_spirv_need(entry&&fp64,"LLVM log1p FP64 module required");
 for(uint32_t mode:{uint32_t(spv::ExecutionModeRoundingModeRTE),uint32_t(spv::ExecutionModeSignedZeroInfNanPreserve)}){const uint32_t a[]={(4u<<16)|spv::OpExecutionMode,entry,mode,64};v.insert(v.begin()+at,std::begin(a),std::end(a));}return v;
}
template<class P>inline void llvm_log1p_require(bool physical,bool enabled,const P&p){native_spirv_need(physical&&enabled,"LLVM log1p: physical and logically enabled shaderFloat64 required");native_spirv_need(p.shaderRoundingModeRTEFloat64&&p.shaderSignedZeroInfNanPreserveFloat64,"LLVM log1p: FP64 RTE and signed-zero/Inf/NaN required");}
}
