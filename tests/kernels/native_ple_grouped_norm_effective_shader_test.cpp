#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#ifdef STRATA_PLE_NORM_RCP_FMA_OBSERVATION
#include "strata_native_ple_grouped_norm_rcp_fma_t1_spv.hpp"
#define strata_native_ple_grouped_norm_t1_spv strata_native_ple_grouped_norm_rcp_fma_t1_spv
#endif
#ifndef STRATA_PLE_NORM_RCP_FMA_OBSERVATION
#include "strata_native_ple_grouped_norm_t1_spv.hpp"
#endif
#include "strata_native_observe_test_helper.hpp"
#include <cstdio>
#include <set>
#include <map>
#include <stdexcept>
#include <string>
using namespace strata::core::portable_detail;
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv){try{
 auto code=native_component_float_controls(strata_native_ple_grouped_norm_t1_spv,sizeof(strata_native_ple_grouped_norm_t1_spv)/4,NativeComponentAddressing::descriptor);
 size_t dots=0,fmas=0,modes=0,barriers=0,divs=0,muls=0,conversions=0;bool geometry=false;std::set<uint32_t> precise;
 for(size_t p=5;p<code.size();p+=code[p]>>16){const auto n=code[p]>>16,op=code[p]&65535;need(n&&p+n<=code.size(),"PLE grouped norm malformed module");
  if(op==spv::OpDecorate&&code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
  dots+=op==spv::OpSDot;fmas+=op==4427;barriers+=op==spv::OpControlBarrier;
  divs+=op==spv::OpFDiv;muls+=op==spv::OpFMul;conversions+=op==spv::OpConvertUToF;
  if(op==spv::OpExecutionMode&&n==6&&code[p+2]==spv::ExecutionModeLocalSize)
   geometry=code[p+3]==1024&&code[p+4]==1&&code[p+5]==1;
  if(op==spv::OpFAdd||op==spv::OpFMul||op==spv::OpFDiv)need(precise.count(code[p+2]),"PLE grouped norm arithmetic can contract");
  if(op==spv::OpExecutionMode&&n>3&&code[p+3]==32){need(code[p+2]!=spv::ExecutionModeDenormPreserve&&code[p+2]!=spv::ExecutionModeDenormFlushToZero,"PLE grouped norm unexpected denorm policy");modes+=code[p+2]==spv::ExecutionModeRoundingModeRTE||code[p+2]==spv::ExecutionModeSignedZeroInfNanPreserve;}
 }
 #ifdef STRATA_PLE_NORM_RCP_FMA_OBSERVATION
 need(dots==0&&fmas==2&&divs==1&&muls==3&&conversions==1&&geometry&&modes==2&&barriers>=7,"PLE norm reciprocal/FMA operations differ");
 std::map<uint32_t,uint32_t> constants;uint32_t reciprocal=0,divisor=0;bool fused=false;
 for(size_t p=5;p<code.size();p+=code[p]>>16){auto op=code[p]&65535;
  if(op==spv::OpConstant&&(code[p]>>16)==4)constants[code[p+2]]=code[p+3];
  if(op==spv::OpConvertUToF)divisor=code[p+2];
 }
 for(size_t p=5;p<code.size();p+=code[p]>>16)if((code[p]&65535)==spv::OpFDiv){
  need(constants[code[p+3]]==0x3f800000u&&code[p+4]==divisor,"reciprocal must be one / runtime columns");reciprocal=code[p+2];
 }
 for(size_t p=5;p<code.size();p+=code[p]>>16)if((code[p]&65535)==4427&&code[p+4]==reciprocal){
  need(constants[code[p+5]]==0x358637bdu,"fused epsilon differs");fused=true;
 }
 need(reciprocal&&fused,"reciprocal must feed explicit sum * reciprocal + epsilon FMA");
#else
 need(dots==0&&fmas==1&&divs==1&&muls==2&&conversions==1&&geometry&&modes==2&&barriers>=7,"PLE grouped norm effective operations differ");
#endif
 for(unsigned flags=0;flags<4;++flags){auto effective=native_q8_test_helper_effective(code,flags&1,flags&2);
  const std::string path=argc==2?std::string(argv[1])+"."+std::to_string(flags)+".spv":"";
  native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),flags&1,flags&2,argc==2?path.c_str():nullptr);
  auto changed=effective;for(size_t p=5;p<changed.size();p+=changed[p]>>16)if((changed[p]&65535)==4427){changed[p+3]^=1;break;}
  bool refused=false;try{native_q8_validate_observation_effective(code.data(),code.size(),changed.data(),changed.size(),flags&1,flags&2,nullptr);}catch(...){refused=true;}need(refused,"PLE grouped norm effective FMA mutation accepted");
 }
 std::puts("PLE grouped norm effective modules and FMA tamper refusal passed; no GPU execution");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
