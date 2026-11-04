#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_ple_gate_t1_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
using namespace strata::core::portable_detail;
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv){try{
 auto code=native_component_float_controls(strata_native_ple_gate_t1_spv,sizeof(strata_native_ple_gate_t1_spv)/4,NativeComponentAddressing::descriptor);
 size_t dots=0,fmas=0,modes=0,barriers=0;std::set<uint32_t> precise;
 for(size_t p=5;p<code.size();p+=code[p]>>16){const auto n=code[p]>>16,op=code[p]&65535;need(n&&p+n<=code.size(),"PLE gate malformed module");
  if(op==spv::OpDecorate&&code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
  dots+=op==spv::OpSDot;fmas+=op==4427;barriers+=op==spv::OpControlBarrier;
  if(op==spv::OpFAdd||op==spv::OpFMul||op==spv::OpFDiv)need(precise.count(code[p+2]),"PLE gate arithmetic can contract");
  if(op==spv::OpExecutionMode&&n>3&&code[p+3]==32){need(code[p+2]!=spv::ExecutionModeDenormPreserve&&code[p+2]!=spv::ExecutionModeDenormFlushToZero,"PLE gate unexpected denorm policy");modes+=code[p+2]==spv::ExecutionModeRoundingModeRTE||code[p+2]==spv::ExecutionModeSignedZeroInfNanPreserve;}
 }
 need(dots==0&&fmas==1&&modes==2&&barriers>=7,"PLE gate effective operations differ");
 for(unsigned flags=0;flags<4;++flags){auto effective=native_q8_test_helper_effective(code,flags&1,flags&2);
  const std::string path=argc==2?std::string(argv[1])+"."+std::to_string(flags)+".spv":"";
  native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),flags&1,flags&2,argc==2?path.c_str():nullptr);
  auto changed=effective;for(size_t p=5;p<changed.size();p+=changed[p]>>16)if((changed[p]&65535)==4427){changed[p+3]^=1;break;}
  bool refused=false;try{native_q8_validate_observation_effective(code.data(),code.size(),changed.data(),changed.size(),flags&1,flags&2,nullptr);}catch(...){refused=true;}need(refused,"PLE gate effective FMA mutation accepted");
 }
 std::puts("PLE gate effective modules and FMA tamper refusal passed; no GPU execution");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
