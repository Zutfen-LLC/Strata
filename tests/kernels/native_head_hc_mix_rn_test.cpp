#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_head_hc_mix_rn_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include "native_hc_fixture.hpp"
#include <set>
#include <string>
namespace rn { using uint=uint32_t;
#include "../../src/kernels/vulkan/native_f32_div_rn.glsl"
}
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try{
 native_hc_need(rn::strata_f32_div_rn_bits(0x33c013a7u,0x3fffffffu)==0x334013a8u,"retained SiLU division differs");
 native_hc_need(rn::strata_f32_div_rn_bits(0xb3c013a7u,0x3fffffffu)==0xb34013a8u,"signed SiLU division differs");
 native_hc_need(rn::strata_f32_div_rn_bits(0u,0x3f800000u)==0u && rn::strata_f32_div_rn_bits(0x80000000u,0x3f800000u)==0x80000000u,"signed zero differs");
 auto code=native_component_float_controls(strata_native_head_hc_mix_rn_spv,sizeof(strata_native_head_hc_mix_rn_spv)/4,NativeComponentAddressing::descriptor);
 std::set<uint32_t> floats,ones,precise,adds;size_t divs=0,loops=0,fmas=0;
 for(size_t p=5;p<code.size();p+=code[p]>>16){const auto words=code[p]>>16,op=code[p]&65535;native_hc_need(words&&p+words<=code.size(),"malformed head mix module");
  if(op==spv::OpTypeFloat){native_hc_need(code[p+2]!=64,"unexpected float64");if(code[p+2]==32)floats.insert(code[p+1]);}
  if(op==spv::OpTypeInt)native_hc_need(code[p+2]!=64,"unexpected int64");
  if(op==spv::OpConstant && floats.count(code[p+1]) && words==4 && code[p+3]==0x3f800000u)ones.insert(code[p+2]);
  if(op==spv::OpDecorate && code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
  if(op==spv::OpFAdd)adds.insert(code[p+2]);
  loops+=op==spv::OpLoopMerge;fmas+=op==4427;
 }
 for(size_t p=5;p<code.size();p+=code[p]>>16)if((code[p]&65535)==spv::OpFDiv){++divs;native_hc_need(ones.count(code[p+3]) && adds.count(code[p+4]) && precise.count(code[p+2]),"only protected sigmoid reciprocal may use floating division");}
 native_hc_need(divs>0 && loops>=3 && fmas>0,"head candidate lost sigmoid, integer division or FMA structure");
 for(unsigned flags=0;flags<4;++flags){auto effective=native_q8_test_helper_effective(code,flags&1,flags&2);const std::string path=argc==2?std::string(argv[1])+"."+std::to_string(flags)+".spv":"";
  native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),flags&1,flags&2,argc==2?path.c_str():nullptr);}
 puts("head-only RNE SiLU retained operands and effective shader constraints pass; no GPU execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
