#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_q8_wo_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include <set>
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try{
 auto code=native_component_float_controls(strata_q8_wo_spv,sizeof(strata_q8_wo_spv)/4,NativeComponentAddressing::descriptor);
 size_t dots=0,fmas=0,adds=0,muls=0;std::set<uint32_t> precise;
 for(size_t p=5;p<code.size();p+=code[p]>>16){auto n=code[p]>>16,op=code[p]&65535;native_spirv_need(n&&p+n<=code.size(),"WO malformed module");
  if(op==spv::OpDecorate&&n>=3&&code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
  dots+=op==spv::OpSDot;fmas+=op==4427;
  if(op==spv::OpFAdd){++adds;native_spirv_need(precise.count(code[p+2]),"WO reduction can contract");}
  if(op==spv::OpFMul){++muls;native_spirv_need(precise.count(code[p+2]),"WO scale can contract");}
 }
 native_spirv_need(dots==2&&fmas==1&&adds==4&&muls==1,"WO operation profile differs");
 for(unsigned flags=0;flags<4;++flags){auto effective=native_q8_test_helper_effective(code,flags&1,flags&2);
  std::string path=argc==2?std::string(argv[1])+"."+std::to_string(flags)+".spv":"";
  native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),flags&1,flags&2,path.empty()?nullptr:path.c_str());
  auto changed=effective;bool found=false;for(size_t p=5;p<changed.size();p+=changed[p]>>16)if((changed[p]&65535)==4427){changed[p+3]^=1;found=true;break;}
  native_spirv_need(found,"WO FMA missing");bool refused=false;try{native_q8_validate_observation_effective(code.data(),code.size(),changed.data(),changed.size(),flags&1,flags&2,nullptr);}catch(...){refused=true;}
  native_spirv_need(refused,"WO effective FMA mutation accepted");
 }
 native_spirv_need(argc==1||argc==2,"WO effective test arguments");puts("WO effective modules and exact helper transformation pass; no GPU");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
