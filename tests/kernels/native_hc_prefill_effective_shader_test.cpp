#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_hc_prefill_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include <stdexcept>
#include <cstdio>
using namespace strata::core::portable_detail;
int main(){try{
 auto code=native_component_float_controls(strata_native_hc_prefill_spv,sizeof(strata_native_hc_prefill_spv)/4,NativeComponentAddressing::descriptor);
 size_t fma=0;for(size_t i=5;i<code.size();i+=code[i]>>16){if(!(code[i]>>16)||i+(code[i]>>16)>code.size())throw std::runtime_error("malformed SPIR-V");fma+=(code[i]&65535)==4427;}
 if(fma<3)throw std::runtime_error("expected square and channel-order FMA operations");
 for(unsigned flags=0;flags<4;++flags){auto effective=native_q8_test_helper_effective(code,flags&1,flags&2);native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),flags&1,flags&2,nullptr);}
 puts("HC prefill effective shader transformations passed; no device arithmetic");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
