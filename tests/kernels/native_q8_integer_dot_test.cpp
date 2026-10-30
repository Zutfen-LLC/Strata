#include "native_q8_integer_dot.hpp"
#include <vulkan/vulkan.h>
#include <cstdio>
#include <vector>
using strata::core::portable_detail::NativeQ8IntegerDotState;
struct DotReceipt {NativeQ8IntegerDotState state;bool chained,accelerated;size_t extensions;};
#include "strata_native_q8_dot_test.hpp"
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try {
    for(bool compiled:{false,true})for(bool extension:{false,true})for(bool disabled:{false,true})
    for(bool accelerated:{false,true})for(bool supported:{false,true})for(bool created:{false,true}) {
        const bool request=compiled && extension && !disabled;
        const auto result=dot_feature_chain(request,accelerated,supported,created);
        need(result.chained==request && result.extensions==size_t(request),"feature chain/extension does not preserve admission/disable controls");
        need(result.accelerated==(request && accelerated),"GGML performance flag changed");
        need(result.state.enabled==(request && supported && created),"functional enablement incorrectly depends on acceleration or ignores creation failure");
        bool admitted=true;try{result.state.require_enabled(result.accelerated);}catch(...){admitted=false;}
        need(admitted==result.state.enabled,"component admission does not use logical enablement");
    }
    const auto vega=dot_feature_chain(true,false,true,true);
    need(vega.state.enabled && !vega.accelerated,"supported unaccelerated feature refused");
    puts("native Q8 dot: actual feature-chain block, unsupported/disabled/creation-failure and acceleration independence checked; no GPU execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
