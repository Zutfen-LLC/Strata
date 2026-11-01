#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_observe_test_helper.hpp"
#include "strata_q8_gu_spv.hpp"
#include "strata_q8_down_spv.hpp"
#include "strata_native_router_spv.hpp"
#ifdef STRATA_TEST_STRICT_FMA
#include "strata_q8_gu_strict_fma_spv.hpp"
#include "strata_q8_down_strict_fma_spv.hpp"
#include "strata_native_shared_spv.hpp"
#include "strata_native_moe_combine_spv.hpp"
#endif
#ifdef STRATA_TEST_ROUTER_TRACE
#include "strata_native_router_trace_spv.hpp"
#endif
#include <cstdio>
using namespace strata::core::portable_detail;
static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F> static void refused(F f,const char* diagnostic) {
    try{f();}catch(const std::exception& e){need(std::string(e.what())==diagnostic,"wrong ABI refusal");return;}
    throw std::runtime_error("wrong addressing ABI accepted");
}
static void check(const uint32_t* shader,size_t count,NativeComponentAddressing addressing,const std::string& path) {
    auto code=native_component_float_controls(shader,count,addressing);
    need(!code.empty(),"empty controlled shader");
    for(unsigned flags=0;flags<4;++flags) {
        auto effective=native_q8_test_helper_effective(code,flags&1,flags&2);
        const auto file=path.empty()?path:path+"."+std::to_string(flags)+".spv";
        native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),
            flags&1,flags&2,file.empty()?nullptr:file.c_str());
    }
    if(addressing==NativeComponentAddressing::descriptor) {
        // Reproduce the original common-factory misclassification, and require
        // that the physical64 guard still rejects these descriptor-only modules.
        refused([&]{q8_gu_float_controls(shader,count);},"Q8 BDA: 64-bit physical address ABI required");
    } else {
        refused([&]{native_component_float_controls(shader,count,NativeComponentAddressing::descriptor);},
            "native component: logical descriptor address ABI required");
        for(auto capability:{spv::CapabilityInt64,spv::CapabilityPhysicalStorageBufferAddresses}) {
            std::vector<uint32_t> changed(shader,shader+count);bool removed=false;
            for(size_t p=5;p<changed.size();p+=changed[p]>>16)
                if((changed[p]&65535)==spv::OpCapability && changed[p+1]==uint32_t(capability)) {
                    changed.erase(changed.begin()+p,changed.begin()+p+2);removed=true;break;
                }
            need(removed,"required physical64 capability absent in original module");
            refused([&]{q8_gu_float_controls(changed.data(),changed.size());},"Q8 BDA: 64-bit physical address ABI required");
        }
    }
    // Changing only the addressing model must fail for either caller contract.
    std::vector<uint32_t> changed(shader,shader+count);bool altered=false;
    for(size_t p=5;p<changed.size();p+=changed[p]>>16)
        if((changed[p]&65535)==spv::OpMemoryModel) {
            changed[p+1]=addressing==NativeComponentAddressing::descriptor ?
                spv::AddressingModelPhysicalStorageBuffer64:spv::AddressingModelLogical;
            altered=true;break;
        }
    need(altered,"original memory model absent");
    refused([&]{native_component_float_controls(changed.data(),changed.size(),addressing);},
        addressing==NativeComponentAddressing::descriptor ? "native component: logical descriptor address ABI required":
        "Q8 BDA: 64-bit physical address ABI required");
}
int main(int argc,char** argv){try{
#define CHECK(module,abi) check(module,sizeof(module)/4,NativeComponentAddressing::abi,argc==2?std::string(argv[1])+"."+#module:"")
    CHECK(strata_q8_gu_spv,physical64);CHECK(strata_q8_down_spv,physical64);
    CHECK(strata_native_router_spv,descriptor);
#ifdef STRATA_TEST_STRICT_FMA
    CHECK(strata_q8_gu_strict_fma_spv,physical64);CHECK(strata_q8_down_strict_fma_spv,physical64);
    CHECK(strata_native_shared_spv,descriptor);CHECK(strata_native_moe_combine_spv,descriptor);
#endif
#ifdef STRATA_TEST_ROUTER_TRACE
    CHECK(strata_native_router_trace_spv,descriptor);
#endif
#undef CHECK
    puts("native component compiled addressing contracts and capability/model refusal passed; no GPU execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
