#include "../../src/core/native_q8_observation.hpp"
#include <cassert>
using namespace strata::core::portable_detail;
int main() {
    std::vector<uint32_t> in={0x07230203,0x10000,0,10,0,
        (4u<<16)|spv::OpEntryPoint,0,1,0,
        (4u<<16)|spv::OpExecutionMode,1,spv::ExecutionModeSignedZeroInfNanPreserve,32,
        (4u<<16)|spv::OpExecutionMode,1,spv::ExecutionModeRoundingModeRTE,32};
    auto check=[&](const std::vector<uint32_t>& e,bool r,bool d,bool ok) {
        bool success=true; try { native_q8_validate_observation_effective(in.data(),in.size(),e.data(),e.size(),r,d,nullptr); } catch (...) { success=false; }
        assert(success==ok);
    };
    for(int flags=0;flags<4;++flags) {
        auto e=in;
        if(flags&1) e.insert(e.end(),{(4u<<16)|spv::OpExecutionMode,1,spv::ExecutionModeRoundingModeRTE,16});
        if(flags&2) e.insert(e.end(),{(4u<<16)|spv::OpExecutionMode,1,spv::ExecutionModeDenormPreserve,16});
        if(flags) { const char ext[]="SPV_KHR_float_controls"; uint32_t words[1+(sizeof(ext)+3)/4]={}; words[0]=(uint32_t(sizeof(words)/4)<<16)|spv::OpExtension; std::memcpy(words+1,ext,sizeof(ext)); e.insert(e.end(),std::begin(words),std::end(words)); }
        if(flags&1) e.insert(e.end(),{(2u<<16)|spv::OpCapability,spv::CapabilityRoundingModeRTE});
        if(flags&2) e.insert(e.end(),{(2u<<16)|spv::OpCapability,spv::CapabilityDenormPreserve});
        check(e,flags&1,flags&2,true);
        auto bad=e; bad.push_back(0); check(bad,flags&1,flags&2,false);
        bad=e; bad[5]=0xffff0000; check(bad,flags&1,flags&2,false);
        bad=e; bad.pop_back(); check(bad,flags&1,flags&2,false);
        if(flags) { bad=e; bad.insert(bad.end(),e.end()-2,e.end()); check(bad,flags&1,flags&2,false); }
    }
}
