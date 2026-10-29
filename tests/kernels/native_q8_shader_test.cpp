#include "native_q8_spirv.hpp"
#include "strata_native_stages_spv.hpp"
#include "native_q8_fixture.hpp"
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    auto words=native_stage_float_controls(strata_native_stages_spv,sizeof(strata_native_stages_spv)/4);
    size_t divides=0,contractions=0,controls=0,barriers=0;
    for(size_t pos=5;pos<words.size();){uint32_t len=words[pos]>>16,op=words[pos]&65535;
        need(len && pos+len<=words.size(),"bad instruction");
        if(op==spv::OpFDiv)++divides;
        if(op==spv::OpControlBarrier)++barriers;
        if(op==spv::OpDecorate && words[pos+2]==spv::DecorationNoContraction)++contractions;
        if(op==spv::OpExecutionMode && len==4 && words[pos+3]==32 &&
            (words[pos+2]==spv::ExecutionModeDenormPreserve || words[pos+2]==spv::ExecutionModeSignedZeroInfNanPreserve ||
             words[pos+2]==spv::ExecutionModeRoundingModeRTE))++controls;
        // Logical warp must not silently become a physical subgroup operation.
        need(!(op>=spv::OpGroupNonUniformElect && op<=spv::OpGroupNonUniformQuadSwap),"physical subgroup instruction");
        pos+=len;}
    need(divides==3 && contractions>=7 && controls==3 && barriers>=3,"native shader arithmetic/controls missing");
    if(argc==2){std::ofstream o(argv[1],std::ios::binary);for(uint32_t w:words)word(o,w);need(bool(o),"SPIRV write failed");}
    else need(argc==1,"optional argument: controlled SPIRV output path");
    std::puts("native shader: division, NoContraction, FP32 controls and logical-warp barriers checked; not GPU proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
