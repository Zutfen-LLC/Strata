#include "native_q8_gu_spirv.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_moe_combine_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include "native_q8_fixture.hpp"
#include <set>
#include <map>
using namespace strata::core::portable_detail;
using native_q8_fixture::need;
int main(int argc,char** argv){try {
    auto code=q8_gu_float_controls(strata_native_moe_combine_spv,sizeof(strata_native_moe_combine_spv)/4);
    size_t first=0,fma=0,shared=0,loops=0;std::set<uint32_t> precise;
    std::map<uint32_t,std::vector<uint32_t>> definitions;
    uint32_t first_id=0,fma_id=0,accumulator=0,shared_id=0,vector_type=0,loop_index=0,loop_bound=0;
    for(size_t p=5;p<code.size();p+=code[p]>>16) {
        const uint32_t n=code[p]>>16,op=code[p]&65535;
        need(n && p+n<=code.size(),"combine malformed shader");
        if(op==spv::OpDecorate && code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
        if(op==spv::OpVectorTimesScalar){++first;first_id=code[p+2];vector_type=code[p+1];need(precise.count(first_id),"combine first product not separately rounded");}
        if(op==4427){++fma;fma_id=code[p+2];accumulator=code[p+5];need(code[p+1]==vector_type,"combine FMA vector type changed");} // OpFmaKHR
        if(op==spv::OpFAdd){++shared;shared_id=code[p+2];need(code[p+3]==accumulator,"combine shared row added before ordered FMA loop");need(precise.count(shared_id),"combine shared add can contract");}
        if(op==spv::OpLoopMerge)++loops;
        if(op==spv::OpPhi || op==spv::OpIAdd || op==spv::OpConstant)
            definitions[code[p+2]]=std::vector<uint32_t>(code.begin()+p,code.begin()+p+n);
        if(op==spv::OpTypeVector || op==spv::OpTypeFloat)
            definitions[code[p+1]]=std::vector<uint32_t>(code.begin()+p,code.begin()+p+n);
        if(op==spv::OpULessThan){loop_index=code[p+3];loop_bound=code[p+4];}
        need(!(op==spv::OpExtInst && code[p+4]==50),"combine contains legacy Fma");
        need(op!=spv::OpFDiv && op!=spv::OpFMul && op!=spv::OpFSub,"combine introduced other FP arithmetic");
        need(!(op==spv::OpDecorate && code[p+2]==spv::DecorationFPFastMathMode),"combine fast math mode");
        need(!(op>=spv::OpGroupNonUniformElect && op<=spv::OpGroupNonUniformQuadSwap),"combine subgroup reduction introduced");
    }
    need(first==1 && fma==1 && shared==1 && loops==1,"combine native arithmetic sequence changed");
    const auto& sum_phi=definitions.at(accumulator);
    need((sum_phi[0]&65535)==spv::OpPhi && sum_phi.size()==7 && sum_phi[3]==first_id && sum_phi[5]==fma_id,
        "combine accumulator does not start at rounded product and carry strict FMA results");
    const auto& index_phi=definitions.at(loop_index);const auto& bound=definitions.at(loop_bound);
    need((index_phi[0]&65535)==spv::OpPhi && index_phi.size()==7 && bound.size()==4 && bound[3]==10,
        "combine loop bound is not K10");
    need(definitions.at(index_phi[3])[3]==1,"combine FMA loop does not start at expert 1");
    const auto& increment=definitions.at(index_phi[5]);
    need((increment[0]&65535)==spv::OpIAdd && increment.size()==5 && increment[3]==loop_index &&
        definitions.at(increment[4])[3]==1,"combine expert order is not increasing by one");
    const auto& vector=definitions.at(vector_type);
    need((vector[0]&65535)==spv::OpTypeVector && vector.size()==4 && vector[3]==4 &&
        (definitions.at(vector[2])[0]&65535)==spv::OpTypeFloat && definitions.at(vector[2])[2]==32,
        "combine is not four native FP32 lanes");
    for(int mask=0;mask<4;++mask) {
        auto effective=native_q8_test_helper_effective(code,mask&1,mask&2);
        const std::string path=argc==2?std::string(argv[1])+"."+std::to_string(mask)+".spv":"";
        native_q8_validate_observation_effective(code.data(),code.size(),effective.data(),effective.size(),mask&1,mask&2,argc==2?path.c_str():nullptr);
        auto altered=effective;for(size_t p=5;p<altered.size();p+=altered[p]>>16)if((altered[p]&65535)==4427){altered[p+3]^=1;break;}
        bool rejected=false;try{native_q8_validate_observation_effective(code.data(),code.size(),altered.data(),altered.size(),mask&1,mask&2,nullptr);}catch(...){rejected=true;}
        need(rejected,"combine effective FMA operand mutation accepted");
    }
    puts("native K10 combine shader order/contraction and exact pinned helper transformations checked; no GPU execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
