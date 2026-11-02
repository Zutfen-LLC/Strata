#include "native_q8_fixture.hpp"
#include "native_q8_observation.hpp"
#include "native_q8_rcp_shape.hpp"
#include "strata_native_stages_spv.hpp"
#include "strata_native_quant_rcp_spv.hpp"
#include "ggml.h"
#include <vulkan/vulkan.hpp>
#include "strata_native_observe_test_helper.hpp"
#include <cfenv>
#include <map>
#include <set>
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;

struct Op {uint32_t code;std::vector<uint32_t> words;};
std::vector<Op> ops(const std::vector<uint32_t>& code) {
    std::vector<Op> out;
    for(size_t p=5;p<code.size();) {const uint32_t n=code[p]>>16,op=code[p]&65535;
        need(n && p+n<=code.size(),"reciprocal observation: malformed SPIR-V");
        out.push_back({op,{code.begin()+p,code.begin()+p+n}});p+=n;}
    return out;
}
void shader_contract(const std::vector<uint32_t>& base,const std::vector<uint32_t>& candidate) {
    const auto a=ops(base),b=ops(candidate);std::map<uint32_t,size_t> ac,bc;
    std::map<uint32_t,size_t> all_a,all_b;
    for(const auto&x:a)++all_a[x.code];for(const auto&x:b)++all_b[x.code];
    ++all_a[spv::OpConstant];++all_a[spv::OpDecorate];all_a[spv::OpFMul]+=2;--all_a[spv::OpFDiv];
    need(all_a==all_b,"reciprocal observation: unexpected SPIR-V operation profile");
    auto counted=[](uint32_t op){return op==spv::OpFAdd||op==spv::OpFSub||op==spv::OpFMul||op==spv::OpFDiv||op==spv::OpFNegate||op==spv::OpExtInst||op==spv::OpConvertFToS;};
    for(const auto&x:a)if(counted(x.code))++ac[x.code];
    for(const auto&x:b)if(counted(x.code))++bc[x.code];
    need(ac[spv::OpFDiv]==3&&ac[spv::OpFMul]==1&&bc[spv::OpFDiv]==2&&bc[spv::OpFMul]==3,"reciprocal observation: unexpected divide/multiply count");
    ac.erase(spv::OpFDiv);ac.erase(spv::OpFMul);bc.erase(spv::OpFDiv);bc.erase(spv::OpFMul);
    need(ac==bc,"reciprocal observation: unrelated arithmetic changed");
    std::set<uint32_t> floats,precise;std::map<uint32_t,uint32_t> constants;
    std::map<uint32_t,Op> definitions;
    for(const auto&x:b){const auto&w=x.words;
        if(x.code==spv::OpTypeFloat && w.size()==3 && w[2]==32)floats.insert(w[1]);
        if(x.code==spv::OpDecorate && w.size()>=3 && w[2]==spv::DecorationNoContraction)precise.insert(w[1]);
    }
    for(const auto&x:b){const auto&w=x.words;
        if(x.code==spv::OpConstant&&w.size()==4&&floats.count(w[1]))constants[w[2]]=w[3];
        if((x.code==spv::OpFMul||x.code==spv::OpFDiv||x.code==spv::OpPhi||x.code==spv::OpSelect)&&w.size()>=4)definitions.emplace(w[2],x);
    }
    uint32_t scale=0,reciprocal=0,ratio=0,denom=0;unsigned scale_count=0,recip_count=0,ratio_count=0;
    for(const auto&x:b)if(x.code==spv::OpFMul && x.words.size()==5){const auto&w=x.words;
        if((constants.count(w[3])&&constants[w[3]]==0x3c010204u)||(constants.count(w[4])&&constants[w[4]]==0x3c010204u)){scale=w[2];++scale_count;}}
    need(scale_count==1 && precise.count(scale),"reciprocal observation: exact protected scale multiplication missing");
    for(const auto&x:b)if(x.code==spv::OpFDiv && x.words.size()==5){const auto&w=x.words;
        if(constants.count(w[3])&&constants[w[3]]==0x3f800000u){reciprocal=w[2];denom=w[4];++recip_count;}}
    need(recip_count==1&&precise.count(reciprocal),"reciprocal observation: protected 1/d missing");
    need(definitions.count(denom)&&definitions.at(denom).code==spv::OpPhi,"reciprocal observation: full F32 finite-scale phi missing");
    auto phi=definitions.at(denom).words;bool scale_input=false,clamp_input=false;
    for(size_t i=3;i+1<phi.size();i+=2){uint32_t value=phi[i];scale_input|=value==scale;
        if(definitions.count(value)&&definitions.at(value).code==spv::OpSelect){auto w=definitions.at(value).words;
            if(w.size()==6 && constants.count(w[4]) && constants.count(w[5])){
                auto p=constants[w[4]],q=constants[w[5]];
                clamp_input|=(p==0xc77fe000u&&q==0x477fe000u)||(q==0xc77fe000u&&p==0x477fe000u);}}}
    need(scale_input&&clamp_input,"reciprocal observation: full scale or finite clamp altered before reciprocal");
    for(const auto&x:b)if(x.code==spv::OpFMul&&x.words.size()==5){const auto&w=x.words;
        if(w[3]==reciprocal||w[4]==reciprocal){ratio=w[2];++ratio_count;}}
    need(ratio_count==1&&precise.count(ratio),"reciprocal observation: protected reciprocal-to-ratio multiply missing");
    bool ratio_consumed=false;for(const auto&x:b)if(x.code==spv::OpExtInst&&x.words.size()>=6&&x.words[5]==ratio)ratio_consumed=true;
    need(ratio_consumed,"reciprocal observation: ratio does not feed original quantization");
}
void boundaries() {
    need(std::fesetround(FE_TONEAREST)==0,"host nearest-even required");
    volatile float factor=value(0x3c010204u), nine=9.f;
    volatile float scale9=nine*factor,div9=nine/127.f;
    need(bits(scale9)==0x3d912244u&&bits(div9)==0x3d912245u,"independent scale boundary");
    volatile float a=value(0x41740000u), positive=value(0x40f40000u), negative=-positive;
    volatile float d=a*factor, reciprocal=1.f/d, p=positive*reciprocal,n=negative*reciprocal,direct=positive/d;
    need(bits(d)==0x3df5ebd8u&&bits(reciprocal)==0x41053ef3u&&bits(p)==0x427dffffu&&bits(n)==0xc27dffffu&&bits(direct)==0x427e0000u,"independent reciprocal and tie boundary");
    need(std::round(p)==63.f&&std::round(n)==-63.f&&std::round(direct)==64.f,"independent signed code boundary");
    const auto h=half_rne(d);volatile float expanded=ggml_fp16_to_fp32(h), x=value(0xc12a0000u);
    volatile float own=x*reciprocal, premature=x*(1.f/expanded);
    need(bits(expanded)==0x3df5e000u&&bits(own)==0xc2b0f79bu&&bits(premature)==0xc2b10021u,"independent half-scale boundary");
    need(std::round(own)==-88.f&&std::round(premature)==-89.f,"quantization must use full F32 scale");
}
int main(int argc,char**argv){try {
    native_q8_rcp_require_mode(0);for(uint32_t mode:{1u,2u,3u}){bool refused=false;try{native_q8_rcp_require_mode(mode);}catch(...){refused=true;}need(refused,"reciprocal observation: nonzero mode accepted");}
    boundaries();const auto base=native_stage_float_controls(strata_native_stages_spv,sizeof(strata_native_stages_spv)/4);
    const auto candidate=native_stage_float_controls(strata_native_quant_rcp_spv,sizeof(strata_native_quant_rcp_spv)/4);
    shader_contract(base,candidate);
    // The validator must reject a wrong scale constant and missing protection.
    auto wrong=candidate;bool modified=false;
    for(size_t p=5;p<wrong.size();p+=wrong[p]>>16)if((wrong[p]&65535)==spv::OpConstant&&(wrong[p]>>16)==4&&wrong[p+3]==0x3c010204u){wrong[p+3]^=1;modified=true;break;}
    need(modified,"reciprocal observation: constant mutation failed");bool rejected=false;try{shader_contract(base,wrong);}catch(...){rejected=true;}need(rejected,"reciprocal observation: wrong constant accepted");
    uint32_t one_id=0,rcp_id=0;
    for(size_t p=5;p<candidate.size();p+=candidate[p]>>16)if((candidate[p]&65535)==spv::OpConstant&&(candidate[p]>>16)==4&&candidate[p+3]==0x3f800000u)one_id=candidate[p+2];
    for(size_t p=5;p<candidate.size();p+=candidate[p]>>16)if((candidate[p]&65535)==spv::OpFDiv&&(candidate[p]>>16)==5&&candidate[p+3]==one_id)rcp_id=candidate[p+2];
    need(one_id&&rcp_id,"reciprocal observation: negative-control IDs missing");wrong=candidate;modified=false;
    for(size_t p=5;p<wrong.size();p+=wrong[p]>>16)if((wrong[p]&65535)==spv::OpDecorate&&(wrong[p]>>16)>=3&&wrong[p+1]==rcp_id&&wrong[p+2]==spv::DecorationNoContraction){wrong[p+2]=spv::DecorationRelaxedPrecision;modified=true;break;}
    need(modified,"reciprocal observation: protection mutation failed");rejected=false;try{shader_contract(base,wrong);}catch(...){rejected=true;}need(rejected,"reciprocal observation: unprotected reciprocal accepted");
    wrong=candidate;modified=false;
    for(size_t p=5;p<wrong.size();p+=wrong[p]>>16)if((wrong[p]&65535)==spv::OpFMul&&(wrong[p]>>16)==5&&(wrong[p+3]==rcp_id||wrong[p+4]==rcp_id)){wrong[p+3]=one_id;wrong[p+4]=one_id;modified=true;break;}
    need(modified,"reciprocal observation: dataflow mutation failed");rejected=false;try{shader_contract(base,wrong);}catch(...){rejected=true;}need(rejected,"reciprocal observation: disconnected reciprocal accepted");
    const auto observation=native_q8_observation_code(candidate);
    for(int mask=0;mask<4;++mask){const auto effective=native_q8_test_helper_effective(observation,mask&1,mask&2);
        const std::string path=argc==3&&std::string(argv[1])=="--modules"?std::string(argv[2])+"."+std::to_string(mask)+".spv":"";
        native_q8_validate_observation_effective(observation.data(),observation.size(),effective.data(),effective.size(),mask&1,mask&2,path.empty()?nullptr:path.c_str());}
    need(argc==1||(argc==3&&std::string(argv[1])=="--modules"),"optional --modules PREFIX");
    puts("reciprocal observation: exact SPIR-V dataflow, clamp, controls and host boundary vectors pass; no GPU equivalence claim");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
