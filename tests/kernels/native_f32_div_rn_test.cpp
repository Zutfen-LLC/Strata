#include "native_q8_geometry_fixture.hpp"
#include "native_q8_observation.hpp"
#include <vulkan/vulkan.hpp>
#include "strata_native_quant_rn_spv.hpp"
#include "strata_native_observe_test_helper.hpp"
#include <cfenv>
#include <set>
namespace rn {using uint=uint32_t;
#include "../../src/kernels/vulkan/native_f32_div_rn.glsl"
}
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
bool nan_bits(uint32_t x){return (x&0x7fffffff)>0x7f800000;}
float divide(float a,float b){return value(rn::strata_f32_div_rn_bits(bits(a),bits(b)));}
void check(uint32_t a,uint32_t b){volatile float x=value(a),y=value(b);const float reference=x/y;
    const auto got=rn::strata_f32_div_rn_bits(a,b);
    const bool match=nan_bits(bits(reference))?nan_bits(got):bits(reference)==got;
    if(!match)fprintf(stderr,"a=%08x b=%08x result=%08x reference=%08x\n",a,b,got,bits(reference));
    need(match,"RN division differs from host IEEE reference");}
Bytes quantize_rn(const std::vector<float>& x){Bytes out(x.size()/32*36);
    for(size_t block=0;block<x.size()/32;++block){std::array<float,32> m,s;
        for(size_t lane=0;lane<32;++lane){m[lane]=std::fabs(x[block*32+lane]);s[lane]=x[block*32+lane];}
        for(size_t offset=16;offset;offset>>=1){const auto pm=m,ps=s;for(size_t lane=0;lane<32;++lane){m[lane]=std::fmax(pm[lane],pm[lane^offset]);s[lane]=ps[lane]+ps[lane^offset];}}
        const float d=finite_half(divide(m[0],127.f));const auto hd=half_rne(d),hs=half_rne(finite_half(s[0]));
        const size_t at=block*36;out[at]=uint8_t(hd);out[at+1]=uint8_t(hd>>8);out[at+2]=uint8_t(hs);out[at+3]=uint8_t(hs>>8);
        for(size_t lane=0;lane<32;++lane){float q=std::round(divide(x[block*32+lane],d));q=q>127.f?127.f:q< -127.f?-127.f:q;
            out[at+4+lane]=m[0]==0.f || std::isnan(q)?0:uint8_t(int(q));}}
    return out;
}
int main(int argc,char** argv){try {
    need(std::fesetround(FE_TONEAREST)==0,"host nearest-even mode unavailable");
    need(bits(value(1)/value(0x3f800000))==1,"host flushes subnormal reference inputs");
    const std::vector<uint32_t> cases{0,0x80000000,1,2,3,0x007fffff,0x00800000,0x00800001,
        0x3f000000,0x3f7fffff,0x3f800000,0x3f800001,0x40000000,0x7f7fffff,0x7f800000,
        0xff800000,0x7fc12345,0x7f812345,0xffc54321,0xbf800000,0x80000001,0x807fffff,0xff7fffff};
    for(auto a:cases)for(auto b:cases)check(a,b);
    need(rn::strata_f32_div_rn_bits(1,0x40000000)==0,"minimum subnormal half tie must round to even zero");
    need(rn::strata_f32_div_rn_bits(3,0x40000000)==2,"subnormal tie must round to even significand");
    need(rn::strata_f32_div_rn_bits(0x80000001,0x40000000)==0x80000000,"underflow zero sign lost");
    need(rn::strata_f32_div_rn_bits(0x7f812345,0x3f800000)==0x7fc12345,"explicit NaN quiet/payload convention changed");
    need(rn::strata_f32_div_rn_bits(0,0)==0x7fc00000,"invalid division NaN convention changed");
    uint32_t state=0x12345678;auto random=[&](){state^=state<<13;state^=state>>17;state^=state<<5;return state;};
    for(size_t n=0;n<250000;++n){const auto a=random(),b=random();check(a,b);}
    const uint32_t xi[]={0x3f1c0000,0xbf180000,0xbf1c0000,0x3f100000};
    const uint32_t amax[]={0x3f9c0000,0x3f980000,0x3f9c0000,0x3f900000};
    const uint32_t ratios[]={0x427e0000,0xc27e0000,0xc27e0000,0x427dffff};
    for(size_t k=0;k<4;++k)need(rn::strata_f32_div_rn_bits(xi[k],rn::strata_f32_div_rn_bits(amax[k],bits(127.f)))==ratios[k],"retained boundary regression");
    need(quantize_rn(synthetic().x)==quantize(synthetic().x),"existing 832-element native host fixture differs");
    const auto code=native_stage_float_controls(strata_native_quant_rn_spv,sizeof(strata_native_quant_rn_spv)/4);
    size_t divides=0,loops=0;std::set<uint32_t> precise,div_ids;
    for(size_t p=5;p<code.size();p+=code[p]>>16){const auto op=code[p]&65535;
        if(op==spv::OpFDiv){++divides;div_ids.insert(code[p+2]);}
        if(op==spv::OpLoopMerge)++loops;
        if(op==spv::OpDecorate && code[p+2]==spv::DecorationNoContraction)precise.insert(code[p+1]);
        if(op==spv::OpTypeFloat || op==spv::OpTypeInt)need(code[p+2]!=64,"RN shader introduced a 64-bit type");}
    need(divides==1 && loops>=3,"only the original activation division should remain floating point");
    for(auto id:div_ids)need(precise.count(id),"activation division lost its controls");
    const auto observation=native_q8_observation_code(code);
    for(int mask=0;mask<4;++mask){const auto effective=native_q8_test_helper_effective(observation,mask&1,mask&2);
        const std::string path=argc==3 && std::string(argv[1])=="--modules"?std::string(argv[2])+"."+std::to_string(mask)+".spv":"";
        native_q8_validate_observation_effective(observation.data(),observation.size(),effective.data(),effective.size(),mask&1,mask&2,path.empty()?nullptr:path.c_str());}
    if(argc==5 && std::string(argv[1])=="--fixture" && std::string(argv[3])=="--native"){
        const auto input=geometry::read_fixture(argv[2]);std::ifstream in(argv[4],std::ios::binary);need(bool(in),"native capture read failed");
        need(word(in)==0x314d384e && word(in)==1,"native capture role/status");
        const size_t sizes[]={2160,1920,1920,1920,540,7680};Bytes expected;
        for(size_t k=0;k<6;++k){need(word(in)==sizes[k],"native capture section size");Bytes section(sizes[k]*4);
            in.read(reinterpret_cast<char*>(section.data()),section.size());need(bool(in),"native capture truncated");if(k==0)expected=section;}
        need(in.peek()==EOF,"native capture trailing bytes");const auto got=quantize_rn(input);
        size_t differences=0;for(size_t k=0;k<got.size();++k)differences+=got[k]!=expected[k];
        printf("retained_native_input_bytes=%zu different_bytes=%zu\n",got.size(),differences);need(differences==0,"native fixture differs");
    }else need(argc==1 || (argc==3 && std::string(argv[1])=="--modules"),"optional --fixture FILE --native FILE or --modules PREFIX");
    puts("shared RN division: 250000 random pairs, IEEE edge classes/ties and shader constraints passed; host execution, no GPU proof");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
