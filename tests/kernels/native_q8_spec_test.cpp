#include "native_q8_fixture.hpp"
using namespace native_q8_fixture;
int main(){try {
    need(half_rne(0.f)==0 && half_rne(-0.f)==0x8000,"signed half zero");
    need(half_rne(65504.f)==0x7bff && half_rne(65520.f)==0x7c00,"half overflow tie");
    need(half_rne(value(0x33800000))==1 && half_rne(value(0x33000000))==0,"half subnormal tie");
    need(half_rne(1.00048828125f)==0x3c00 && half_rne(1.00146484375f)==0x3c02,"half even ties");
    need(bits(finite_half(-value(0x7f800000)))==bits(-65504.f),"infinite clamp");
    need(std::isnan(finite_half(value(0x7fc12345))),"NaN preserved by clamp");
    auto f=synthetic();auto q=quantize(f.x);
    need(q[3*36+5]==1 && q[3*36+6]==255 && q[3*36+7]==2 && q[3*36+8]==254,"roundf away ties");
    need(q[4*36+2]==0 && q[4*36+3]==0x4c,"XOR raw sum differs from linear/code-derived sum");
    need(q[6*36+2]==255 && q[6*36+3]==123,"raw sum finite clamp");
    need(q[7*36]==255 && q[7*36+1]==123 && q[7*36+4]==127,"scale finite clamp");
    for(size_t i=4;i<36;++i)need(q[i]==0 && q[36+i]==0,"zero block codes");
    // Exhaust all finite half values: exact f32 expansion must round-trip.
    for(uint32_t h=0;h<65536;++h){uint32_t e=(h>>10)&31,m=h&1023; if(e==31)continue;
        float x=e?std::ldexp(1.f+float(m)/1024.f,int(e)-15):std::ldexp(float(m),-24);
        if(h&0x8000)x=-x;need(half_rne(x)==h,"finite half roundtrip");}
    std::puts("native Q8_1 host specification: half/ties/clamp/zero passed (not GPU proof)");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
