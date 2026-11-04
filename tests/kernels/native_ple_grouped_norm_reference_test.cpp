// Host source-order controls for native_gr_rms_norm_weighted's PLE geometry.
// Division and rsqrt here are host primitives, not a CUDA bit-identity oracle.
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace {
constexpr size_t columns=2560,lanes=1024;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
float add(float a,float b){volatile float r=a+b;return r;}
float mul(float a,float b){volatile float r=a*b;return r;}
uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
float from_bits(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
float row_sum(const float* input){
 std::array<float,lanes> p{};
 for(size_t lane=0;lane<lanes;++lane)
  for(size_t col=lane;col<columns;col+=lanes)p[lane]=std::fma(input[col],input[col],p[lane]);
 for(size_t offset=16;offset;offset>>=1){auto previous=p;for(size_t i=0;i<lanes;++i)p[i]=add(previous[i],previous[i^offset]);}
 std::array<float,32> warp{};for(size_t i=0;i<32;++i)warp[i]=p[i*32];
 for(size_t offset=16;offset;offset>>=1){auto previous=warp;for(size_t i=0;i<32;++i)warp[i]=add(previous[i],previous[i^offset]);}
 return warp[0];
}
}
int main(){try{
 need(std::fesetround(FE_TONEAREST)==0,"host rounding mode");
 std::vector<float> x(4*columns,0.0f);
 for(size_t c=0;c<columns;++c){x[c]=1.0f;x[columns+c]=2.0f;x[2*columns+c]=-0.5f;}
 const std::array<float,4> expected{{2560.f,10240.f,640.f,0.f}};
 for(size_t row=0;row<4;++row)need(bits(row_sum(x.data()+row*columns))==bits(expected[row]),"literal row sum differs");
 // Three iterations per thread for lanes 0..511, two for the rest. These
 // isolated powers catch a 512-thread traversal or a missing final fragment.
 std::fill(x.begin(),x.end(),0.0f);x[0]=1.f;x[1024]=2.f;x[2048]=4.f;x[511]=8.f;x[1535]=16.f;x[2559]=32.f;
 need(bits(row_sum(x.data()))==bits(1365.f),"fragment mapping differs");
 // The large term loses two lane-local ones before the XOR tree; a serial
 // sum loses many more. This literal detects a change in reduction layout.
 std::fill(x.begin(),x.end(),1.0f);x[0]=4096.f;
 need(bits(row_sum(x.data()))==0x4b8004ffu,"native XOR reduction literal differs");
 float serial=0;for(size_t i=0;i<columns;++i)serial=std::fma(x[i],x[i],serial);
 need(bits(serial)!=0x4b8004ffu,"serial reduction control did not diverge");
 // These three same-lane values distinguish fused from separately rounded
 // products. CUDA instruction inspection is still required for qualification.
 std::fill(x.begin(),x.end(),0.0f);x[0]=from_bits(0x3fd1ab15u);x[1024]=from_bits(0x3f96bf95u);x[2048]=from_bits(0x40c687f2u);
 need(bits(row_sum(x.data()))==0x422a3e70u,"fused lane accumulation literal differs");
 const float scale=from_bits(0x3f01d651u),value=from_bits(0x3eca235du),weight=from_bits(0x3e0868c4u);
 need(bits(mul(mul(scale,value),weight))==0x3cda828bu&&bits(mul(mul(value,weight),scale))==0x3cda828au,"weighted output association differs");
 std::puts("PLE grouped RMS source-order sum controls passed; CUDA math remains unqualified");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"PLE grouped norm reference: %s\n",e.what());return 1;}}
