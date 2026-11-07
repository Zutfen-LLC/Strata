// Independent host controls for the native prefill mapping, not device evidence.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
static void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
static uint32_t bits(float v){uint32_t b;std::memcpy(&b,&v,4);return b;}
static float value(uint32_t b){float v;std::memcpy(&v,&b,4);return v;}
int main(){try{
 // Literal values make the two multiply orderings round differently.
 bool differs=false;for(uint32_t i=0;i<10000&&!differs;++i){float r=value(0x3f000001u+i*13u),rs=value(0x3e800001u+i*7u),w=value(0x3f800001u+i*17u);volatile float a=r*rs,b=r*w;volatile float pre=a*w,decode=b*rs;differs=bits(pre)!=bits(decode);}need(differs,"normalization multiplication order control");
 volatile float a=1.00000011920928955078125f,b=a,c=-1.0000002384185791015625f;volatile float product=a*b;volatile float split=product+c;float fused=std::fma(float(a),float(b),float(c));need(split==0&&fused==0x1p-46f,"fused mixing control");
 // Each lane visits d=lane+256*k. Warp XOR reductions then reduce eight
 // totals using the same 32-lane XOR network, with 24 zero lanes.
 std::array<float,4> sums{};for(int ch=0;ch<4;++ch){float lanes[256]{};for(int lane=0;lane<256;++lane)for(int k=0;k<10;++k){int d=lane+256*k;float r=value(0x3e800000u+uint32_t((d*9701+ch*29011)&0x007fffff));lanes[lane]=std::fma(r,r,lanes[lane]);}
  for(int step=16;step;step>>=1){float old[256];std::memcpy(old,lanes,sizeof old);for(int lane=0;lane<256;++lane)lanes[lane]=old[lane]+old[lane^step];}
  float first[32]{};for(int i=0;i<8;++i)first[i]=lanes[32*i];for(int step=16;step;step>>=1){float old[32];std::memcpy(old,first,sizeof old);for(int lane=0;lane<32;++lane)first[lane]=old[lane]+old[lane^step];}sums[ch]=first[0];}
 need(bits(sums[0])==0x43b8e9e4u&&bits(sums[1])==0x43b92d85u&&bits(sums[2])==0x43b9714eu&&bits(sums[3])==0x43b9b540u,"channel mapping control");
 printf("prefill arithmetic host controls passed; sums %08x %08x %08x %08x; not GPU evidence\n",bits(sums[0]),bits(sums[1]),bits(sums[2]),bits(sums[3]));return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
