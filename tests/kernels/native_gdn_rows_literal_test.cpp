// Literal causal-state control for the row recurrence; does not execute Vulkan.
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
namespace {
using State=std::array<float,128>;
void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
// One head/column, K and Q are one-hot with coefficient 0.5; decay and
// beta are 0.5. Explicit FMA association matches the S128/RG4 row algorithm.
float step(State& s,float value){
 float partial[4]{};
 for(size_t g=0;g<4;++g)for(size_t r=0;r<32;++r){size_t i=32*g+r;partial[g]=std::fma(s[i],i==0?0.5f:0.0f,partial[g]);}
 float kv=((partial[0]+partial[1])+partial[2])+partial[3];
 float delta=std::fma(-0.5f,kv,value)*0.5f;float out[4]{};
 for(size_t g=0;g<4;++g)for(size_t r=0;r<32;++r){size_t i=32*g+r;float update=(i==0?0.5f:0.0f)*delta;s[i]=std::fma(0.5f,s[i],update);out[g]=std::fma(s[i],i==0?0.5f:0.0f,out[g]);}
 return ((out[0]+out[1])+out[2])+out[3];
}
void exercise(float seed){
 State initial;initial.fill(seed);State serial=initial;std::array<float,5> values{1,2,3,4,5},expected{};
 for(size_t i=0;i<5;++i)expected[i]=step(serial,values[i]);
 State segmented=initial;std::array<float,5> actual{};
 // Two T2 segments and one T1 tail; each segment retains state between rows.
 for(size_t first:{size_t(0),size_t(2),size_t(4)}){size_t count=first==4?1:2;for(size_t r=0;r<count;++r)actual[first+r]=step(segmented,values[first+r]);}
 need(actual==expected&&segmented==serial,"T2/T1 causal state");
 if(seed==0.25f){
  State s=initial;need(step(s,1)==0.1796875f&&s[0]==0.359375f&&s[1]==0.125f,"literal first row");
  need(step(s,2)==0.32861328125f&&s[0]==0.6572265625f&&s[1]==0.0625f,"literal second row must consume row0 state");
  need(step(s,3)==0.518768310546875f&&s[0]==1.03753662109375f&&s[1]==0.03125f,"literal T1 tail");
 }
}
}
int main(){try{exercise(0.25f);exercise(-0.5f);exercise(0.25f);std::puts("literal causal recurrence state controls passed");return 0;}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
