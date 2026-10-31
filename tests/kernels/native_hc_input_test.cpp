#include "native_hc_fixture.hpp"
#include <algorithm>
#include <limits>
using namespace native_hc_fixture;
template<class F>void refuses(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}native_hc_need(bad,"native HC: refusal absent");}
int main(){try{
#ifndef _WIN32
 const char* prior=std::getenv("STRATA_HC_REQ8");const std::string saved=prior?prior:"";setenv("STRATA_HC_REQ8","1",1);refuses([]{native_hc_environment();});if(prior)setenv("STRATA_HC_REQ8",saved.c_str(),1);else unsetenv("STRATA_HC_REQ8");
#endif
 for(uint32_t t=1;t<=8;++t)native_hc_shape(2560,4,t);
 for(uint32_t t:{0u,9u,UINT32_MAX})refuses([&]{native_hc_shape(2560,4,t);});
 refuses([]{native_hc_shape(2559,4,1);});refuses([]{native_hc_shape(2560,3,1);});refuses([]{native_hc_shape(2560,4,1,1e-5f);});refuses([]{native_hc_shape(2560,4,1,1e-6f,true);});refuses([]{native_hc_shape(2560,4,1,1e-6f,false,8);});
 // Independent nearest-neighbor BF16 oracle, exact double Q8 value; no bit-bias rounding.
 std::vector<double> table;for(uint32_t h=0;h<0x7f80;++h)table.push_back(double(native_hc_value(h<<16)));
 for(uint32_t scale:{0u,1u,0x3ffu,0x400u,0x1401u,0x3555u,0x3c00u,0x3c01u,0x7bffu,0x8000u,0x8400u,0xbc01u})for(int code=-128;code<128;++code){uint8_t data[34]{};uint16_t half=uint16_t(scale);std::memcpy(data,&half,2);std::fill(data+2,data+34,uint8_t(code));const auto out=native_hc_q8_bf16(data,34,32);const bool sign=((scale>>15)!=0)^(code<0);double exact=std::abs(double(ggml_fp16_to_fp32(half))*double(code));auto it=std::lower_bound(table.begin(),table.end(),exact);size_t hi=it-table.begin(),chosen=hi;if(hi&&hi<table.size()){double lo_dist=exact-table[hi-1],hi_dist=table[hi]-exact;if(lo_dist<hi_dist||(lo_dist==hi_dist&&((hi-1)%2)==0))chosen=hi-1;}if(hi==table.size())chosen=hi-1;uint16_t expected=uint16_t(chosen)|(sign?0x8000u:0u);if(exact==0)expected=uint16_t((scale&0x8000u)^(code<0?0x8000u:0));native_hc_need(out[0]==expected,"native HC: original pack BF16 RNE mismatch");}
 native_hc_need(native_hc_bf16_rne(native_hc_value(0x3f808000))==0x3f80&&native_hc_bf16_rne(native_hc_value(0x3f818000))==0x3f82,"native HC: even/odd BF16 ties");
 refuses([]{native_hc_normal(std::numeric_limits<float>::denorm_min());});refuses([]{native_hc_normal(INFINITY);});refuses([]{native_hc_bf16_rne(NAN);});
 uint8_t bad[34]{};refuses([&]{native_hc_q8_bf16(bad,33,32);});bad[1]=0x7c;refuses([&]{native_hc_q8_bf16(bad,34,32);});
 // Independent closed-form constant streams and basis-row injection expose indexing/T stride.
 for(uint32_t t:{1u,8u}){Fixture f;f.t=t;f.x.resize(t*10240);f.w.assign(10240,1);f.wi.assign(40960,0);for(size_t token=0;token<t;++token)for(int c=0;c<4;++c)std::fill(f.x.begin()+token*10240+c*2560,f.x.begin()+token*10240+(c+1)*2560,float(c+token));for(int c=0;c<4;++c)f.wi[c*10240+c*2560]=0x3f80;const auto got=native_hc_oracle(f.x,f.w,f.wi,t);for(uint32_t token=0;token<t;++token)for(int c=0;c<4;++c){float x=float(c+token),rs=1.f/std::sqrt(x*x+1e-6f);native_hc_need(got.rs[token*4+c]==rs&&got.inject[token*4+c]==x*rs,"native HC: closed-form stream/basis source oracle");}}
 auto f=synthetic();refuses([&]{auto x=f.x;x[0]=std::numeric_limits<float>::max();native_hc_oracle(x,f.w,f.wi,f.t);});refuses([&]{auto x=f.x;x[0]=1e-20f;native_hc_oracle(x,f.w,f.wi,f.t);});
 puts("native HC shape, BF16 independent nearest-neighbor oracle, ties, normal-domain refusal and stream/T basis controls passed");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
