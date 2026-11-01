#pragma once
#include "native_hc_fixture.hpp"
#include "native_gdn_log1p_domain.hpp"
#include <cfenv>
namespace native_gdn_log1p {
using namespace native_hc_fixture;
constexpr size_t count=256;
constexpr std::array<const char*,6>fields{"result","compensated","u","log","numerator","denominator"};
inline void validate(const std::vector<float>&v){native_hc_need(v.size()==count&&std::fegetround()==FE_TONEAREST,"GDN log1p fixed count/RNE host");for(float x:v)native_gdn_log1p_input(x);}
inline std::vector<float>synthetic(unsigned mode){native_hc_need(mode<3,"GDN log1p fixture mode");std::vector<float>v(count);
 for(size_t i=0;i<count;++i){uint32_t b=0;
  if(mode==0){if(i==1)b=0x80000000u;else if(i>=2){int e=-126+int((i-2)%103);b=uint32_t(e+127)<<23;int d=int(i%3)-1;if(int64_t(b)+d>=0x800000)b=uint32_t(int64_t(b)+d);}}
  if(mode==1){const uint32_t pivot[]={0x33800000u,0x34000000u,0x3f800000u,0x40000000u,0x4de75844u,0x4e000000u};int d=int(i%17)-8;b=pivot[(i/17)%6];if(b==0x4e000000u&&d>0)d=-d;b=uint32_t(int64_t(b)+d);}
  if(mode==2){int e=-23+int(i%52);b=(uint32_t(e+127)<<23)|uint32_t((i*1664525u+1013904223u)&0x7fffffu);}
  v[i]=value(b);
 }validate(v);return v;}
inline void write(const std::string&path,const std::vector<float>&v){validate(v);native_hc_need(!std::filesystem::exists(path),"GDN log1p fresh fixture");std::ofstream o(path,std::ios::binary);word(o,0x31504c47);word(o,1);word(o,count);for(float x:v)word(o,bits(x));native_hc_need(bool(o),"GDN log1p fixture write");}
inline std::vector<float>read(const std::string&path){std::ifstream in(path,std::ios::binary);native_hc_need(bool(in)&&word(in)==0x31504c47&&word(in)==1&&word(in)==count,"GDN log1p fixture header");std::vector<float>v(count);for(float&x:v)x=value(word(in));native_hc_need(in.peek()==EOF,"GDN log1p fixture trailing bytes");validate(v);return v;}
inline void contract(const std::string&path){native_hc_need(!std::filesystem::exists(path),"GDN log1p fresh output contract");std::ofstream o(path);o<<"{\"fields\":{";for(size_t i=0;i<fields.size();++i)o<<(i?",":"")<<"\""<<fields[i]<<".f32le\":1024";o<<"},\"count\":256,\"qualified\":false}\n";native_hc_need(bool(o),"GDN log1p contract write");}
inline void exact(const std::vector<float>&a,const std::vector<float>&b,const char*message){native_hc_need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),message);}
}
