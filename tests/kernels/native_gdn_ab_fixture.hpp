#pragma once
#include "native_hc_fixture.hpp"
#include "native_gdn_domain.hpp"
#include <cfenv>
namespace native_gdn_ab {
using namespace native_hc_fixture;
constexpr size_t N=2560,H=48;
constexpr std::array<size_t,8>counts{N,H*N/2,H*N/2,H,H,H,H,7*H};
constexpr std::array<const char*,9>fields{"alpha","beta-logit","v","softplus-exp","softplus","beta-exp","beta-denominator","gate","beta"};
struct Fixture{std::vector<float>x,dt,a;std::vector<uint16_t>wa,wb;};
inline void validate(const Fixture&f){native_hc_need(f.x.size()==N&&f.wa.size()==N*H&&f.wb.size()==N*H&&f.dt.size()==H&&f.a.size()==H&&std::fegetround()==FE_TONEAREST,"GDN ab T1/N2560/H48/RNE");for(auto*v:{&f.x,&f.dt,&f.a})for(float x:*v)native_hc_normal(x);for(auto*w:{&f.wa,&f.wb})for(uint16_t b:*w)native_hc_normal(value(uint32_t(b)<<16));}
// Native gdn_ab_body accumulates continuously across chunks, unlike the HC dot.
inline std::array<std::vector<float>,2>dot(const Fixture&f){validate(f);std::array<std::vector<float>,2>out{std::vector<float>(H),std::vector<float>(H)};
 for(size_t which=0;which<2;++which)for(size_t row=0;row<H;++row){auto&w=which?f.wb:f.wa;std::array<float,32>lanes{};
  for(size_t lane=0;lane<32;++lane)for(size_t j=lane;j<N/8;j+=32)for(size_t k=0;k<8;++k)lanes[lane]=native_hc_normal(std::fma(value(uint32_t(w[row*N+j*8+k])<<16),f.x[j*8+k],lanes[lane]));
  for(size_t offset:{16,8,4,2,1}){auto old=lanes;for(size_t lane=0;lane<32;++lane)lanes[lane]=native_hc_normal(old[lane]+old[lane^offset]);}out[which][row]=lanes[0];}
 return out;
}
inline Fixture synthetic(unsigned mode){native_hc_need(mode<4,"GDN ab fixture mode");Fixture f;f.x.resize(N);f.wa.resize(N*H);f.wb.resize(N*H);f.dt.resize(H);f.a.assign(H,-1.f);
 if(mode==1){f.x[0]=1;const float targets[]={-140,-40,-2,0,value(0x419fffff),20,value(0x41a00001),90};const float b[]={-100,-40,-1,0,1,40,100,4};for(size_t h=0;h<H;++h){f.wa[h*N]=native_hc_bf16_rne(1);f.wb[h*N]=native_hc_bf16_rne(b[h%8]);f.dt[h]=targets[h%8]-1.f;f.a[h]=-float(h%5+1)/4.f;}}
 if(mode==2){for(size_t j=0;j<N;++j)f.x[j]=float(int(j*17%37)-18)/128.f;for(size_t j=0;j<N*H;++j){f.wa[j]=native_hc_bf16_rne(float(int(j*23%41)-20)/256.f);f.wb[j]=native_hc_bf16_rne(float(int(j*31%43)-21)/256.f);}for(size_t h=0;h<H;++h){f.dt[h]=float(int(h%11)-5)/4.f;f.a[h]=-float(h%7+1)/8.f;}}
 if(mode==3){f.x[0]=16777216.f;f.x[8]=1;f.x[16]=-16777216.f;f.x[256]=1;f.x[512]=-16777216.f;f.x[4]=1;f.x[260]=value(0x3f800001);for(size_t c:{0,256,512})f.wa[c]=native_hc_bf16_rne(1);for(size_t c:{0,8,16})f.wa[N+c]=native_hc_bf16_rne(1);f.wa[2*N+4]=native_hc_bf16_rne(-1.0078125f);f.wa[2*N+260]=native_hc_bf16_rne(1.0078125f);}
 validate(f);return f;
}
inline void write(const std::string&path,const Fixture&f){validate(f);native_hc_need(!std::filesystem::exists(path),"GDN ab fresh fixture");std::ofstream o(path,std::ios::binary);for(uint32_t u:{0x31424147u,1u,uint32_t(N),uint32_t(H),1u,30u,0u,0u})word(o,u);for(float x:f.x)word(o,bits(x));for(auto*w:{&f.wa,&f.wb})for(size_t i=0;i<w->size();i+=2)word(o,uint32_t((*w)[i])|(uint32_t((*w)[i+1])<<16));for(auto*v:{&f.dt,&f.a})for(float x:*v)word(o,bits(x));native_hc_need(bool(o),"GDN ab fixture write");}
inline Fixture read(const std::string&path){std::ifstream in(path,std::ios::binary);for(uint32_t u:{0x31424147u,1u,uint32_t(N),uint32_t(H),1u,30u,0u,0u})native_hc_need(bool(in)&&word(in)==u,"GDN ab fixed geometry/type fixture");Fixture f;f.x.resize(N);f.wa.resize(N*H);f.wb.resize(N*H);f.dt.resize(H);f.a.resize(H);for(float&x:f.x)x=value(word(in));for(auto*w:{&f.wa,&f.wb})for(size_t i=0;i<w->size();i+=2){uint32_t b=word(in);(*w)[i]=uint16_t(b);(*w)[i+1]=uint16_t(b>>16);}for(auto*v:{&f.dt,&f.a})for(float&x:*v)x=value(word(in));native_hc_need(in.peek()==EOF,"GDN ab fixture trailing");validate(f);return f;}
inline void contract(const std::string&path){native_hc_need(!std::filesystem::exists(path),"GDN ab fresh output contract");std::ofstream o(path);o<<"{\"fields\":{";for(size_t i=0;i<fields.size();++i)o<<(i?",":"")<<"\""<<fields[i]<<".f32le\":192";o<<"},\"count\":48,\"qualified\":false}\n";native_hc_need(bool(o),"GDN ab contract write");}
inline void exact(const std::vector<float>&a,const std::vector<float>&b,const char*message){native_hc_need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),message);}
}
