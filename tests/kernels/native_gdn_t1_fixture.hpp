#pragma once
#include "native_hc_fixture.hpp"
#include <array>
namespace native_gdn_t1 {
using namespace native_hc_fixture;
constexpr std::array<size_t,13> counts{786432,10240,48,48,6144,128,786432,6144,48,6144,49,6144,6144};
using Fixture=std::array<std::vector<float>,6>;
inline void validate(const Fixture&f){for(size_t i=0;i<6;++i){native_hc_need(f[i].size()==counts[i],"GDN fixture shape");for(float x:f[i]){native_hc_normal(x);if(i==2)native_hc_need(x<=0&&x>=-87.33651733398438f,"GDN decay domain");if(i==4)native_hc_need(std::fabs(x)<=87.33651733398438f,"GDN output gate domain");}}}
inline Fixture synthetic(unsigned mode){native_hc_need(mode<3,"GDN fixture case");Fixture f;for(size_t i=0;i<6;++i)f[i].resize(counts[i]);std::fill(f[3].begin(),f[3].end(),0.5f);std::fill(f[5].begin(),f[5].end(),1.f);
 if(mode==1){f[1][0]=0.5f;f[1][2048]=0.25f;f[1][4096]=2.f;}
 if(mode==2){for(size_t j=0;j<f[0].size();++j)f[0][j]=float(int(j*17%31)-15)/2048.f;for(size_t j=0;j<f[1].size();++j)f[1][j]=float(int(j*23%37)-18)/128.f;for(size_t j=0;j<48;++j){f[2][j]=-float(j%7+1)/16.f;f[3][j]=float(j%5+1)/8.f;}for(size_t j=0;j<6144;++j)f[4][j]=float(int(j*11%33)-16)/4.f;for(size_t j=0;j<128;++j)f[5][j]=0.5f+float(j%17)/16.f;}validate(f);return f;}
inline void write(const std::string&path,const Fixture&f){validate(f);native_hc_need(!std::filesystem::exists(path),"GDN fresh fixture");std::ofstream o(path,std::ios::binary);word(o,0x314e4447);word(o,1);for(auto&a:f)for(float x:a)word(o,bits(x));native_hc_need(bool(o),"GDN fixture write");}
inline Fixture read(const std::string&path){std::ifstream in(path,std::ios::binary);native_hc_need(bool(in)&&word(in)==0x314e4447&&word(in)==1,"GDN fixture header");Fixture f;for(size_t i=0;i<6;++i){f[i].resize(counts[i]);for(float&x:f[i])x=value(word(in));}native_hc_need(in.peek()==EOF,"GDN fixture extra bytes");validate(f);return f;}
// Independent source-order control; primitive values are the backend's actual exp,
// rsqrt and sigmoid. Local basic operations/state/y must reproduce exactly.
struct Control{std::vector<float> state,oc,y,arg;};
inline Control control(const Fixture&f,const std::vector<float>&eg,const std::vector<float>&rs,const std::vector<float>&sig){
 validate(f);native_hc_need(eg.size()==48&&rs.size()==49&&sig.size()==6144,"GDN control primitive shape");Control out{std::vector<float>(786432),std::vector<float>(6144),std::vector<float>(6144),std::vector<float>(48)};
 auto ck=[](float x){return native_hc_normal(x);};
 for(size_t head=0;head<48;++head){float factor=ck(eg[head]);size_t qh=head%16;
  for(size_t col=0;col<128;++col){float kv[4]={};for(size_t group=0;group<4;++group)for(size_t r=group*32;r<(group+1)*32;++r)kv[group]=ck(std::fma(f[0][(r*48+head)*128+col],f[1][2048+qh*128+r],kv[group]));
   float sum=ck(ck(ck(kv[0]+kv[1])+kv[2])+kv[3]);float delta=ck(ck(std::fma(-factor,sum,f[1][4096+head*128+col]))*f[3][head]);float o[4]={};
   for(size_t group=0;group<4;++group)for(size_t r=group*32;r<(group+1)*32;++r){size_t i=(r*48+head)*128+col;float update=ck(f[1][2048+qh*128+r]*delta);out.state[i]=ck(std::fma(factor,f[0][i],update));o[group]=ck(std::fma(out.state[i],f[1][qh*128+r],o[group]));}
   out.oc[head*128+col]=ck(ck(ck(ck(o[0]+o[1])+o[2])+o[3])*rs[48]);
  }
  std::array<float,128>a{};for(size_t c=0;c<128;++c)a[c]=ck(out.oc[head*128+c]*out.oc[head*128+c]);for(size_t shift:{16,8,4,2,1}){auto prev=a;for(size_t c=0;c<128;++c)a[c]=ck(prev[c]+prev[c^shift]);}
  float sum=ck(ck(ck(a[0]+a[32])+a[64])+a[96]);out.arg[head]=ck(ck(sum/128.f)+1e-6f);
  for(size_t c=0;c<128;++c){size_t i=head*128+c;out.y[i]=ck(ck(ck(out.oc[i]*rs[head])*f[5][c])*sig[i]);}
 }return out;
}
inline void exact(const std::vector<float>&a,const std::vector<float>&b,const char*message){native_hc_need(a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*4)==0,message);}
}
