#pragma once
#include "native_hc_fixture.hpp"
#include <cfenv>
namespace native_gdn_conv {
using namespace native_hc_fixture;
constexpr std::array<size_t,10> counts{30720,10240,40960,30720,10240,10240,10240,10240,32,32};
using Fixture=std::array<std::vector<float>,3>;
inline void validate(const Fixture&f){native_hc_need(std::fegetround()==FE_TONEAREST,"GDN conv requires RNE host");for(size_t i=0;i<3;++i){native_hc_need(f[i].size()==counts[i],"GDN conv fixed C10240/S128/qk32/T1");for(float x:f[i])native_hc_normal(x);}}
inline Fixture synthetic(unsigned mode){native_hc_need(mode<3,"GDN conv fixture mode");Fixture f;for(size_t i=0;i<3;++i)f[i].resize(counts[i]);
 if(mode==1){for(size_t c:{size_t(0),size_t(4095),size_t(4096),size_t(10239)}){f[0][c*3]=1;f[0][c*3+1]=2;f[0][c*3+2]=3;f[1][c]=4;for(size_t k=0;k<4;++k)f[2][c*4+k]=1;}}
 if(mode==2){for(size_t i=0;i<f[0].size();++i)f[0][i]=float(int(i*17%31)-15)/32.f;for(size_t i=0;i<f[1].size();++i)f[1][i]=float(int(i*23%37)-18)/32.f;for(size_t i=0;i<f[2].size();++i)f[2][i]=float(int(i*11%29)-14)/16.f;
 // Held-out signed tail/boundary channels. Native evaluates the same expression.
 for(size_t c=0;c<8;++c){f[0][c*3]=f[0][c*3+1]=f[0][c*3+2]=0;f[1][c]=std::array<float,8>{0,88,104,-89,-128,20,-4,4}[c];for(size_t k=0;k<4;++k)f[2][c*4+k]=k==3?1.f:0.f;}
 // Contraction-sensitive v channels bypass L2; their positive SiLU denominator
 // rounds to one, exposing the actual production convolution bits directly.
 for(size_t c:{size_t(4096),size_t(4097)}){float x=value(0x40330828)*2.f,y=value(0x402e4d67)*2.f;if(c==4097)std::swap(x,y);f[0][c*3]=x;f[0][c*3+1]=y;f[0][c*3+2]=f[1][c]=0;f[2][c*4]=x;f[2][c*4+1]=y;f[2][c*4+2]=f[2][c*4+3]=0;}}

 validate(f);return f;}
inline void write(const std::string&path,const Fixture&f){validate(f);native_hc_need(!std::filesystem::exists(path),"GDN conv fresh fixture");std::ofstream o(path,std::ios::binary);word(o,0x31564347);word(o,1);for(auto&a:f)for(float x:a)word(o,bits(x));native_hc_need(bool(o),"GDN conv fixture write");}
// Shared producer output contract: convolution emits h/history, never recurrence y.
inline void contract(const std::string&path){native_hc_need(!std::filesystem::exists(path),"GDN conv fresh output contract");std::ofstream o(path);o<<"{\"fields\":{";const char*names[]={"next-history.f32le","h.f32le","pre.f32le","exp.f32le","silu.f32le","rs.f32le","norm-arg.f32le"};for(size_t i=0;i<7;++i)o<<(i?",":"")<<"\""<<names[i]<<"\":"<<counts[i+3]*4;o<<"},\"qualified\":false}\n";native_hc_need(bool(o),"GDN conv output contract write");}
inline Fixture read(const std::string&path){std::ifstream in(path,std::ios::binary);native_hc_need(bool(in)&&word(in)==0x31564347&&word(in)==1,"GDN conv fixture header");Fixture f;for(size_t i=0;i<3;++i){f[i].resize(counts[i]);for(float&x:f[i])x=value(word(in));}native_hc_need(in.peek()==EOF,"GDN conv trailing bytes");validate(f);return f;}
struct Control{std::vector<float> history,h,pre,silu,arg;};
inline Control control(const Fixture&f,const std::vector<float>&ex,const std::vector<float>&rs,bool commit,const std::vector<float>*actual_silu=nullptr){validate(f);native_hc_need(ex.size()==10240&&rs.size()==32,"GDN conv primitive shape");native_hc_need(!actual_silu||actual_silu->size()==10240,"GDN conv actual SiLU primitive shape");Control out{f[0],std::vector<float>(10240),std::vector<float>(10240),std::vector<float>(10240),std::vector<float>(32)};auto ck=[](float x){return native_hc_normal(x);};
 for(size_t c=0;c<10240;++c){if(commit){out.history[c*3]=f[0][c*3+1];out.history[c*3+1]=f[0][c*3+2];out.history[c*3+2]=f[1][c];}
  float second=ck(f[0][c*3+1]*f[2][c*4+1]);float v=ck(std::fma(f[0][c*3],f[2][c*4],second));v=ck(std::fma(f[0][c*3+2],f[2][c*4+2],v));v=ck(std::fma(f[1][c],f[2][c*4+3],v));out.pre[c]=v;
  const float den=1.f+ex[c];native_hc_need((std::isfinite(den)&&den>=1.f)||den==INFINITY,"GDN conv actual denominator");out.h[c]=out.silu[c]=ck(actual_silu?(*actual_silu)[c]:v/den);
 }
 for(size_t group=0;group<32;++group){std::array<float,128>a{};for(size_t col=0;col<128;++col){float y=out.silu[group*128+col];a[col]=ck(y*y);}for(size_t shift:{16,8,4,2,1}){auto prev=a;for(size_t col=0;col<128;++col)a[col]=ck(prev[col]+prev[col^shift]);}float sum=ck(ck(ck(a[0]+a[32])+a[64])+a[96]);out.arg[group]=ck(sum+1e-6f);for(size_t col=0;col<128;++col)out.h[group*128+col]=ck(out.silu[group*128+col]*rs[group]);}
 return out;
}
inline void exact(const std::vector<float>&a,const std::vector<float>&b,const char*message){native_hc_need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),message);}
}
