#pragma once
#include "native_hc_fixture.hpp"
#include <string>
namespace head_hc_fixture {
using namespace native_hc_fixture;
constexpr size_t D=10240, P=3276800;
struct Fixture {std::vector<float> x,w;std::vector<uint16_t> down,up;};
inline Fixture make(const std::string& kind){
 native_hc_need(kind=="zero"||kind=="sparse"||kind=="dense"||kind=="cancel"||kind=="tails","head fixture: unknown case");
 Fixture f; f.x.resize(D);f.w.resize(D);f.down.resize(P);f.up.resize(P);
 for(size_t i=0;i<D;++i){
  f.w[i]=0.75f+float((i*7)%17)*0.015625f;
  f.x[i]=kind=="zero"?0.f:kind=="sparse"?(i%257==0?float(int(i%9)-4)*0.125f:0.f):float(int((i*73+i/2560*19)%251)-125)*0.0078125f;
  if(kind=="dense" && i%97==0) f.x[i]*=1024.f;
  if(kind=="dense" && f.x[i]!=0.f && i%3==0) f.x[i]=std::nextafter(f.x[i],INFINITY);
  if(kind=="cancel") f.x[i]=i<2560?0.f:(i&1?-1.f:1.f)*(1.f+float(i%7)*0.0009765625f);
 }
 for(size_t i=0;i<P;++i){
  float a=float(int((i*17+i/10240*7)%31)-15)*0.0009765625f;
  float b=float(int((i*13+i/320*11)%37)-18)*0.00390625f;
  if(kind=="sparse"){a=i%1021?0.f:a;b=i%53?0.f:b;}
  if(kind=="cancel"){a=(i&1?-1.f:1.f)*(0.015625f+float(i%5)*0.0001220703125f);b=(i&1?-1.f:1.f)*0.03125f;}
  if(kind=="tails") b*=16.f;
  f.down[i]=native_hc_bf16_rne(a);f.up[i]=native_hc_bf16_rne(b);
 }
 if(kind=="tails") {std::fill(f.x.begin(),f.x.end(),0.5f);std::fill(f.w.begin(),f.w.end(),1.f);std::fill(f.down.begin(),f.down.end(),0);std::fill(f.up.begin(),f.up.end(),0);for(size_t i=0;i<D;++i)f.down[i]=native_hc_bf16_rne(0.03125f);f.up[0]=native_hc_bf16_rne(2.f);f.up[320]=native_hc_bf16_rne(-2.f);}
 if(kind=="cancel") {f.x[0]=100000000.f;f.x[8]=1.f;f.x[256]=-100000000.f;f.w[0]=f.w[8]=f.w[256]=1.f;for(size_t i=0;i<D;++i)f.down[i]=0;f.down[0]=f.down[8]=f.down[256]=native_hc_bf16_rne(1.f);}
 return f;
}
inline void write(const std::string& path,const Fixture& f){native_hc_need(!std::filesystem::exists(path),"head fixture: output exists");std::ofstream out(path,std::ios::binary);out.write("HCHD001\0",8);for(float x:f.x)word(out,bits(x));for(float x:f.w)word(out,bits(x));for(const auto* v:{&f.down,&f.up})for(auto h:*v){out.put(char(h));out.put(char(h>>8));}native_hc_need(bool(out),"head fixture: write");}
inline Fixture read(const std::string& path){native_hc_need(std::filesystem::file_size(path)==8+2*D*4+2*P*2,"head fixture: length");std::ifstream in(path,std::ios::binary);char m[8];in.read(m,8);native_hc_need(std::memcmp(m,"HCHD001\0",8)==0,"head fixture: magic");Fixture f;f.x.resize(D);f.w.resize(D);f.down.resize(P);f.up.resize(P);for(auto* v:{&f.x,&f.w})for(float& x:*v){x=value(word(in));native_hc_normal(x);}for(auto* v:{&f.down,&f.up})for(auto& h:*v){int a=in.get(),b=in.get();native_hc_need(a!=EOF&&b!=EOF,"head fixture: read");h=uint16_t(a|(b<<8));native_hc_normal(value(uint32_t(h)<<16));}native_hc_need(in.peek()==EOF,"head fixture: trailing data");return f;}
}
