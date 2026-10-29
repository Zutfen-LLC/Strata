#pragma once
// Synthetic, bounded interchange shared by the native oracle and Vulkan runner.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace native_q8_fixture {
inline void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
inline uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
inline float value(uint32_t b){float x;std::memcpy(&x,&b,4);return x;}
inline void word(std::ostream& f,uint32_t n){for(int i=0;i<4;++i)f.put(char(n>>(8*i)));}
inline uint32_t word(std::istream& f){uint32_t n=0;for(int i=0;i<4;++i){int c=f.get();need(c!=EOF,"truncated input");n|=uint32_t(c)<<(8*i);}return n;}
struct Fixture {uint32_t n_ff=64;std::vector<int32_t> starts{2,5,5,9};std::vector<float> x,gate,up;};
inline Fixture synthetic() {
    Fixture f;f.x.resize(13*64);f.gate.resize(f.x.size());f.up.resize(f.x.size());
    for(size_t i=0;i<f.x.size();++i){f.x[i]=float(int(i%61)-30)*.125f;
        f.gate[i]=float(int(i%197)-98)*.25f;f.up[i]=float(int(i%47)-23)*.125f;}
    std::fill_n(f.x.begin(),32,0.f);std::fill_n(f.x.begin()+32,32,-0.f);
    for(int i=0;i<32;++i)f.x[64+i]=i%2?float(i):-float(i);
    std::fill_n(f.x.begin()+96,32,0.f);f.x[96]=127.f;
    f.x[97]=.5f;f.x[98]=-.5f;f.x[99]=1.5f;f.x[100]=-1.5f;
    for(int i=0;i<32;++i)f.x[128+i]=i%4==0?1e20f:i%4==1?1.f:i%4==2?-1e20f:1.f;
    for(int i=0;i<32;++i)f.x[160+i]=value(uint32_t(i+1));
    std::fill_n(f.x.begin()+192,32,65504.f); // sum clamp
    std::fill_n(f.x.begin()+224,32,8388608.f); // scale clamp and q clamp
    f.x[256]=value(0x7fc12345);f.x[257]=1.f; // fmaxf numeric vs NaN
    f.x[288]=value(0x7f800000);f.x[289]=value(0xff800000);
    std::fill_n(f.x.begin()+320,32,value(0x7fc12345));
    for(int i=0;i<32;++i)f.x[352+i]=value(0x00800000u+uint32_t(i));
    // Half rounding boundaries in raw sums, cancellation and scales.
    for(int b=12;b<18;++b){std::fill_n(f.x.begin()+b*32,32,0.f);
        f.x[b*32]=b==12?65519.f:b==13?65520.f:b==14?-65520.f:
            b==15?value(0x33800000):b==16?value(0x33000000):1.00048828125f;}
    f.gate[128]=0.f;f.gate[129]=-0.f;f.gate[130]=value(1);f.gate[131]=-value(1);
    f.gate[192]=100.f;f.gate[193]=-100.f;f.gate[194]=value(0x7f800000);
    f.gate[195]=value(0xff800000);f.gate[196]=value(0x7fc12345);
    f.up[197]=value(0x7f800000);f.gate[197]=0.f;
    return f;
}
inline void write_fixture(const std::string& path,const Fixture& f){
    std::ofstream o(path,std::ios::binary);need(bool(o),"cannot create fixture");
    word(o,0x3151384e);word(o,f.n_ff);word(o,uint32_t(f.x.size()));word(o,uint32_t(f.starts.size()));
    for(int32_t n:f.starts)word(o,uint32_t(n));
    for(const auto* v:{&f.x,&f.gate,&f.up})for(float x:*v)word(o,bits(x));need(bool(o),"fixture write failed");
}
inline Fixture read_fixture(const std::string& path){
    std::ifstream i(path,std::ios::binary);need(bool(i),"cannot read fixture");Fixture f;
    need(word(i)==0x3151384e,"bad fixture magic");f.n_ff=word(i);uint32_t n=word(i),ns=word(i);
    need(n && n<=1048576 && f.n_ff && f.n_ff%32==0 && n%f.n_ff==0 && ns && ns<=15,"unbounded fixture");
    f.starts.resize(ns);for(auto& s:f.starts)s=int32_t(word(i));
    for(size_t j=0;j<ns;++j)need(f.starts[j]>=0 && uint32_t(f.starts[j])<=n/f.n_ff &&
        (!j || f.starts[j]>=f.starts[j-1]),"invalid group interval");
    for(auto* v:{&f.x,&f.gate,&f.up}){v->resize(n);for(float& x:*v)x=value(word(i));}
    need(i.peek()==EOF,"trailing fixture bytes");return f;
}
using Bytes=std::vector<uint8_t>;
struct Result {Bytes input,hidden,separate,fused,empty;};
inline Result blank(size_t n){return {Bytes(n/32*36,0xa5),Bytes(n*4,0xa5),Bytes(n/32*36,0xa5),
    Bytes(n/32*36,0xa5),Bytes(n/32*36,0xa5)};}
inline void write_result(const std::string& path,const Result& r){
    std::ofstream o(path,std::ios::binary);need(bool(o),"cannot create result");word(o,0x3152384e);
    for(const auto* b:{&r.input,&r.hidden,&r.separate,&r.fused,&r.empty}){
        word(o,uint32_t(b->size()));o.write(reinterpret_cast<const char*>(b->data()),b->size());}
    need(bool(o),"result write failed");
}
// Host arithmetic specification only. CUDA oracle, not this model, decides GPU parity.
inline uint16_t half_rne(float x){
    uint32_t b=bits(x),s=(b>>16)&0x8000,e=(b>>23)&255,m=b&0x7fffff;
    if(e==255)return uint16_t(s|0x7c00|(m?0x200|(m>>13):0));
    if(e>142)return uint16_t(s|0x7c00);if(e<102)return uint16_t(s);
    uint32_t sh=e<113?126-e:13,sig=m|0x800000,q=sig>>sh,r=sig&((1u<<sh)-1),h=1u<<(sh-1);
    q+=r>h || (r==h && (q&1));return uint16_t(s|(e<113?q:((e-113)<<10)+q));
}
inline float finite_half(float x){return std::fabs(x)>65504.f?std::copysign(65504.f,x):x;}
inline Bytes quantize(const std::vector<float>& x){
    need(x.size()%32==0,"partial native block");Bytes out(x.size()/32*36);
    for(size_t b=0;b<x.size()/32;++b){std::array<float,32> m,s;
        for(int l=0;l<32;++l){m[l]=std::fabs(x[b*32+l]);s[l]=x[b*32+l];}
        for(int o=16;o;o>>=1){auto old_m=m,old_s=s;for(int l=0;l<32;++l){m[l]=std::fmax(old_m[l],old_m[l^o]);s[l]=old_s[l]+old_s[l^o];}}
        float d=finite_half(m[0]/127.f);uint16_t hd=half_rne(finite_half(d)),hs=half_rne(finite_half(s[0]));
        out[b*36]=uint8_t(hd);out[b*36+1]=uint8_t(hd>>8);out[b*36+2]=uint8_t(hs);out[b*36+3]=uint8_t(hs>>8);
        for(int l=0;l<32;++l){float q=std::round(x[b*32+l]/d);q=q>127.f?127.f:q< -127.f?-127.f:q;
            out[b*36+4+l]=m[0]==0.f || std::isnan(q)?0:uint8_t(int(q));}
    }return out;
}
}
