#pragma once
// Bounded standalone input for the native projection-input q8_1 launcher.
// Native Strata source remains the method and the CUDA launcher remains oracle.
#include "native_q8_fixture.hpp"
#include <filesystem>
#include <sstream>
namespace native_projection_q8 {
using namespace native_q8_fixture;
constexpr size_t elements=6144, blocks=elements/32, bytes=blocks*36;
constexpr uint32_t canary=0xa5a5a5a5;
inline void validate(const std::vector<float>& x) {
    need(x.size()==elements,"projection q8: expected exactly 6144 values");
#ifndef STRATA_PROJECTION_Q8_REPLAY
    std::array<uint32_t,8*32> fixed{};
    std::fill(fixed.begin()+32,fixed.begin()+64,0x80000000u);
    fixed[64]=bits(127.f);fixed[65]=bits(.5f);fixed[66]=bits(-.5f);fixed[67]=bits(1.5f);fixed[68]=bits(-1.5f);
    fixed[96]=bits(127.f);fixed[97]=bits(.03125f);
    fixed[128]=bits(1e8f);fixed[129]=bits(1.f);fixed[130]=bits(-1e8f);fixed[131]=bits(1.f);
    fixed[160]=bits(2e7f);fixed[192]=bits(-127.f);fixed[193]=bits(-.03125f);
    fixed[224]=bits(127.06201171875f);
    for(size_t i=0;i<fixed.size();++i)need(bits(x[i])==fixed[i],"projection q8: pinned input changed");
 #endif
    for(size_t b=0;b<blocks;++b) {
        std::array<float,32> sum{},maximum{};
        for(size_t i=0;i<32;++i) {
            float v=x[b*32+i];
            need(std::isfinite(v) && (v==0 || std::isnormal(v)),"projection q8: nonfinite/subnormal input");
            sum[i]=v;maximum[i]=std::fabs(v);
        }
        for(size_t o=16;o;o>>=1) {auto old=sum;auto m=maximum;for(size_t i=0;i<32;++i){
            sum[i]=old[i]+old[i^o];maximum[i]=std::fmax(m[i],m[i^o]);
            need(std::isfinite(sum[i]) && (sum[i]==0 || std::isnormal(sum[i])),"projection q8: unsupported sum");
        }}
        if(maximum[0]) {float d=finite_half(maximum[0]/127.f);
            need(std::isnormal(d),"projection q8: unsupported scale");
            for(size_t i=0;i<32;++i){float q=x[b*32+i]/d;
                need(std::isfinite(q) && (q==0 || std::isnormal(q)),"projection q8: unsupported ratio");
            }
        }
    }
}
inline std::vector<float> synthetic(unsigned mode) {
    need(mode<5,"projection q8: fixture mode");std::vector<float> x(elements,0.f);
    for(size_t i=0;i<elements;++i) {
        if(mode==1 && i%97==0)x[i]=float(int(i%31)-15);
        if(mode==2)x[i]=float(int((i*37)%251)-125)*0.125f;
        if(mode==3)x[i]=i%4==0?0x1p24f:i%4==1?1.f:i%4==2?-0x1p24f:1.f;
        if(mode==4)x[i]=float(int((i*19)%127)-63)*0.03125f;
    }
    auto clear=[&](size_t b){std::fill_n(x.begin()+b*32,32,0.f);};
    clear(0);clear(1);std::fill_n(x.begin()+32,32,-0.f);
    clear(2);x[64]=127.f;x[65]=.5f;x[66]=-.5f;x[67]=1.5f;x[68]=-1.5f;
    clear(3);x[96]=127.f;x[97]=0.03125f; // raw sum exactly halfway between adjacent half values
    clear(4);x[128]=1e8f;x[129]=1.f;x[130]=-1e8f;x[131]=1.f;
    clear(5);x[160]=2e7f; // finite scale and sum clamp
    clear(6);x[192]=-127.f;x[193]=-0.03125f; // negative half tie
    clear(7);x[224]=127.06201171875f; // scale is exactly halfway above one
    validate(x);return x;
}
#ifdef STRATA_PROJECTION_Q8_REPLAY
constexpr uint32_t fixture_version=2;
#else
constexpr uint32_t fixture_version=1;
#endif
inline void write(const std::string& path,const std::vector<float>& x) {
    validate(x);need(!std::filesystem::exists(path),"projection q8: fixture exists");
    std::ofstream o(path,std::ios::binary);word(o,0x31513850);word(o,fixture_version);word(o,uint32_t(x.size()));
    for(float v:x)word(o,bits(v));need(bool(o),"projection q8: fixture write");
}
inline std::vector<float> read(const std::string& path) {
    std::ifstream i(path,std::ios::binary);need(bool(i)&&word(i)==0x31513850&&word(i)==fixture_version&&word(i)==elements,"projection q8: bad fixture header");
    std::vector<float>x(elements);for(float&v:x)v=value(word(i));need(i.peek()==EOF,"projection q8: trailing fixture data");validate(x);return x;
}
inline void contract(const std::vector<float>& x,const std::string& path) {
    validate(x);need(!std::filesystem::exists(path),"projection q8: contract exists");std::ofstream o(path);
    o<<"{\"fields\":{\"qx.bin\":"<<bytes<<"},\"qualified\":false}\n";need(bool(o),"projection q8: contract write");
}
inline Bytes load(const std::filesystem::path& p) {
    std::ifstream i(p,std::ios::binary);need(bool(i),"projection q8: missing output");Bytes v(bytes);i.read((char*)v.data(),v.size());need(size_t(i.gcount())==v.size()&&i.peek()==EOF,"projection q8: output size");return v;
}
inline void save(const std::filesystem::path&p,const Bytes&v) {
    need(v.size()==bytes&&!std::filesystem::exists(p),"projection q8: output exists or wrong size");std::ofstream o(p,std::ios::binary);o.write((const char*)v.data(),v.size());need(bool(o),"projection q8: output write");
}
// Independent literal vectors: no host quantizer is used to form these bytes.
inline void pinned(const Bytes& v) {
    need(v.size()==bytes,"projection q8: packed size");
    auto check=[&](size_t b,uint16_t d,uint16_t s,std::initializer_list<int> first) {
        std::array<uint8_t,36>w{};w[0]=uint8_t(d);w[1]=uint8_t(d>>8);w[2]=uint8_t(s);w[3]=uint8_t(s>>8);
        size_t i=4;for(int q:first)w[i++]=uint8_t(q);
        need(std::equal(w.begin(),w.end(),v.begin()+b*36),"projection q8: independent packed control failed");
    };
    check(0,0,0,{});check(1,0,0x8000,{});check(2,0x3c00,0x57f0,{127,1,-1,2,-2});
    check(3,0x3c00,0x57f0,{127,0});check(4,0x7bff,0x4000,{127,0,-127,0});
    check(5,0x7bff,0x7bff,{127});check(6,0x3c00,0xd7f0,{-127,0});check(7,0x3c00,0x57f1,{127});
}
// Diagnostic only: intermediate values are host RNE, not native instruction snapshots.
inline void trace(const std::vector<float>& x,const std::filesystem::path& p) {
    validate(x);need(!std::filesystem::exists(p),"projection q8: trace exists");std::ofstream o(p);need(bool(o),"projection q8: trace create");
    o<<"block,lane,amax_bits,raw_sum_bits,raw_scale_bits,scale_bits,ratio_bits,rounded_bits,code\n";
    for(size_t b=0;b<blocks;++b){std::array<float,32> m{},s{};for(size_t l=0;l<32;++l){m[l]=fabsf(x[b*32+l]);s[l]=x[b*32+l];}
        for(size_t j=16;j;j>>=1){auto om=m,os=s;for(size_t l=0;l<32;++l){m[l]=fmaxf(om[l],om[l^j]);s[l]=os[l]+os[l^j];}}
        float raw_scale=m[0]/127.f;float d=finite_half(raw_scale);for(size_t l=0;l<32;++l){float r=m[0]?x[b*32+l]/d:0.f;float q=roundf(r);
            o<<b<<','<<l<<','<<bits(m[0])<<','<<bits(s[0])<<','<<bits(raw_scale)<<','<<bits(d)<<','<<bits(r)<<','<<bits(q)<<','<<int(std::max(-127.f,std::min(127.f,q)))<<'\n';}
    }need(bool(o),"projection q8: trace write");
}
}
