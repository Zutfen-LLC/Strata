#pragma once
#include "native_q8_fixture.hpp"
#include "../../src/core/native_hc_layout.hpp"
#include <filesystem>
namespace native_hc_fixture {
using namespace strata::core::portable_detail;
using native_q8_fixture::word;using native_q8_fixture::bits;using native_q8_fixture::value;
struct Fixture {uint32_t t=1;std::vector<float> x,w;std::vector<uint16_t> wi;};
inline void validate(const Fixture& f){native_hc_shape(2560,4,f.t);(void)native_hc_oracle(f.x,f.w,f.wi,f.t);}
inline Fixture synthetic(uint32_t t=8){Fixture f;f.t=t;f.x.resize(size_t(t)*10240);f.w.resize(10240);f.wi.resize(40960);
 for(size_t i=0;i<f.x.size();++i){const auto token=i/10240;float v=float(int((i*73)%257)-128)*0.0078125f;if(token==0)v=0;if(token==1)v=i%2?-0.f:0.f;if(token==2)v=i%2?0.125f:-0.125f;if(token==3&&v!=0)v=std::nextafter(v,INFINITY);if(token==4)v*=1024.f;f.x[i]=v;}
 for(size_t i=0;i<f.w.size();++i)f.w[i]=0.75f+float(i%13)*0.03125f;
 for(size_t i=0;i<f.wi.size();++i)f.wi[i]=native_hc_bf16_rne(float(int((i*31)%97)-48)*0.001953125f);
 validate(f);return f;
}
inline void write_fixture(const std::string& path,const Fixture& f){validate(f);native_hc_need(!std::filesystem::exists(path),"native HC: fresh fixture required");std::ofstream o(path,std::ios::binary);word(o,0x3143484e);word(o,1);word(o,2560);word(o,4);word(o,f.t);word(o,bits(1e-6f));word(o,30);word(o,0);
 for(float v:f.x)word(o,bits(v));for(float v:f.w)word(o,bits(v));for(size_t i=0;i<f.wi.size();i+=2)word(o,uint32_t(f.wi[i])|(uint32_t(f.wi[i+1])<<16));native_hc_need(bool(o),"native HC: fixture write");}
inline Fixture read_fixture(const std::string& path){std::ifstream in(path,std::ios::binary);native_hc_need(bool(in),"native HC: fixture read");native_hc_need(word(in)==0x3143484e&&word(in)==1,"native HC: fixture header");auto h=word(in),c=word(in);Fixture f;f.t=word(in);auto eps=value(word(in));auto type=word(in),flags=word(in);native_hc_shape(h,c,f.t,eps,flags!=0,type);f.x.resize(size_t(f.t)*10240);f.w.resize(10240);f.wi.resize(40960);for(auto& v:f.x)v=value(word(in));for(auto& v:f.w)v=value(word(in));for(size_t i=0;i<f.wi.size();i+=2){auto v=word(in);f.wi[i]=v;f.wi[i+1]=v>>16;}native_hc_need(in.peek()==EOF,"native HC: fixture trailing bytes");validate(f);return f;}
inline void floats(const std::string& path,const std::vector<float>& v){std::ofstream f(path,std::ios::binary);native_hc_need(bool(f),"native HC: output open");for(auto x:v)word(f,bits(x));native_hc_need(bool(f),"native HC: output write");}
}
