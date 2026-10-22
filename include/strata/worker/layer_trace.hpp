// Bounded first-token MoE boundary records, opt-in experiment only.
#pragma once
#include "strata/worker/gpu_capture.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
namespace strata::worker::diagnostic::layer {
constexpr uint32_t max_layers=128;
constexpr size_t record_cap=256u<<10;
enum Kind : uint32_t { Host=1, Device=2, Gpu=3, Window=4, Partition=5 };
struct Record {
    uint32_t kind=0,layer=0,layers=0,H=0,K=0,experts=0;
    std::vector<std::vector<uint8_t>> fields;
};
template<class T> std::vector<uint8_t> bytes(const T* p,size_t n) {
    std::vector<uint8_t> out(n*sizeof(T)); if(n)std::memcpy(out.data(),p,out.size());return out;
}
inline bool geometry(const Record& r) {
    return r.kind>=Host && r.kind<=Partition && r.layers>0 && r.layers<=max_layers && r.layer<r.layers &&
        r.H>0 && r.H<=2560 && r.K>0 && r.K<=10 && r.experts>0 && r.experts<=512;
}
inline std::vector<uint8_t> encode(const Record& r) {
    if(!geometry(r)||r.fields.size()>8)throw std::runtime_error("layer trace geometry/field cap");
    std::vector<uint8_t> out{'S','G','L','A','Y','0','0','1'};
    for(uint32_t x:{r.kind,r.layer,r.layers,r.H,r.K,r.experts,(uint32_t)r.fields.size()})gpu_codec::put(out,x,4);
    for(const auto& field:r.fields){if(field.size()>record_cap)throw std::runtime_error("layer field cap");section(out,field);}
    if(out.size()>record_cap)throw std::runtime_error("layer record cap");return out;
}
inline bool decode(const std::vector<uint8_t>& raw,Record& r,std::string& error) {
    if(raw.size()<36||raw.size()>record_cap||std::memcmp(raw.data(),"SGLAY001",8)){error="layer record header";return false;}
    gpu_codec::Cursor c{raw};c.i=8;
    r.kind=c.get(4);r.layer=c.get(4);r.layers=c.get(4);r.H=c.get(4);r.K=c.get(4);r.experts=c.get(4);
    const auto n=c.get(4);if(!geometry(r)||n>8){error="layer record geometry";return false;}
    r.fields.assign(n,{});for(auto& field:r.fields)if(!section(c,field)){error="layer record field";return false;}
    if(!c.done()){error="layer record trailing bytes";return false;}return true;
}
inline std::string name(uint32_t kind,uint32_t layer) {
    const char* names[]={"invalid","host","device","gpu","window","partition"};
    if(kind<Host||kind>Partition||layer>=max_layers)throw std::runtime_error("layer filename bounds");
    char out[40];std::snprintf(out,sizeof out,"%s-%03u.trace",names[kind],layer);return out;
}
inline bool write(const char* directory,const Record& r,std::string& error) {
    auto raw=encode(r);return file(directory,name(r.kind,r.layer).c_str(),raw,true,error);
}
inline bool read(const std::string& directory,uint32_t kind,uint32_t layer,Record& r,std::string& error) {
    std::vector<uint8_t> raw;
    if(!file(directory,name(kind,layer).c_str(),raw,false,error)||raw.size()>record_cap||!decode(raw,r,error))return false;
    if(r.kind!=kind||r.layer!=layer){error="layer filename/identity mismatch";return false;}return true;
}
template<class T> std::vector<T> values(const std::vector<uint8_t>& raw,size_t n) {
    if(raw.size()!=n*sizeof(T))throw std::runtime_error("layer field shape");
    std::vector<T> out(n);if(n)std::memcpy(out.data(),raw.data(),raw.size());return out;
}
inline std::vector<float> finite(const std::vector<uint8_t>& raw,size_t n) {
    auto out=values<float>(raw,n);for(float x:out)if(!std::isfinite(x))throw std::runtime_error("nonfinite layer field");return out;
}
struct Difference {bool bitwise,material;double maximum,relative,rmse,limit;};
inline Difference difference(const std::vector<float>& a,const std::vector<float>& b) {
    if(a.empty()||a.size()!=b.size())throw std::runtime_error("layer comparison shape");
    double maximum=0,magnitude=0,absolute=0,denom=0,square=0;
    for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite layer comparison");
        const double d=std::fabs((double)b[i]-a[i]);maximum=std::max(maximum,d);magnitude=std::max(magnitude,std::fabs((double)a[i]));
        absolute+=d;denom+=std::fabs((double)a[i]);square+=d*d;}
    const double limit=1e-5*std::max(1.,magnitude);
    return {std::memcmp(a.data(),b.data(),a.size()*4)==0,maximum>limit,maximum,absolute/std::max(denom,1e-12),std::sqrt(square/a.size()),limit};
}
} // namespace strata::worker::diagnostic::layer
