#pragma once
#include "native_q8_fixture.hpp"
#include "../../src/core/native_router_layout.hpp"
#include <filesystem>
#include <set>
namespace native_router_fixture {
using native_q8_fixture::need;
using native_q8_fixture::bits;
using native_q8_fixture::value;
using native_q8_fixture::word;
constexpr size_t prefixes[3]={3,5,7};
struct Fixture {uint32_t tokens=8;std::vector<float> logits;};
inline void validate(const Fixture& f) {
    strata::core::portable_detail::native_router_shape(512,10,f.tokens);
    need(f.logits.size()==size_t(f.tokens)*512,"router fixture shape");
    for(float x:f.logits)need(std::isfinite(x),"router fixture requires finite logits");
}
inline Fixture synthetic() {
    Fixture f;f.logits.resize(f.tokens*512);
    // Equal scores; signed-zero ties; repeated winning expert lanes; exact
    // and adjacent ties; dynamic ranges; overflowed subtraction; uneven lanes.
    for(size_t t=0;t<f.tokens;++t)for(size_t e=0;e<512;++e) {
        float x=0;
        switch(t) {
        case 0:x=0;break;
        case 1:x=e%2?-0.f:0.f;break;
        case 2:x=-16.f;if(e%32==7 && e/32<10)x=8.f-float(e/32)*0.125f;break;
        case 3:x=float(int((e*73)%509)-254)*0.03125f;break;
        case 4:x=e<16?std::nextafter(1.f,2.f):1.f;break;
        case 5:x=e==511?0.f:-104.f;break;
        case 6:x=e==33?std::numeric_limits<float>::max():-std::numeric_limits<float>::max();break;
        case 7:x=-float((e*31)%512)*0.125f;break;
        }
        f.logits[t*512+e]=x;
    }
    validate(f);return f;
}
inline void write_fixture(const std::string& path,const Fixture& f) {
    validate(f);need(!std::filesystem::exists(path),"router refuses existing fixture");
    std::ofstream out(path,std::ios::binary);need(bool(out),"router fixture create");
    word(out,0x3154524e);word(out,1);word(out,f.tokens);word(out,512);word(out,10);
    for(float x:f.logits)word(out,bits(x));need(bool(out),"router fixture write");
}
inline Fixture read_fixture(const std::string& path) {
    std::ifstream in(path,std::ios::binary);need(bool(in),"router fixture read");
    need(word(in)==0x3154524e && word(in)==1,"router fixture header");Fixture f;f.tokens=word(in);
    need(word(in)==512 && word(in)==10,"router fixture geometry");
    strata::core::portable_detail::native_router_shape(512,10,f.tokens);
    f.logits.resize(size_t(f.tokens)*512);for(float& x:f.logits)x=value(word(in));
    need(in.peek()==EOF,"router fixture trailing bytes");validate(f);return f;
}
inline std::array<std::vector<uint32_t>,3> padded(const Fixture& f) {
    validate(f);std::array<std::vector<uint32_t>,3> result;
    for(size_t i=0;i<3;++i)result[i].assign(prefixes[i]+size_t(f.tokens)*(i==0?512:10)+9,0xa5a5a5a5);
    for(size_t e=0;e<f.logits.size();++e)result[0][prefixes[0]+e]=bits(f.logits[e]);return result;
}
inline void reserve(const std::string& dir) {
    need(std::filesystem::create_directory(dir),"router refuses existing output directory");
}
inline void receipt(const std::string& dir,const Fixture& f,
    const std::array<std::vector<uint32_t>,3>& before,const std::array<std::vector<uint32_t>,3>& after) {
    bool guards=true,valid=true;
    for(size_t i=0;i<3;++i) {
        need(before[i].size()==after[i].size(),"router readback size");
        std::ofstream raw(dir+"/view-"+std::to_string(i)+".bin",std::ios::binary);
        need(bool(raw),"router views create");
        for(size_t j=0;j<after[i].size();++j) {
            word(raw,after[i][j]);
            if(i==0 || j<prefixes[i] || j>=prefixes[i]+f.tokens*10)guards&=after[i][j]==before[i][j];
        }
        need(bool(raw),"router views write");
    }
    std::ofstream ids(dir+"/ids.i32",std::ios::binary),weights(dir+"/weights.f32",std::ios::binary);
    need(bool(ids)&&bool(weights),"router outputs create");
    for(size_t t=0;t<f.tokens;++t) {
        std::set<uint32_t> unique;
        for(size_t k=0;k<10;++k) {
            auto id=after[1][prefixes[1]+t*10+k],w=after[2][prefixes[2]+t*10+k];
            valid&=id<512 && unique.insert(id).second && std::isfinite(value(w)) && value(w)>=0.f;
            word(ids,id);word(weights,w);
        }
    }
    need(bool(ids)&&bool(weights),"router outputs write");
    std::ofstream report(dir+"/report.json");need(bool(report),"router report create");
    report<<"{\"observation_only\":true,\"qualified\":false,\"tokens\":"<<f.tokens
        <<",\"experts\":512,\"k\":10,\"sentinels_ok\":"<<(guards?"true":"false")
        <<",\"valid_output\":"<<(valid?"true":"false")<<",\"native_comparison\":\"required\"}\n";
    need(bool(report),"router report write");
    need(guards && valid,"router invalid output/input mutation; receipts retained");
}
// Host arithmetic witness only: receives already computed FP32 probabilities.
// It does not emulate CUDA expf/fast division and cannot qualify a GPU router.
struct Selection {std::array<int32_t,10> ids{};std::array<float,10> probabilities{};float selected_sum=0;};
inline float xor_sum(std::array<float,32> lane) {
    for(size_t mask=16;mask;mask>>=1) {
        auto previous=lane;
        for(size_t l=0;l<32;++l){volatile float x=previous[l]+previous[l^mask];lane[l]=x;}
    }
    return lane[0];
}
inline Selection select(std::array<float,512> probabilities) {
    Selection s;std::array<float,32> sums{};
    for(size_t k=0;k<10;++k) {
        int best=0;for(int e=1;e<512;++e)if(probabilities[e]>probabilities[best])best=e;
        s.ids[k]=best;s.probabilities[k]=probabilities[best];
        volatile float next=sums[best&31]+probabilities[best];sums[best&31]=next;
        probabilities[best]=-std::numeric_limits<float>::infinity();
    }
    s.selected_sum=std::max(xor_sum(sums),0x1p-14f);return s;
}
}
