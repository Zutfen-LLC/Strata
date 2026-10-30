#pragma once
#include "native_q8_fixture.hpp"
#include "../../src/core/native_moe_combine_layout.hpp"
#include <filesystem>
namespace native_moe_fixture {
using native_q8_fixture::need;
using native_q8_fixture::bits;
using native_q8_fixture::value;
using native_q8_fixture::word;
constexpr uint32_t H=2560,K=10;
constexpr size_t prefixes[4]={4,3,8,12};
struct Fixture {uint32_t tokens=2;std::vector<float> parts,weights,shared;};
inline void validate(const Fixture& f) {
    strata::core::portable_detail::native_moe_combine_shape(H,f.tokens);
    need(f.parts.size()==size_t(f.tokens)*K*H && f.weights.size()==size_t(f.tokens)*K &&
        f.shared.size()==size_t(f.tokens)*H,"combine fixture shape");
    for(const auto* v:{&f.parts,&f.weights,&f.shared})for(float x:*v)
        need(std::isfinite(x) && (x==0 || std::fpclassify(x)!=FP_SUBNORMAL),"combine fixture requires finite normal/zero inputs");
}
inline Fixture synthetic() {
    Fixture f;f.parts.resize(2*K*H);f.weights.resize(2*K);f.shared.resize(2*H);
    f.weights[0]=f.weights[1]=0x1.000002p-3f;
    for(size_t k=2;k<K;++k)f.weights[k]=0.09375f;
    for(size_t k=0;k<K;++k)f.weights[K+k]=k<8?0.125f:0.f;
    const float a=0x1.fffffcp+2f;
    for(size_t col=0;col<H;++col) {
        auto part=[&](size_t t,size_t k)->float&{return f.parts[(t*K+k)*H+col];};
        switch(col%4) {
        case 0:part(0,0)=a;part(0,1)=-a;break; // first product rounding vs fusing the first product
        case 1:part(0,0)=-8.f;part(0,1)=a;break; // strict FMA vs separate multiply/add
        case 2:for(size_t k=0;k<K;++k)part(0,k)=float(int(k)-4)*0.25f;f.shared[col]=-0.125f;break;
        case 3:part(0,0)=-0.f;f.shared[col]=0.f;break;
        }
        switch(col%4) {
        case 0:part(1,0)=8.f;part(1,1)=0x1p30f;part(1,2)=-0x1p30f;part(1,3)=8.f;break; // expert order
        case 1:part(1,0)=0x1p30f;part(1,1)=-0x1p30f;f.shared[H+col]=1.f;break; // shared added last
        case 2:for(size_t k=0;k<8;++k)part(1,k)=float((int(k)%3)-1)*float(col+1)*0.015625f;f.shared[H+col]=0.5f;break;
        case 3:part(1,0)=4.f;part(1,9)=0x1p20f;f.shared[H+col]=-0.25f;break; // zero routing weight is retained
        }
    }
    validate(f);return f;
}
inline std::vector<float> reference(const Fixture& f) {
    validate(f);std::vector<float> output(f.tokens*H);
    for(size_t t=0;t<f.tokens;++t)for(size_t h=0;h<H;++h) {
        volatile float first=f.parts[t*K*H+h]*f.weights[t*K];
        float sum=first;
        for(size_t k=1;k<K;++k)sum=std::fma(f.parts[(t*K+k)*H+h],f.weights[t*K+k],sum);
        volatile float combined=sum+f.shared[t*H+h];output[t*H+h]=combined;
    }
    return output;
}
inline void fresh(const std::string& path) {need(!std::filesystem::exists(path),"combine refuses existing output");}
inline void write_fixture(const std::string& path,const Fixture& f) {
    validate(f);fresh(path);std::ofstream out(path,std::ios::binary);need(bool(out),"combine fixture create failed");
    word(out,0x31434d4e);word(out,1);word(out,f.tokens);word(out,H);word(out,K);
    for(const auto* v:{&f.parts,&f.weights,&f.shared})for(float x:*v)word(out,bits(x));
    need(bool(out),"combine fixture write failed");
}
inline Fixture read_fixture(const std::string& path) {
    std::ifstream in(path,std::ios::binary);need(bool(in),"combine fixture read failed");
    need(word(in)==0x31434d4e && word(in)==1,"combine fixture header");Fixture f;f.tokens=word(in);
    need(word(in)==H && word(in)==K,"combine fixture geometry");
    strata::core::portable_detail::native_moe_combine_shape(H,f.tokens);
    f.parts.resize(size_t(f.tokens)*K*H);f.weights.resize(f.tokens*K);f.shared.resize(f.tokens*H);
    for(auto* v:{&f.parts,&f.weights,&f.shared})for(float& x:*v)x=value(word(in));
    need(in.peek()==EOF,"combine fixture trailing bytes");validate(f);return f;
}
inline std::array<std::vector<uint32_t>,4> padded(const Fixture& f) {
    std::array<std::vector<uint32_t>,4> result;
    const std::vector<float>* inputs[]={&f.parts,&f.weights,&f.shared};
    for(size_t i=0;i<4;++i) {
        const size_t count=i<3?inputs[i]->size():f.tokens*H;
        result[i].assign(prefixes[i]+count+7,0xa5a5a5a5);
        if(i<3)for(size_t j=0;j<count;++j)result[i][prefixes[i]+j]=bits((*inputs[i])[j]);
    }
    return result;
}
inline size_t check_and_write(const std::string& prefix,const Fixture& f,
    const std::array<std::vector<uint32_t>,4>& before,const std::array<std::vector<uint32_t>,4>& after) {
    const auto expected=reference(f);size_t different=0,nonfinite=0;bool sentinels=true;
    for(size_t i=0;i<4;++i) {
        need(before[i].size()==after[i].size(),"combine readback size mismatch");
        for(size_t j=0;j<before[i].size();++j)
            if(i<3 || j<prefixes[i] || j>=prefixes[i]+expected.size())sentinels&=before[i][j]==after[i][j];
    }
    std::ofstream got(prefix+".output.f32",std::ios::binary),ref(prefix+".reference.f32",std::ios::binary),views(prefix+".views.bin",std::ios::binary);
    need(bool(got)&&bool(ref)&&bool(views),"combine raw output create failed");
    for(size_t j=0;j<expected.size();++j){const uint32_t b=after[3][prefixes[3]+j];
        different+=b!=bits(expected[j]);nonfinite+=(b&0x7f800000u)==0x7f800000u;word(got,b);word(ref,bits(expected[j]));}
    for(const auto& v:after){word(views,uint32_t(v.size()));for(auto x:v)word(views,x);}
    need(bool(got)&&bool(ref)&&bool(views),"combine raw output write failed");
    std::ofstream report(prefix+".json");need(bool(report),"combine report create failed");
    report<<"{\"observation_only\":true,\"qualified\":false,\"quantizer_qualified\":false,\"fp32_denorm_mode\":\"unspecified\",\"tokens\":"<<f.tokens
          <<",\"h\":2560,\"k\":10,\"words\":"<<expected.size()<<",\"different_bits\":"<<different<<",\"nonfinite\":"<<nonfinite
          <<",\"sentinels_ok\":"<<(sentinels?"true":"false")<<",\"exact\":"<<(!different&&!nonfinite&&sentinels?"true":"false")<<"}\n";
    need(bool(report),"combine report write failed");
    need(sentinels,"combine sentinel/input mutation; raw receipts retained");
    need(!nonfinite,"combine nonfinite output; raw receipts retained");return different;
}
inline void fresh_outputs(const std::string& prefix) {
    for(const char* suffix:{".output.f32",".reference.f32",".views.bin",".json",".effective.spv"})fresh(prefix+suffix);
}
}
