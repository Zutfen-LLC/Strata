#pragma once
#include "native_q8_fixture.hpp"
#include "../../src/core/native_shared_layout.hpp"
#include <filesystem>
namespace native_shared_fixture {
using native_q8_fixture::need;
using native_q8_fixture::bits;
using native_q8_fixture::value;
using native_q8_fixture::word;
using namespace strata::core::portable_detail;
constexpr size_t N=15;
enum Slot {X,WG,WU,WD,WS,QX,GATE,UP,H,QH,DOWN,RAW,SIGMOID,OUT,DUMMY};
inline constexpr const char* names[N]={"x","wg","wu","wd","ws","qx","gate","up","h","qh","down","raw","sigmoid","out","dummy"};
using Images=std::array<std::vector<uint32_t>,N>;
inline size_t prefix(size_t i) {return 2*(i%4)+2;} // float2 aligned, descriptor-misaligned views
struct Fixture {uint32_t h=2560,f=640,t=8;Images data;};
inline std::array<size_t,N> counts(uint32_t h,uint32_t f,uint32_t t) {
    native_shared_shape(h,f,t);
    std::array<size_t,N> n{size_t(h)*t,size_t(native_shared_weight_bytes(h,f)/4),size_t(native_shared_weight_bytes(h,f)/4),
        size_t(native_shared_weight_bytes(f,h)/4),h/2,size_t(h/32)*9*t,size_t(f)*t,size_t(f)*t,size_t(f)*t,
        size_t(f/32)*9*t,size_t(h)*t,t,t,size_t(h)*t,1};
    uint64_t total=0;for(auto c:n)total+=(c+32)*4;
    need(total<=256ull*1024*1024,"shared fixture exceeds 256 MiB budget");return n;
}
inline void validate(const Fixture& f) {
    auto n=counts(f.h,f.f,f.t);
    for(size_t i=0;i<5;++i)need(f.data[i].size()==n[i],"shared fixture input length");
    for(auto b:f.data[X])need(std::isfinite(value(b)),"shared nonfinite activation");
    for(auto word:f.data[WS])for(unsigned shift:{0u,16u})need(((word>>shift)&0x7f80u)!=0x7f80u,"shared nonfinite BF16 weight");
    for(auto slot:{WG,WU,WD}) {
        const auto* bytes=reinterpret_cast<const unsigned char*>(f.data[slot].data());
        for(size_t i=0;i<f.data[slot].size()*4;i+=34) {
            unsigned half=bytes[i]|(unsigned(bytes[i+1])<<8);
            need((half&0x7c00)!=0x7c00,"shared nonfinite Q8 weight scale");
        }
    }
}
inline Fixture synthetic(uint32_t h=2560,uint32_t width=640,uint32_t t=8) {
    Fixture f;f.h=h;f.f=width;f.t=t;auto n=counts(h,width,t);
    for(size_t i=0;i<5;++i)f.data[i].resize(n[i]);
    for(size_t i=0;i<n[X];++i) {
        float x=float(int((i*73+11)%257)-128)*0.0078125f;
        if(i%7==0)x=std::nextafter(x,std::numeric_limits<float>::infinity());
        f.data[X][i]=bits(x);
    }
    // Include zeros, cancellation, and quantization ties in each token.
    for(size_t token=0;token<t;++token) {
        for(size_t j=0;j<32;++j)f.data[X][token*h+j]=bits(token%2?-0.f:0.f);
        if(h>=64) {f.data[X][token*h+32]=bits(127.f);f.data[X][token*h+33]=bits(.5f);f.data[X][token*h+34]=bits(-.5f);}
    }
    for(auto slot:{WG,WU,WD}) {
        auto* bytes=reinterpret_cast<unsigned char*>(f.data[slot].data());
        for(size_t b=0;b<f.data[slot].size()*4/34;++b) {
            unsigned half=0x1800u+unsigned((b+slot)%5)*0x400u;
            bytes[b*34]=half&255;bytes[b*34+1]=half>>8;
            for(size_t j=0;j<32;++j)bytes[b*34+2+j]=uint8_t(int((b*17+j*31+slot*11)%255)-127);
        }
    }
    for(size_t pair=0;pair<n[WS];++pair) {
        float a=float(int(pair%13)-6)*0.0009765625f,b=-a*.5f;
        f.data[WS][pair]=uint32_t(native_shared_promoted_bf16(a))|(uint32_t(native_shared_promoted_bf16(b))<<16);
    }
    validate(f);return f;
}
inline void write_fixture(const std::string& path,const Fixture& f) {
    validate(f);need(!std::filesystem::exists(path),"shared refuses existing fixture");
    std::ofstream out(path,std::ios::binary);need(bool(out),"shared fixture create");
    word(out,0x3153484e);word(out,1);word(out,f.h);word(out,f.f);word(out,f.t);word(out,8);word(out,1); // Q8_0, raw BF16
    for(size_t i=0;i<5;++i)for(auto b:f.data[i])word(out,b);need(bool(out),"shared fixture write");
}
inline Fixture read_fixture(const std::string& path) {
    std::ifstream in(path,std::ios::binary);need(bool(in),"shared fixture open");
    need(word(in)==0x3153484e && word(in)==1,"shared fixture header");Fixture f;f.h=word(in);f.f=word(in);f.t=word(in);
    need(word(in)==8 && word(in)==1,"shared fixture requires Q8_0 projections and raw BF16 scalar weights");
    auto n=counts(f.h,f.f,f.t);
    for(size_t i=0;i<5;++i){f.data[i].resize(n[i]);for(auto& b:f.data[i])b=word(in);}
    need(in.peek()==EOF,"shared fixture trailing bytes");validate(f);return f;
}
inline Images padded(const Fixture& f) {
    validate(f);auto n=counts(f.h,f.f,f.t);Images result;
    for(size_t i=0;i<N;++i){result[i].assign(prefix(i)+n[i]+9,0xa5a5a5a5u);
        if(i<5)std::copy(f.data[i].begin(),f.data[i].end(),result[i].begin()+prefix(i));}
    return result;
}
inline void reserve(const std::string& path) {need(std::filesystem::create_directory(path),"shared refuses existing output directory");}
inline void receipt(const std::string& dir,const Fixture& f,const Images& before,const Images& after) {
    auto n=counts(f.h,f.f,f.t);bool guards=true,finite=true;
    for(size_t i=0;i<N;++i) {
        need(after[i].size()==before[i].size(),"shared readback size");
        std::ofstream out(dir+"/"+names[i]+".bin",std::ios::binary);need(bool(out),"shared receipt open");
        for(size_t j=0;j<after[i].size();++j) {
            if(i<5 || i==DUMMY || j<prefix(i) || j>=prefix(i)+n[i])guards&=after[i][j]==before[i][j];
            if(j>=prefix(i) && j<prefix(i)+n[i]) {
                word(out,after[i][j]);
                if(i>=GATE && i!=QH && i!=DUMMY)finite&=std::isfinite(value(after[i][j]));
            }
        }
        need(bool(out),"shared receipt write");
    }
    std::ofstream report(dir+"/report.json");need(bool(report),"shared report open");
    report<<"{\"observation_only\":true,\"qualified\":false,\"h\":"<<f.h<<",\"f\":"<<f.f<<",\"tokens\":"<<f.t
        <<",\"sentinels_ok\":"<<(guards?"true":"false")<<",\"finite\":"<<(finite?"true":"false")<<"}\n";
    need(bool(report),"shared report write");need(guards && finite,"shared invalid output/guards; receipts retained");
}
inline void arguments(int argc,char** argv) {
    need(argc==9 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--fixture" && std::string(argv[7])=="--output",
        "usage: native_shared_{component,reference} --device INDEX --expect-device DESCRIPTION --fixture FILE --output NEW_DIR");
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"shared invalid device index");
    need(std::string(argv[4]).size()>0,"shared requires expected device description");
}
}
