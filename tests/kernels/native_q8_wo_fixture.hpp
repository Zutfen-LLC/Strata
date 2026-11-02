#pragma once
#include "native_q8_fixture.hpp"
#include "native_q8_wo_layout.hpp"
#include <array>
#include <filesystem>
namespace native_q8_wo_fixture {
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
constexpr size_t blocks=192, lanes=32, max_rows=2560;
inline void set16(Bytes& b,size_t i,uint16_t v){b.at(i)=uint8_t(v);b.at(i+1)=uint8_t(v>>8);}
inline uint16_t get16(const Bytes& b,size_t i){return uint16_t(b.at(i))|uint16_t(b.at(i+1))<<8;}
inline uint16_t get16(const uint8_t* b,size_t i){return uint16_t(b[i])|uint16_t(b[i+1])<<8;}
inline float half(uint16_t h){unsigned e=(h>>10)&31,m=h&1023;
    need(e!=31,"WO: unsupported half scale");
    if(e==0 && m){float v=std::ldexp(float(m),-24);return h&0x8000?-v:v;}
    return value((uint32_t(h&0x8000)<<16)|(e?(uint32_t(e+112)<<23)|(uint32_t(m)<<13):0));
}
inline float input_half(uint16_t h){need(((h>>10)&31)!=0 || (h&1023)==0,"WO: unsupported half scale in Q8_1 input");return half(h);}
inline float checked(float x){need(std::isfinite(x)&&(x==0.f||std::isnormal(x)),"WO: unsupported intermediate");return x;}
inline int s8(uint8_t x){return x<128?int(x):int(x)-256;}
struct Fixture {size_t rows;Bytes weights,input;};
inline Fixture synthetic(size_t rows){
    need(rows>=8 && rows<=max_rows,"WO: synthetic rows");Fixture f{rows,Bytes(rows*q8_wo_row_bytes),Bytes(q8_wo_input_bytes)};
    for(size_t b=0;b<blocks;++b){set16(f.input,b*36,0x3c00);set16(f.input,b*36+2,0x3555);
        for(size_t j=0;j<32;++j){int q=int((b*11+j*7)%15)-7;f.input[b*36+4+j]=uint8_t(q?q:8);}
        f.input[b*36+4]=1;
    }
    for(size_t q=0;q<8;++q)f.input[36+4+q*4]=uint8_t(q+1);
    f.input[4]=uint8_t(-128);f.input[5]=127;f.input[6]=1;f.input[7]=uint8_t(-1);
    set16(f.input,2*36,0x6c00);set16(f.input,10*36,0x6c00); // 4096
    for(size_t r=0;r<rows;++r)for(size_t b=0;b<blocks;++b){size_t at=r*q8_wo_row_bytes+b*34;set16(f.weights,at,0x3c00);
        if(r>=8)for(size_t j=0;j<32;++j){int q=int((r*3+b*5+j)%15)-7;f.weights[at+2+j]=uint8_t(q?q:8);}
        if(r==8)for(size_t j=0;j<32;++j)f.weights[at+2+j]=uint8_t(s8(f.input[b*36+4+j])<0?-1:1);
    }
    auto put=[&](size_t row,size_t b,size_t j,int q){f.weights[row*q8_wo_row_bytes+b*34+2+j]=uint8_t(q);};
    put(1,1,0,1);put(2,0,0,-128);put(2,0,1,127);put(2,0,2,-1);put(2,0,3,2);
    put(3,1,0,3);set16(f.weights,3*q8_wo_row_bytes+34,0x3800); // 0.5
    put(4,1,0,1);put(4,9,0,-1);
    for(size_t q=0;q<4;++q){put(6,1,q*8,1);put(7,1,q*8+4,1);}
    put(5,2,0,1);put(5,10,0,-1);put(5,34,0,1);put(5,42,0,1);
    set16(f.weights,5*q8_wo_row_bytes+2*34,0x7000);set16(f.weights,5*q8_wo_row_bytes+10*34,0x7000); // 8192
    return f;
}
inline Fixture subnormal_fixture(){
    auto f=synthetic(max_rows);
    std::fill(f.weights.begin()+8*q8_wo_row_bytes,f.weights.end(),uint8_t(0));
    set16(f.input,3*36,0x3800);set16(f.input,4*36,0x4000);
    auto put=[&](size_t row,size_t block,uint16_t scale,int q){size_t at=row*q8_wo_row_bytes+block*34;set16(f.weights,at,scale);f.weights[at+2]=uint8_t(q);};
    for(unsigned m=1;m<=1023;++m){put(7+m,1,uint16_t(m),1);put(1030+m,1,uint16_t(0x8000|m),1);}
    put(2054,1,0x03ff,1);put(2055,1,0x0400,1);put(2056,3,0x0001,1);put(2057,4,0x83ff,1);
    put(2058,1,0x0001,-128);put(2059,1,0x0001,1);put(2059,9,0x8001,1);
    put(2060,1,0x0400,1);put(2060,9,0x0001,1);put(2061,1,0x8400,1);put(2061,9,0x8001,1);
    put(2062,4,0x03ff,1);put(2063,3,0x8001,1);
    return f;
}
struct Result {float value;std::array<float,128> partial;};
inline Result reference(const uint8_t* weights,size_t weights_size,const uint8_t* input,size_t input_size,size_t rows,size_t row){
    need(weights&&input&&row<rows,"WO: row");q8_wo_require_layout(rows,weights_size,input_size,rows*4);
    std::array<float,128> partial{};
    for(size_t tid=0;tid<128;++tid)for(size_t b=tid/4;b<blocks;b+=32){
        size_t q=2*(tid%4),wi=row*q8_wo_row_bytes+b*34,xi=b*36;int sum=0;
        for(size_t j=0;j<8;++j)sum+=s8(weights[wi+2+q*4+j])*s8(input[xi+4+q*4+j]);
        float scale=checked(half(get16(weights,wi))*input_half(get16(input,xi)));
        partial[tid]=checked(std::fma(scale,float(sum),partial[tid]));
    }
    auto mixed=partial;for(size_t lane=0;lane<32;++lane){float v=mixed[lane];for(size_t warp=1;warp<4;++warp)v=checked(v+partial[warp*32+lane]);mixed[lane]=v;}
    for(size_t offset=16;offset;offset>>=1){auto old=mixed;for(size_t lane=0;lane<32;++lane)mixed[lane]=checked(old[lane]+old[lane^offset]);}
    return {mixed[0],partial};
}
inline Result reference(const Fixture& f,size_t row){return reference(f.weights.data(),f.weights.size(),f.input.data(),f.input.size(),f.rows,row);}
// Existing expert helper visits block indices in a different order.
inline float expert_order(const Fixture& f,size_t row){
    std::array<float,32> p{};
    for(size_t lane=0;lane<32;++lane)for(size_t k=lane;k<blocks*4;k+=32){size_t b=k/4,q=2*(k%4),wi=row*q8_wo_row_bytes+b*34,xi=b*36;int dot=0;
        for(size_t j=0;j<8;++j)dot+=s8(f.weights[wi+2+q*4+j])*s8(f.input[xi+4+q*4+j]);
        p[lane]=checked(std::fma(checked(half(get16(f.weights,wi))*input_half(get16(f.input,xi))),float(dot),p[lane]));
    }
    for(size_t o=16;o;o>>=1){auto old=p;for(size_t i=0;i<32;++i)p[i]=checked(old[i]+old[i^o]);}return p[0];
}
inline void literal_controls(const Fixture& f){
    const float expected[]={0.f,1.f,32510.f,1.5f,0.f,0.f,16.f,20.f};
    for(size_t i=0;i<8;++i)need(bits(reference(f,i).value)==bits(expected[i]),"WO: independent literal mismatch");
    const auto integer=reference(f,2).partial;need(bits(integer[0])==bits(32510.f),"WO: signed DP4A integer controls");
    const auto halfscale=reference(f,3).partial;need(bits(halfscale[4])==bits(1.5f),"WO: half scale control");
    const auto first=reference(f,6).partial,second=reference(f,7).partial;
    const float first_expected[]={1.f,3.f,5.f,7.f},second_expected[]={2.f,4.f,6.f,8.f};
    for(size_t q=0;q<4;++q){need(bits(first[4+q])==bits(first_expected[q]),"WO: first DP4A fragment");need(bits(second[4+q])==bits(second_expected[q]),"WO: second DP4A fragment");}
    const auto p=reference(f,5).partial;need(bits(p[8])==bits(33554432.f)&&bits(p[40])==bits(-33554432.f),"WO: native partial ordering");
    need(bits(expert_order(f,5))==bits(2.f),"WO: expert/native partition distinction");
}
inline bool is_subnormal_fixture(const Fixture& f){return f.rows==max_rows && get16(f.weights,8*q8_wo_row_bytes+34)==1 && get16(f.weights,1031*q8_wo_row_bytes+34)==0x8001;}
inline float subnormal_expected(size_t row){
    if(row>=8 && row<=1030)return std::ldexp(float(row-7),-24);
    if(row>=1031 && row<=2053)return -std::ldexp(float(row-1030),-24);
    switch(row){case 2054:return std::ldexp(1023.f,-24);case 2055:return std::ldexp(1024.f,-24);
        case 2056:return std::ldexp(1.f,-25);case 2057:return -std::ldexp(1023.f,-23);
        case 2058:return -std::ldexp(128.f,-24);case 2059:return 0.f;
        case 2060:return std::ldexp(1025.f,-24);case 2061:return -std::ldexp(1025.f,-24);
        case 2062:return std::ldexp(1023.f,-23);case 2063:return -std::ldexp(1.f,-25);default:return 0.f;}
}
inline void subnormal_output_literals(const Fixture& f,const std::vector<float>& out){
    if(!is_subnormal_fixture(f))return;need(out.size()==max_rows,"WO: subnormal output rows");
    for(size_t row=8;row<max_rows;++row)if(bits(out[row])!=bits(subnormal_expected(row)))throw std::runtime_error("WO: subnormal literal differs at row "+std::to_string(row));
}
inline Bytes load(const std::filesystem::path& path,size_t bytes){std::ifstream i(path,std::ios::binary);need(bool(i),"WO: input missing");Bytes b(bytes);i.read((char*)b.data(),bytes);need(size_t(i.gcount())==bytes&&i.peek()==EOF,"WO: input size");return b;}
inline void save(const std::filesystem::path& path,const Bytes& b){need(!std::filesystem::exists(path),"WO: output exists");std::ofstream o(path,std::ios::binary);o.write((const char*)b.data(),b.size());need(bool(o),"WO: output write");}
inline void validate(const Fixture& f){q8_wo_require_layout(f.rows,f.weights.size(),f.input.size(),f.rows*4);for(size_t row=0;row<f.rows;++row)(void)reference(f,row);}
inline Fixture read_fixture(const std::string& w,const std::string& x,size_t rows){
    need(rows>=8 && rows<=max_rows,"WO: bounded synthetic row count");
    Fixture f{rows,load(w,rows*q8_wo_row_bytes),load(x,q8_wo_input_bytes)};validate(f);literal_controls(f);return f;
}
inline void contract(const Fixture& f,const std::string& name){
    need(!std::filesystem::exists(name),"WO: contract exists");std::ofstream o(name);o<<"{\"fields\":{\"out.f32le\":"<<f.rows*4<<"},\"qualified\":false}\n";need(bool(o),"WO: contract write");
}
inline void output_literals(const std::vector<float>& out){
    need(out.size()>=8,"WO: output rows");for(float v:out)(void)checked(v);const float expected[]={0.f,1.f,32510.f,1.5f,0.f,0.f,16.f,20.f};
    for(size_t i=0;i<8;++i)need(bits(out[i])==bits(expected[i]),"WO: output literal differs");
}
}
