#pragma once
#define STRATA_PROJECTION_Q8_QKV 1
#include "native_qsa_projection_q8_fixture.hpp"
#include "native_qsa_qkv_layout.hpp"
#include "native_qsa_wo_domain.hpp"
#include <array>
#include <filesystem>
namespace native_qsa_qkv_fixture {
namespace q = native_projection_q8;
using native_q8_fixture::Bytes;
using native_q8_fixture::need;
using namespace strata::core::portable_detail;
constexpr std::array<size_t,3> rows{{12288,512,512}};
constexpr std::array<const char*,3> names{{"q","k","v"}};
inline std::vector<float> input() {
    auto x=q::synthetic(2);
    for(size_t b : {10u,34u,42u}) {std::fill_n(x.begin()+b*32,32,0.f);x[b*32]=127.f;}
    q::validate(x);return x;
}
#ifdef STRATA_QKV_Q81_SCALE_SUITE
constexpr size_t scale_cases=16, scale_slots=69;
inline std::vector<size_t> scale_blocks(){std::vector<size_t> b;for(size_t i=8;i<80;++i)if(i!=10&&i!=34&&i!=42)b.push_back(i);return b;}
inline uint16_t expected_scale(size_t which,size_t slot){
    if(which<15){size_t m=which*scale_slots+slot+1;return m<=1023?uint16_t(m):uint16_t(0);}
    constexpr uint16_t v[]={0,0,1,1,2,2,1022,1022,1023,1023,1024,1024};
    return slot<12?v[slot]:0;
}
inline std::vector<float> scale_input(size_t which){
    need(which<scale_cases,"QSA QKV: scale fixture index");auto x=input();auto blocks=scale_blocks();need(blocks.size()==scale_slots,"QSA QKV: scale fixture geometry");
    for(size_t j=0;j<blocks.size();++j){size_t b=blocks[j];std::fill_n(x.begin()+b*32,32,0.f);float units=0;
        if(which<15){size_t m=which*scale_slots+j+1;if(m<=1023)units=float(m);}
        else if(j<12){constexpr float v[]={0.484375f,0.5f,0.515625f,1.484375f,1.5f,1.515625f,1022.484375f,1022.5f,1022.515625f,1023.484375f,1023.5f,1023.515625f};units=v[j];}
        if(units){float a=std::ldexp(units*127.f,-24);if(j&1)a=-a;x[b*32]=a;x[b*32+1]=-a*0.25f;x[b*32+4]=a*0.125f;}
    }q::validate(x);return x;
}
inline std::string case_name(size_t which){return std::string("case-")+(which<10?"0":"")+std::to_string(which)+".bin";}
inline size_t case_index(const std::string& path){auto name=std::filesystem::path(path).filename().string();for(size_t i=0;i<scale_cases;++i)if(name==case_name(i))return i;throw std::runtime_error("QSA QKV: unknown scale fixture");}
inline std::vector<float> read_input(const std::string& path){size_t i=case_index(path);auto x=q::read(path),expected=scale_input(i);need(std::memcmp(x.data(),expected.data(),x.size()*sizeof(float))==0,"QSA QKV: scale fixture changed");return x;}
inline void check_scale_words(const Bytes& packed,size_t which){auto blocks=scale_blocks();for(size_t j=0;j<blocks.size();++j){size_t at=blocks[j]*36;uint16_t h=uint16_t(packed[at])|(uint16_t(packed[at+1])<<8);need(h==expected_scale(which,j),"QSA QKV: independent stored scale literal differs");}}
inline Bytes direct_packed(){Bytes out(q::bytes);for(auto entry:std::array<std::pair<size_t,uint16_t>,3>{{{8,1},{9,1023},{11,1024}}}){auto at=entry.first*36;out[at]=uint8_t(entry.second);out[at+1]=uint8_t(entry.second>>8);out[at+4]=7;out[at+8]=uint8_t(-3);}return out;}
#else
inline std::vector<float> read_input(const std::string& path){return q::read(path);}
#endif
inline Bytes weights(size_t role) {
    need(role<rows.size(),"QSA QKV: invalid role");
    Bytes w(rows[role]*qsa_qkv_row_bytes);
    for(size_t r=0;r<rows[role];++r)for(size_t b=0;b<qsa_qkv_blocks;++b){
        const size_t at=r*qsa_qkv_row_bytes+b*34;w[at]=0;w[at+1]=0x3c;
        for(size_t j=0;j<32;++j){int v=int((r*7+b*11+j*5+role*3)%15)-7;w[at+2+j]=uint8_t(v?v:8);}
    }
    auto literal=[&](size_t row,int value){
        auto first=w.begin()+row*qsa_qkv_row_bytes;std::fill(first,first+qsa_qkv_row_bytes,0);
        first[2*34]=0;first[2*34+1]=0x3c;first[2*34+2]=uint8_t(value);
    };
    if(role==0){
        for(size_t h=0;h<24;++h){literal(qsa_qkv_query_row(h,0),1);literal(qsa_qkv_gate_row(h,0),-1);}
        // Native threads first accumulate blocks 2/34 and 10/42 separately.
        // An expert-style order adds 2,10,34,42 in one lane and retains the
        // two tiny terms; the four-warp native result is exactly zero.
        auto first=w.begin()+5*qsa_qkv_row_bytes;std::fill(first,first+qsa_qkv_row_bytes,0);
        for(size_t b : {2u,10u,34u,42u}){size_t at=b*34;uint16_t h=b<32?0x7800:0x0001;
            first[at]=uint8_t(h);first[at+1]=uint8_t(h>>8);first[at+2]=uint8_t(b==10?-1:1);}
    }
    else literal(0,role==1?2:-2);
#ifdef STRATA_QKV_Q81_SCALE_SUITE
    auto direct=[&](size_t row,size_t block,uint16_t scale){auto first=w.begin()+row*qsa_qkv_row_bytes;std::fill(first,first+qsa_qkv_row_bytes,0);size_t at=block*34;first[at]=uint8_t(scale);first[at+1]=uint8_t(scale>>8);first[at+2]=2;first[at+6]=uint8_t(-3);};
    direct(1,8,0x3c00);direct(3,9,0xbc00);direct(4,11,0x8001);
#endif
    return w;
}
#ifdef STRATA_QKV_Q81_SCALE_SUITE
inline void direct_controls(const std::array<Bytes,3>& output){for(size_t i=0;i<3;++i){auto at=[&](size_t row){float x;std::memcpy(&x,output[i].data()+row*4,4);return x;};
    need(at(1)==std::ldexp(23.f,-24),"QSA QKV: direct smallest subnormal literal");
    need(at(3)==-std::ldexp(23.f*1023.f,-24),"QSA QKV: direct largest subnormal signed literal");
    need(at(4)==-std::ldexp(23.f,-38),"QSA QKV: direct normal/subnormal product literal");}}
#endif
inline Bytes read_weights(const std::filesystem::path& path,size_t role) {
    auto expected=weights(role);std::ifstream i(path,std::ios::binary);need(bool(i),"QSA QKV: missing weights");
    Bytes actual(expected.size());i.read(reinterpret_cast<char*>(actual.data()),actual.size());
    need(size_t(i.gcount())==actual.size()&&i.peek()==EOF&&actual==expected,"QSA QKV: weights differ");return actual;
}
inline void output_controls(const std::array<Bytes,3>& output){
    auto f=[&](size_t role,size_t row){need(output[role].size()==rows[role]*4,"QSA QKV: output bytes");float v;std::memcpy(&v,output[role].data()+row*4,4);return v;};
    for(size_t h=0;h<24;++h){need(f(0,qsa_qkv_query_row(h,0))==127.f,"QSA QKV: query row literal");need(f(0,qsa_qkv_gate_row(h,0))==-127.f,"QSA QKV: gate row literal");}
    need(f(0,5)==0.f,"QSA QKV: native four-warp ordering literal");
    need(f(1,0)==254.f&&f(2,0)==-254.f,"QSA QKV: key/value row literal");
    for(size_t i=0;i<3;++i)for(size_t r=0;r<rows[i];++r)need(qsa_wo_normal_or_zero(f(i,r)),"QSA QKV: nonfinite or subnormal output");
}
inline void packed_domain(const Bytes& data){qsa_qkv_packed_domain(data.data(),data.size());}
inline void save(const std::filesystem::path& p,const void* data,size_t size){
    need(!std::filesystem::exists(p),"QSA QKV: output exists");std::ofstream o(p,std::ios::binary);o.write(static_cast<const char*>(data),size);need(bool(o),"QSA QKV: output write");
}
inline void contract(const std::filesystem::path& dir,const std::string& input,const std::string& output){
    (void)read_input(input);for(size_t role=0;role<3;++role)(void)read_weights(dir/(std::string(names[role])+".bin"),role);
    need(!std::filesystem::exists(output),"QSA QKV: contract exists");std::ofstream o(output);
    o<<"{\"fields\":{\"qx.bin\":2880,\"q.f32le\":49152,\"k.f32le\":2048,\"v.f32le\":2048";
#ifdef STRATA_QKV_Q81_SCALE_SUITE
    o<<",\"direct-qx.bin\":2880,\"direct-q.f32le\":49152,\"direct-k.f32le\":2048,\"direct-v.f32le\":2048";
#endif
    o<<"},\"qualified\":false}\n";need(bool(o),"QSA QKV: contract write");
}
}
