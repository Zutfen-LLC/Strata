#pragma once
#include "native_q8_consumer_fixture.hpp"
namespace native_q8_fixture::gu {
struct Oracle {std::vector<uint32_t> gate,up,output;};
inline Oracle read_oracle(const std::string& path){
    std::ifstream in(path,std::ios::binary);need(bool(in),"GU: cannot read native raw oracle");
    need(word(in)==0x3143384e,"GU: wrong native raw oracle magic");
    Oracle out;size_t section=0;
    for(auto* v:{&out.gate,&out.up,&out.output}){const size_t n=section++==2?512:64;
        need(word(in)==n,"GU: wrong native raw oracle section size/order");v->resize(n);
        bool signal=false;
        for(auto& bits:*v){bits=word(in);need((bits&0x7f800000)!=0x7f800000,"GU: nonfinite oracle value");signal|=(bits&0x7fffffff)!=0;}
        need(signal,"GU: degenerate native oracle");
        need(std::equal(v->begin(),v->begin()+n/2,v->begin()+n/2),"GU: native paired oracle rows differ");}
    need(in.peek()==EOF,"GU: trailing native oracle bytes");return out;
}
inline Bytes read_weights(const std::string& path){
    std::ifstream in(path,std::ios::binary);need(bool(in),"GU: cannot read weight receipt");
    auto expected=consumer::weights();Bytes bytes(expected.size());
    in.read(reinterpret_cast<char*>(bytes.data()),bytes.size());need(bool(in) && in.peek()==EOF,"GU: weight receipt size");
    need(bytes==expected,"GU: weight receipt differs from native recipe");return bytes;
}
inline void write_result(const std::string& path,const std::vector<float>& gate,const std::vector<float>& up,bool sentinels){
    std::ofstream out(path,std::ios::binary);need(bool(out),"GU: cannot create raw result");
    word(out,0x3147384e);word(out,sentinels?1:0);
    for(const auto* v:{&gate,&up}){word(out,uint32_t(v->size()));for(float x:*v)word(out,bits(x));}
    need(bool(out),"GU: raw result write failed");
}
}
