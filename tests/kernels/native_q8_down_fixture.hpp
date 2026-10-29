#pragma once
#include "native_q8_gu_fixture.hpp"
namespace native_q8_fixture::down {
inline Bytes read_hq(const std::string& path) {
    std::ifstream in(path,std::ios::binary);need(bool(in),"down: cannot read actual native hq");
    Bytes bytes(consumer::HQ_BYTES);in.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
    need(bool(in) && in.peek()==EOF,"down: hq must be exactly 72 bytes");
    for(size_t p=0;p<bytes.size();p+=36){const uint16_t d=uint16_t(bytes[p])|(uint16_t(bytes[p+1])<<8);
        need((d&0x7c00)!=0x7c00,"down: finite native hq scale required");}
    return bytes;
}
inline void write_result(const std::string& path,const std::array<std::vector<float>,3>& values,bool sentinels) {
    std::ofstream out(path,std::ios::binary);need(bool(out),"down: cannot write raw result");
    word(out,0x3144384e);word(out,sentinels?1:0);
    for(const auto& v:values){need(v.size()==1024,"down: raw scatter section shape");word(out,uint32_t(v.size()));
        for(float f:v)word(out,bits(f));}need(bool(out),"down: raw result write failed");
}
}
