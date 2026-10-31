#pragma once
#include "native_hc_fixture.hpp"
namespace native_hc_mix_fixture {
using namespace native_hc_fixture;
struct Mix {std::vector<float> x;std::vector<uint16_t> down,up;};
inline Mix read(const std::string& path){native_hc_shape(2560,4,1);native_hc_need(std::filesystem::file_size(path)==13148168,"HC mix: fixed fixture length");std::ifstream in(path,std::ios::binary);char magic[8];in.read(magic,8);native_hc_need(std::memcmp(magic,"HCMIX01\0",8)==0,"HC mix: version");Mix f;f.x.resize(10240);f.down.resize(3276800);f.up.resize(3276800);for(auto& v:f.x)v=value(word(in));for(auto* a:{&f.down,&f.up})native_hc_need(bool(in.read(reinterpret_cast<char*>(a->data()),a->size()*2)),"HC mix: weight read");for(float x:f.x)native_hc_normal(x);for(auto* a:{&f.down,&f.up})for(auto v:*a)native_hc_normal(value(uint32_t(v)<<16));return f;}
}
