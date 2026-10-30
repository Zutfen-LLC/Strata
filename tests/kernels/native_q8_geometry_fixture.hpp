#pragma once
#include "native_q8_consumer_fixture.hpp"
namespace native_q8_fixture::geometry {
constexpr size_t H=2560,F=640,ENTRIES=3,TOKENS=3,GROUPS=3,GRID_GROUPS=2;
constexpr size_t WEIGHT_BYTES=5222400,DOWN_OFFSET=3481600;
constexpr size_t INPUT_FLOATS=TOKENS*H,FLOATS=ENTRIES*F,OUTPUT_FLOATS=ENTRIES*H;
constexpr size_t INPUT_BYTES=TOKENS*(H/32)*36,HQ_BYTES=ENTRIES*(F/32)*36;
inline const std::vector<int32_t> STARTS{0,2,2,3},TOKENS_BY_ENTRY{2,0,1},DESTINATIONS{1,2,0};
inline std::vector<float> input() {
    std::vector<float> x(INPUT_FLOATS);
    for(size_t token=0;token<TOKENS;++token)for(size_t i=0;i<H;++i)
        x[token*H+i]=i/32==token?0.f:float(int((i*13+token*29)%127)-48)/64.f;
    return x;
}
inline void write_fixture(const std::string& path) {
    std::ofstream out(path,std::ios::binary);need(bool(out),"geometry: fixture create failed");
    word(out,0x3146384e);word(out,H);word(out,F);word(out,ENTRIES);
    for(float value:input())word(out,bits(value));need(bool(out),"geometry: fixture write failed");
}
inline std::vector<float> read_fixture(const std::string& path) {
    std::ifstream in(path,std::ios::binary);need(bool(in),"geometry: fixture read failed");
    need(word(in)==0x3146384e && word(in)==H && word(in)==F && word(in)==ENTRIES,"geometry: fixture header mismatch");
    auto expected=input();
    for(float value:expected)need(word(in)==bits(value),"geometry: fixture differs from finite synthetic recipe");
    need(in.peek()==EOF,"geometry: trailing fixture bytes");return expected;
}
inline Bytes weights() {return consumer::weights(H,F);}
inline Bytes read_weights(const std::string& path) {
    std::ifstream in(path,std::ios::binary);need(bool(in),"geometry: cannot read weights");
    auto expected=weights();Bytes got(WEIGHT_BYTES);in.read(reinterpret_cast<char*>(got.data()),got.size());
    need(bool(in) && in.peek()==EOF && got==expected,"geometry: weight recipe/size mismatch");return got;
}
inline Bytes bytes(const std::vector<float>& values) {
    Bytes out(values.size()*4);std::memcpy(out.data(),values.data(),out.size());return out;
}
// Both roles have identical ordered sections; distinct magic prevents role reuse.
inline void write_result(const std::string& path,bool observation,bool sentinels,const std::array<Bytes,6>& sections) {
    const size_t sizes[]={INPUT_BYTES,FLOATS*4,FLOATS*4,FLOATS*4,HQ_BYTES,OUTPUT_FLOATS*4};
    std::ofstream out(path,std::ios::binary);need(bool(out),"geometry: result create failed");
    word(out,observation?0x324d384e:0x314d384e);word(out,sentinels?1:0);
    for(size_t i=0;i<sections.size();++i){need(sections[i].size()==sizes[i],"geometry: result section size");
        word(out,uint32_t(sizes[i]/4));out.write(reinterpret_cast<const char*>(sections[i].data()),sizes[i]);}
    need(bool(out),"geometry: result write failed");
}
}
