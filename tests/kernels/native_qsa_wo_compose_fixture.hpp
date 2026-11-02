#pragma once
// Composition fixtures reuse the replay input framing and the frozen WO weights.
// Literal producer and WO controls remain independent standalone fixtures.
#ifndef STRATA_PROJECTION_Q8_REPLAY
#define STRATA_PROJECTION_Q8_REPLAY 1
#endif
#include "native_qsa_projection_q8_fixture.hpp"
#include "native_q8_wo_fixture.hpp"
#include "native_qsa_wo_domain.hpp"

namespace native_qsa_wo_compose {
namespace q = native_projection_q8;
namespace w = native_q8_wo_fixture;
using native_q8_fixture::Bytes;
using native_q8_fixture::need;
constexpr size_t rows = 2560;

inline Bytes expected_weights() {
    auto data=w::synthetic(rows).weights;
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
    auto direct=[&](size_t row,size_t block,uint16_t scale){auto first=data.begin()+row*strata::core::portable_detail::q8_wo_row_bytes;
        std::fill(first,first+strata::core::portable_detail::q8_wo_row_bytes,0);size_t at=block*34;
        first[at]=uint8_t(scale);first[at+1]=uint8_t(scale>>8);first[at+2]=2;first[at+6]=uint8_t(-3);};
    direct(1,8,0x3c00);direct(3,9,0xbc00);direct(4,11,0x8001);
    direct(6,96,0x3c00);direct(7,104,0xbc00);direct(8,112,0x0001);direct(9,120,0x8001);
#endif
    return data;
}
inline Bytes weights(const std::string& path) {
    auto actual=w::load(path,rows*strata::core::portable_detail::q8_wo_row_bytes);
    auto expected=expected_weights();
    need(actual==expected,"composition: frozen synthetic WO weights differ");
    return actual;
}
inline void packed_domain(const Bytes& weights,const Bytes& packed) {
    // Checks the actual Q8_1 packed scales and the WO intermediate domain.
    w::Fixture f{rows,weights,packed};
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
    strata::core::portable_detail::qsa_gdn_output_packed_domain(packed.data(),packed.size());
    w::validate(f,w::InputScaleDomain::gdn_output_finite);
#else
    w::validate(f);
#endif
}
inline void contract(const std::string& weights_path,const std::string& input_path,const std::string& path) {
    (void)weights(weights_path);(void)q::read(input_path);
    need(!std::filesystem::exists(path),"composition: contract exists");
    std::ofstream o(path);o<<"{\"fields\":{\"qx.bin\":6912,\"out.f32le\":10240";
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
    o<<",\"direct-qx.bin\":6912,\"direct-out.f32le\":10240";
#endif
    o<<"},\"qualified\":false}\n";
    need(bool(o),"composition: contract write");
}
inline void save(const std::filesystem::path& path,const void* data,size_t size) {
    need(!std::filesystem::exists(path),"composition: output exists");
    std::ofstream o(path,std::ios::binary);o.write(static_cast<const char*>(data),size);
    need(bool(o),"composition: output write");
}
inline std::vector<float> deterministic() {
    std::vector<float> values(q::elements);
    for(size_t i=0;i<values.size();++i) values[i]=float(int((i*37)%251)-125)*0.125f;
    q::validate(values);return values;
}
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
constexpr size_t scale_cases=7, scale_slots=181;
inline std::vector<size_t> scale_blocks(){std::vector<size_t> out;for(size_t b=8;b<192;++b)if(b!=10&&b!=34&&b!=42)out.push_back(b);return out;}
inline uint16_t expected_scale(size_t which,size_t slot){
    if(which<6){size_t m=which*scale_slots+slot+1;return m<=1023?uint16_t(m):uint16_t(0);}
    constexpr uint16_t v[]={0,0,1,1,2,2,1022,1022,1023,1023,1024,1024};return slot<12?v[slot]:0;
}
inline std::vector<float> scale_input(size_t which){
    need(which<scale_cases,"GDN output: scale case");auto x=q::synthetic(2);auto blocks=scale_blocks();need(blocks.size()==scale_slots,"GDN output: scale geometry");
    for(size_t b:{2u,10u,34u,42u}){std::fill_n(x.begin()+b*32,32,0.f);x[b*32]=b<32?127.f*4096.f:127.f;}
    for(size_t j=0;j<blocks.size();++j){size_t b=blocks[j];std::fill_n(x.begin()+b*32,32,0.f);float units=0.f;
        if(which<6){size_t m=which*scale_slots+j+1;if(m<=1023)units=float(m);}
        else if(j<12){constexpr float v[]={0.484375f,0.5f,0.515625f,1.484375f,1.5f,1.515625f,1022.484375f,1022.5f,1022.515625f,1023.484375f,1023.5f,1023.515625f};units=v[j];}
        if(units){float a=std::ldexp(units*127.f,-24);if(j&1)a=-a;x[b*32]=a;x[b*32+1]=-a*.25f;x[b*32+4]=a*.125f;}}
    q::validate(x);return x;
}
inline std::string case_name(size_t i){return "case-0"+std::to_string(i)+".bin";}
inline void scale_words(const Bytes& packed,size_t which){auto blocks=scale_blocks();for(size_t j=0;j<blocks.size();++j)need(w::get16(packed,blocks[j]*36)==expected_scale(which,j),"GDN output: stored scale literal differs");}
inline Bytes direct_packed(){Bytes out(q::bytes);for(auto entry:std::array<std::pair<size_t,uint16_t>,7>{{{8,1},{9,1023},{11,1024},{96,1},{104,1023},{112,1},{120,1023}}}){auto at=entry.first*36;out[at]=uint8_t(entry.second);out[at+1]=uint8_t(entry.second>>8);out[at+4]=7;out[at+8]=uint8_t(-3);}
    for(size_t b:{2u,10u,34u,42u}){w::set16(out,b*36,b<32?0x6c00:0x3c00);out[b*36+4]=127;}return out;}
inline void direct_controls(const Bytes& out){auto at=[&](size_t row){float v;std::memcpy(&v,out.data()+row*4,4);return v;};
    need(at(1)==std::ldexp(23.f,-24)&&at(3)==-std::ldexp(23.f*1023.f,-24)&&at(4)==-std::ldexp(23.f,-38)&&at(5)==0.f&&
         at(6)==std::ldexp(23.f,-24)&&at(7)==-std::ldexp(23.f*1023.f,-24)&&at(8)==std::ldexp(23.f,-48)&&at(9)==-std::ldexp(23.f*1023.f,-48),"GDN output: direct packed and cancellation literals");}
inline void fixture_scale_words(const std::string& path,const std::vector<float>& input,const Bytes& packed){
    auto name=std::filesystem::path(path).filename().string();if(name.rfind("case-",0)!=0)return;
    for(size_t i=0;i<scale_cases;++i)if(name==case_name(i)){auto expected=scale_input(i);need(std::memcmp(input.data(),expected.data(),input.size()*4)==0,"GDN output: fixture changed");scale_words(packed,i);return;}
    throw std::runtime_error("GDN output: unknown scale fixture");
}
#endif
}
