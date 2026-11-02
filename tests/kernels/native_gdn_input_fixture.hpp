#pragma once
#define STRATA_PROJECTION_Q8_QKV 1
#define STRATA_PROJECTION_Q8_REPLAY 1
#include "native_qsa_projection_q8_fixture.hpp"
#include "native_gdn_input_layout.hpp"
#include "native_qsa_wo_domain.hpp"
#include <array>
#include <filesystem>
namespace native_gdn_input_fixture {
namespace q=native_projection_q8;
using native_q8_fixture::Bytes;
using native_q8_fixture::need;
using namespace strata::core::portable_detail;
constexpr std::array<size_t,2> rows{{10240,6144}};
constexpr std::array<const char*,2> names{{"qkv","z"}};
inline std::vector<float> input(unsigned mode){
 if(mode==0){auto x=q::synthetic(2);for(size_t b:{10u,34u,42u}){std::fill_n(x.begin()+b*32,32,0.f);x[b*32]=127.f;}q::validate(x);return x;}
 need(mode==1,"GDN input: invalid fixture mode");std::vector<float>x(q::elements);
 for(size_t b=0;b<q::blocks;++b){x[b*32]=127.f;x[b*32+1]=-127.f;for(size_t j=2;j<32;++j)x[b*32+j]=j%2? -1.f:1.f;}
 q::validate(x);return x;
}
inline std::pair<std::vector<float>,unsigned> read_input(const std::string& path){auto x=q::read(path);for(unsigned i=0;i<2;++i){auto expected=input(i);if(std::memcmp(x.data(),expected.data(),x.size()*4)==0)return {x,i};}throw std::runtime_error("GDN input: fixture differs");}
inline void pinned(const Bytes&v,unsigned mode){
 if(mode==0){q::pinned(v);return;}need(v.size()==q::bytes,"GDN input: packed size");
 for(size_t b=0;b<q::blocks;++b){std::array<uint8_t,36>expected{};expected[1]=0x3c;expected[4]=127;expected[5]=uint8_t(-127);for(size_t j=2;j<32;++j)expected[4+j]=uint8_t(j%2?-1:1);need(std::equal(expected.begin(),expected.end(),v.begin()+b*36),"GDN input: dense literal packed block");}
}
inline Bytes weights(size_t role){
 need(role<rows.size(),"GDN input: invalid role");Bytes w(rows[role]*gdn_input_row_bytes);
 for(size_t r=0;r<rows[role];++r)for(size_t b=0;b<gdn_input_blocks;++b){size_t at=r*gdn_input_row_bytes+b*34;w[at]=0;w[at+1]=0x3c;for(size_t j=0;j<32;++j){int v=int((r*7+b*11+j*5+role*3)%15)-7;w[at+2+j]=uint8_t(v?v:8);}}
 auto literal=[&](size_t row,int value){auto first=w.begin()+row*gdn_input_row_bytes;std::fill(first,first+gdn_input_row_bytes,0);first[2*34+1]=0x3c;first[2*34+2]=uint8_t(value);};
 literal(0,role?2:1);literal(1,-1);
 {auto first=w.begin()+2*gdn_input_row_bytes;std::fill(first,first+gdn_input_row_bytes,0);first[2*34+1]=0xbc;first[2*34+2]=1;}
 {auto first=w.begin()+3*gdn_input_row_bytes;std::fill(first,first+gdn_input_row_bytes,0);first[2*34+1]=0x3c;first[2*34+2+4]=1;}
 if(role==0){auto first=w.begin()+5*gdn_input_row_bytes;std::fill(first,first+gdn_input_row_bytes,0);for(size_t b:{2u,10u,34u,42u}){size_t at=b*34;uint16_t h=b<32?0x7800:0x0001;first[at]=uint8_t(h);first[at+1]=uint8_t(h>>8);first[at+2]=uint8_t(b==10?-1:1);}}
 return w;
}
inline Bytes read_weights(const std::filesystem::path&p,size_t role){auto expected=weights(role);std::ifstream i(p,std::ios::binary);need(bool(i),"GDN input: missing weights");Bytes actual(expected.size());i.read(reinterpret_cast<char*>(actual.data()),actual.size());need(size_t(i.gcount())==actual.size()&&i.peek()==EOF&&actual==expected,"GDN input: weights differ");return actual;}
inline void output_controls(const std::array<Bytes,2>&out,unsigned mode){auto f=[&](size_t role,size_t row){need(out[role].size()==rows[role]*4,"GDN input: output size");float v;std::memcpy(&v,out[role].data()+row*4,4);return v;};need(f(0,0)==127.f&&f(0,1)==-127.f&&f(0,5)==0.f&&f(1,0)==254.f&&f(1,1)==-127.f&&f(0,2)==-127.f&&f(1,2)==-127.f&&f(0,3)==(mode?1.f:-2.f)&&f(1,3)==(mode?1.f:-2.f),"GDN input: independent ordered literals");for(size_t i=0;i<2;++i)for(size_t r=0;r<rows[i];++r)need(qsa_wo_normal_or_zero(f(i,r)),"GDN input: unsupported output");}
inline void packed_domain(const Bytes&data){need(data.size()==gdn_input_packed_bytes,"GDN input: packed size");qsa_wo_require_scales(qsa_wo_scale_classes(data.data(),gdn_input_blocks,36),"GDN producer Q8_1");}
inline void save(const std::filesystem::path&p,const void*data,size_t size){need(!std::filesystem::exists(p),"GDN input: output exists");std::ofstream o(p,std::ios::binary);o.write(static_cast<const char*>(data),size);need(bool(o),"GDN input: output write");}
inline void contract(const std::filesystem::path&dir,const std::string&in,const std::string&out){(void)read_input(in);for(size_t i=0;i<2;++i)(void)read_weights(dir/(std::string(names[i])+".bin"),i);need(!std::filesystem::exists(out),"GDN input: contract exists");std::ofstream o(out);o<<"{\"fields\":{\"qx.bin\":2880,\"qkv.f32le\":40960,\"z.f32le\":24576},\"qualified\":false}\n";need(bool(o),"GDN input: contract write");}
}
