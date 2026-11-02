#pragma once
#include "native_qsa_indexer_bf16.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace native_qsa_indexer_bf16_fixture {
using namespace strata::core::portable_detail;
struct Fixture{std::vector<float>x;std::array<std::vector<uint16_t>,2>w;};
inline constexpr std::array<size_t,2>rows{{128,512}};
inline void need(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
inline Fixture synthetic(){Fixture f;f.x.assign(2560,0.f);f.x[0]=16777216.f;f.x[512]=1.f;f.x[513]=-16777216.f;f.x[2558]=3.f;f.x[2559]=5.f;
 for(size_t i=0;i<2;++i){f.w[i].assign(rows[i]*2560,0);for(size_t row=0;row<rows[i];++row){auto*b=f.w[i].data()+row*2560;b[0]=0x3f80;b[512]=0x3f80;b[513]=0x3f80;b[2558]=row%2?0x3f80:0x4000;b[2559]=0xbf80;}}
 return f;}
inline std::vector<char> read_file(const std::filesystem::path&p,size_t size){need(!std::filesystem::is_symlink(p)&&std::filesystem::is_regular_file(p)&&std::filesystem::file_size(p)==size,"indexer BF16 fixture span");std::vector<char>b(size);std::ifstream f(p,std::ios::binary);f.read(b.data(),size);need(bool(f),"indexer BF16 fixture read");return b;}
inline Fixture read(const std::filesystem::path&dir){Fixture f;auto x=read_file(dir/"input.f32le",2560*4);f.x.resize(2560);std::memcpy(f.x.data(),x.data(),x.size());
 const std::array<const char*,2>names{{"key.bf16le","query.bf16le"}};for(size_t i=0;i<2;++i){auto data=read_file(dir/names[i],rows[i]*2560*2);f.w[i].resize(rows[i]*2560);std::memcpy(f.w[i].data(),data.data(),data.size());(void)qsa_indexer_bf16_control(f.x.data(),f.w[i].data(),rows[i]);}
 const auto expected=synthetic();need(std::memcmp(f.x.data(),expected.x.data(),f.x.size()*4)==0 && f.w==expected.w,"indexer BF16 frozen fixture differs");return f;}
inline void save(const std::filesystem::path&p,const void*data,size_t size){need(!std::filesystem::exists(p),"indexer BF16 refuses existing file");std::ofstream out(p,std::ios::binary);out.write(static_cast<const char*>(data),size);need(bool(out),"indexer BF16 fixture write");}
inline void write(const std::filesystem::path&dir,const Fixture&f){need(std::filesystem::create_directory(dir),"indexer BF16 refuses existing fixture");save(dir/"input.f32le",f.x.data(),f.x.size()*4);save(dir/"key.bf16le",f.w[0].data(),f.w[0].size()*2);save(dir/"query.bf16le",f.w[1].data(),f.w[1].size()*2);}
inline void contract(const std::filesystem::path&dir,const std::filesystem::path&out){(void)read(dir);need(!std::filesystem::exists(out),"indexer BF16 refuses existing contract");std::ofstream f(out);f<<"{\"fields\":{\"key.f32le\":512,\"query.f32le\":2048},\"qualified\":false}\n";need(bool(f),"indexer BF16 contract write");}
inline void arguments(int argc,char**argv){need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","usage: --device INDEX --expect-device NAME --fixture DIR --output NEW_DIR");std::string i=argv[2];need(!i.empty()&&i.find_first_not_of("0123456789")==std::string::npos&&*argv[4],"indexer BF16 invalid device");}
inline void literals(const std::array<std::vector<float>,2>&v){need(v[0].size()==128&&v[1].size()==512,"indexer BF16 output span");need(v[0][0]==1.f&&v[0][127]==-2.f&&v[1][0]==1.f&&v[1][511]==-2.f,"indexer BF16 independent sentinels differ");}
}
