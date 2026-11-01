#pragma once
#include "native_hc_fixture.hpp"
#include "../../src/core/native_hc_pending.hpp"
namespace native_hc_write_fixture {
using namespace native_hc_fixture;
struct Fixture {std::vector<float> residual,block,inject;};
constexpr size_t bytes=8+40960+10240+16;
inline void validate(const Fixture& f){(void)native_hc_write_control(f.residual,f.block,f.inject);}
inline Fixture synthetic(const std::string& kind){
 native_hc_need(kind=="zero"||kind=="cancel"||kind=="sparse","HC write: synthetic case");
 Fixture f;f.residual.resize(10240);f.block.resize(2560);f.inject={-8.f,-1.f,0.f,4.f};
 for(size_t d=0;d<2560;++d)f.block[d]=float(int(d%7)-3)/128.f;
 for(size_t i=0;i<10240;++i)f.residual[i]=float(int(i%17)-8)/32.f;
 if(kind=="zero"){std::fill(f.residual.begin(),f.residual.end(),0.f);std::fill(f.block.begin(),f.block.end(),0.f);std::fill(f.inject.begin(),f.inject.end(),0.f);}
 if(kind=="cancel"){std::fill(f.inject.begin(),f.inject.end(),-0.f);for(size_t i=0;i<10240;++i)f.residual[i]=-f.block[i%2560];}
 validate(f);return f;
}
inline void write(const std::string& path,const Fixture& f){validate(f);native_hc_need(!std::filesystem::exists(path),"HC write: fresh fixture");std::ofstream o(path,std::ios::binary);o.write("HCWRITE1",8);for(auto* a:{&f.residual,&f.block,&f.inject})for(float v:*a)word(o,bits(v));native_hc_need(bool(o),"HC write: fixture write");}
inline Fixture read(const std::string& path){native_hc_need(std::filesystem::file_size(path)==bytes,"HC write: fixture size");std::ifstream i(path,std::ios::binary);char magic[8];i.read(magic,8);native_hc_need(std::memcmp(magic,"HCWRITE1",8)==0,"HC write: fixture version");Fixture f;f.residual.resize(10240);f.block.resize(2560);f.inject.resize(4);for(auto* a:{&f.residual,&f.block,&f.inject})for(float& v:*a)v=value(word(i));native_hc_need(bool(i)&&i.peek()==EOF,"HC write: fixture read");validate(f);return f;}
}
