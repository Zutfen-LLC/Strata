#pragma once
#include "native_hc_fixture.hpp"
#include "../../src/core/native_hc_pending.hpp"
namespace native_hc_pending_fixture {
using namespace native_hc_fixture;
struct Pending {Fixture norm;std::vector<float> bo,inj;std::vector<uint16_t> down,up;};
constexpr size_t BYTES=8+40960+40960+81920+10240+16+2*6553600;
inline void validate(const Pending& f){native_hc_need(f.norm.t==1&&f.bo.size()==2560&&f.inj.size()==4&&f.down.size()==3276800&&f.up.size()==3276800,"HC pending: fixed geometry");native_hc_fixture::validate(f.norm);auto pending=native_hc_pending_control(f.norm.x,f.bo,f.inj);(void)native_hc_oracle(pending.residual,f.norm.w,f.norm.wi,1);for(auto* w:{&f.down,&f.up})for(auto v:*w)native_hc_normal(value(uint32_t(v)<<16));}
inline Pending synthetic(const std::string& kind){native_hc_need(kind=="zero"||kind=="cancel"||kind=="sparse","HC pending: synthetic case");Pending f;f.norm=native_hc_fixture::synthetic(1);f.bo.resize(2560);f.inj={-8.f,-1.f,0.f,4.f};f.down.assign(3276800,0);f.up.assign(3276800,0);
 for(size_t d=0;d<2560;++d)f.bo[d]=float(int(d%7)-3)*0.0078125f;
 for(size_t i=0;i<10240;++i)f.norm.x[i]=float(int(i%17)-8)*0.03125f;
 if(kind=="zero"){std::fill(f.norm.x.begin(),f.norm.x.end(),0.f);std::fill(f.bo.begin(),f.bo.end(),0.f);std::fill(f.inj.begin(),f.inj.end(),0.f);}
 if(kind=="cancel"){std::fill(f.inj.begin(),f.inj.end(),0.f);for(size_t i=0;i<10240;++i)f.norm.x[i]=-f.bo[i%2560];}
 for(size_t row=0;row<320;++row)f.down[row*10240+(row*31)%10240]=native_hc_bf16_rne(float(int(row%7)-3)*0.03125f);
 for(size_t row=0;row<10240;++row)f.up[row*320+(row*7)%320]=native_hc_bf16_rne(float(int(row%5)-2)*0.03125f);
 validate(f);return f;
}
inline void write(const std::string& path,const Pending& f){validate(f);native_hc_need(!std::filesystem::exists(path),"HC pending: fresh fixture");std::ofstream out(path,std::ios::binary);out.write("HCPEND1\0",8);for(auto* v:{&f.norm.x,&f.norm.w})for(float x:*v)word(out,bits(x));out.write(reinterpret_cast<const char*>(f.norm.wi.data()),81920);for(auto* v:{&f.bo,&f.inj})for(float x:*v)word(out,bits(x));for(auto* v:{&f.down,&f.up})out.write(reinterpret_cast<const char*>(v->data()),6553600);native_hc_need(bool(out),"HC pending: write");}
inline Pending read(const std::string& path){native_hc_need(std::filesystem::file_size(path)==BYTES,"HC pending: bounded fixture length");std::ifstream in(path,std::ios::binary);char magic[8];in.read(magic,8);native_hc_need(std::memcmp(magic,"HCPEND1\0",8)==0,"HC pending: fixture version");Pending f;f.norm.t=1;f.norm.x.resize(10240);f.norm.w.resize(10240);f.norm.wi.resize(40960);f.bo.resize(2560);f.inj.resize(4);f.down.resize(3276800);f.up.resize(3276800);
 for(auto* v:{&f.norm.x,&f.norm.w})for(float& x:*v)x=value(word(in));in.read(reinterpret_cast<char*>(f.norm.wi.data()),81920);for(auto* v:{&f.bo,&f.inj})for(float& x:*v)x=value(word(in));for(auto* v:{&f.down,&f.up})in.read(reinterpret_cast<char*>(v->data()),6553600);native_hc_need(bool(in)&&in.peek()==EOF,"HC pending: fixture truncated/trailing");validate(f);return f;}
}
