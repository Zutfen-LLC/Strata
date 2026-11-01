#include "native_gdn_log1p_fixture.hpp"
#include "native_llvm_log1p_adapter.hpp"
#include "strata_llvm_log1p_oracle.hpp"
using namespace native_gdn_log1p;
static std::vector<float>fixture(unsigned mode){if(mode<3)return native_gdn_log1p::synthetic(mode);native_hc_need(mode==3,"LLVM fixture mode");std::vector<float>x(256);uint32_t seed=0x1941ac53;
 const uint32_t pivots[]={0x35400003,0x3710001b,0x3efd81ad,0x3c800000,0x4de75840,0x800000,0x4e000000,0x3f800000};
 for(size_t i=0;i<256;++i){seed=1664525u*seed+1013904223u;uint32_t b=0x800000+seed%(0x4d800001);if(i<128){int d=int(i%16)-8;b=uint32_t(int64_t(pivots[i/16])+d);if(b<0x800000)b=0x800000;if(b>0x4e000000)b=0x4e000000;}x[i]=value(b);}validate(x);return x;}
int main(int argc,char**argv){try{
 native_hc_need(std::fesetround(FE_TONEAREST)==0,"LLVM oracle RNE");
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],fixture(std::stoul(argv[2])));return 0;}
 if(argc==3&&std::string(argv[1])=="--contract"){std::ofstream o(argv[2]);o<<"{\"fields\":{\"result.f32le\":1024,\"wide.u32le\":2048,\"path.u32le\":1024},\"count\":256,\"qualified\":false}\n";native_hc_need(bool(o),"LLVM output contract");return 0;}
 if(argc==4&&std::string(argv[1])=="--oracle"){
  auto x=read(argv[2]);std::filesystem::path dir=argv[3];native_hc_need(std::filesystem::create_directory(dir),"LLVM fresh CPU oracle output");std::filesystem::copy_file(argv[2],dir/"fixture.bin");std::ofstream f(dir/"result.f32le",std::ios::binary),w(dir/"wide.u32le",std::ios::binary),p(dir/"path.u32le",std::ios::binary);
  for(float v:x){double wide;uint32_t path;float got=strata_llvm_adapter::strata_llvm_log1p(v,wide,path);native_hc_need(bits(got)==bits(strata_llvm_upstream::math::log1pf(v)),"LLVM CPU oracle differs");uint64_t b;std::memcpy(&b,&wide,8);word(f,bits(got));word(w,uint32_t(b));word(w,uint32_t(b>>32));word(p,path);}return 0;
 }
 if(argc==4&&std::string(argv[1])=="--check"){
  auto x=read(argv[2]);std::filesystem::path dir=argv[3];std::ifstream f(dir/"result.f32le",std::ios::binary),w(dir/"wide.u32le",std::ios::binary),p(dir/"path.u32le",std::ios::binary);native_hc_need(bool(f)&&bool(w)&&bool(p),"LLVM required outputs");
  for(float v:x){float got=value(word(f));uint64_t wide=word(w);wide|=uint64_t(word(w))<<32;uint32_t path=word(p);double expected;uint32_t ep;float own=strata_llvm_adapter::strata_llvm_log1p(v,expected,ep);uint64_t eb;std::memcpy(&eb,&expected,8);native_hc_need(bits(got)==bits(strata_llvm_upstream::math::log1pf(v)),"LLVM original algorithm differs");native_hc_need(bits(got)==bits(own)&&wide==eb&&path==ep,"LLVM FP64 source-order/intermediate/path differs");}
  native_hc_need(f.peek()==EOF&&w.peek()==EOF&&p.peek()==EOF,"LLVM output shape");return 0;
 }
 auto check=[](float x){native_gdn_log1p_input(x);double wide;uint32_t path;float a=strata_llvm_adapter::strata_llvm_log1p(x,wide,path),b=strata_llvm_upstream::math::log1pf(x);native_hc_need(bits(a)==bits(b),"LLVM original non-FMA source oracle differs");};
 for(unsigned m=0;m<4;++m)for(float v:fixture(m))check(v);uint32_t seed=0x14c8fb91;for(size_t i=0;i<100000;++i){seed=1664525u*seed+1013904223u;check(value(0x800000+seed%0x4d800001));}
 for(uint32_t x:{0u,0x80000000u,0x35400003u,0x3710001bu,0x3efd81adu,0x4de75840u})check(value(x));
 puts("LLVM adapter matches original accurate non-FMA/RNE algorithm: 100000 deterministic inputs and boundary/hard cases; CPU only");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
