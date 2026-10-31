#pragma once
#include "ggml.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace strata::core::portable_detail {
inline void native_hc_need(bool ok,const char* msg){if(!ok)throw std::invalid_argument(msg);}
inline void native_hc_shape(uint32_t h,uint32_t c,uint32_t t,float eps=1e-6f,bool pending=false,uint32_t type=30){
 const uint16_t endian=1;native_hc_need(*reinterpret_cast<const uint8_t*>(&endian)==1,"native HC: little-endian host required");
 native_hc_need(h==2560&&c==4&&t>=1&&t<=8&&eps==1e-6f&&!pending&&type==30,"native HC: require H2560/HC4/T1..8/eps1e-6/raw BF16/no pending write");
}
inline void native_hc_environment(){
 for(const char* key:{"STRATA_HC_Q8","STRATA_HC_Q8_INJECT","STRATA_HC_REQ8","STRATA_GR_V3","STRATA_HC_SPLIT"}){const char* v=std::getenv(key);native_hc_need(!v||!*v,"native HC: require selected recorded-unset HC mode keys; do not substitute another native mode");}
}
inline uint32_t native_hc_bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
inline float native_hc_value(uint32_t b){float x;std::memcpy(&x,&b,4);return x;}
inline float native_hc_normal(float x){const auto b=native_hc_bits(x)&0x7fffffffu;native_hc_need(b<0x7f800000u && (b==0||b>=0x800000u),"native HC: nonfinite/subnormal arithmetic unsupported");return x;}
// Original iq_pack.py --compat-bf16 contract: Q8_0 dequantize F32 then BF16 nearest-even.
inline uint16_t native_hc_bf16_rne(float x){const auto b=native_hc_bits(x);native_hc_need((b&0x7f800000u)!=0x7f800000u,"native HC: nonfinite BF16 source");const auto h=uint16_t((b+0x7fffu+((b>>16)&1u))>>16);native_hc_need((h&0x7f80u)!=0x7f80u,"native HC: BF16 conversion overflow");return h;}
inline std::vector<uint16_t> native_hc_q8_bf16(const uint8_t* bytes,size_t size,size_t elements){
 native_hc_need(bytes&&elements&&elements%32==0&&size==elements/32*34,"native HC: invalid Q8_0 source size");std::vector<uint16_t> out(elements);
 for(size_t block=0;block<elements/32;++block){uint16_t h;std::memcpy(&h,bytes+block*34,2);native_hc_need((h&0x7c00u)!=0x7c00u,"native HC: nonfinite Q8 scale");const float d=ggml_fp16_to_fp32(h);
  for(size_t j=0;j<32;++j){int8_t q;std::memcpy(&q,bytes+block*34+2+j,1);out[block*32+j]=native_hc_bf16_rne(d*float(q));}}
 return out;
}
struct NativeHcOracle {std::vector<float> rs,xn,inject;};
// Host normal-domain control, not a replacement for the actual CUDA source oracle.
// rsqrt implementation/contraction qualification must use native_hc_reference.
inline NativeHcOracle native_hc_oracle(const std::vector<float>& x,const std::vector<float>& w,const std::vector<uint16_t>& wi,uint32_t t){
 native_hc_shape(2560,4,t);native_hc_need(x.size()==size_t(t)*10240&&w.size()==10240&&wi.size()==40960,"native HC: oracle span mismatch");
 for(float f:x)native_hc_normal(f);
 for(float f:w)native_hc_normal(f);
 for(auto h:wi)native_hc_normal(native_hc_value(uint32_t(h)<<16));
 NativeHcOracle out;out.rs.resize(t*4);out.xn.resize(x.size());out.inject.resize(t*4);
 for(uint32_t token=0;token<t;++token){float partial[4][256]{};
  for(uint32_t tid=0;tid<256;++tid)for(uint32_t j=tid*4;j<10240;j+=1024){const uint32_t c=j/2560;const auto* r=x.data()+token*10240+j;
   float sq=native_hc_normal(r[0]*r[0]);for(int k=1;k<4;++k)sq=native_hc_normal(std::fma(r[k],r[k],sq));partial[c][tid]=native_hc_normal(partial[c][tid]+sq);}
  for(uint32_t offset=16;offset;offset>>=1){auto old=std::array<std::array<float,256>,4>{};for(int c=0;c<4;++c)std::memcpy(old[c].data(),partial[c],1024);for(int c=0;c<4;++c)for(int i=0;i<256;++i)partial[c][i]=native_hc_normal(old[c][i]+old[c][i^offset]);}
  for(int c=0;c<4;++c){float sum=0;for(int warp=0;warp<8;++warp)sum=native_hc_normal(sum+partial[c][warp*32]);const float mean=native_hc_normal(sum/2560.0f);out.rs[token*4+c]=native_hc_normal(1.0f/std::sqrt(native_hc_normal(mean+1e-6f)));}
  for(uint32_t j=0;j<10240;++j)out.xn[token*10240+j]=native_hc_normal(native_hc_normal(x[token*10240+j]*w[j])*out.rs[token*4+j/2560]);
  for(uint32_t row=0;row<4;++row){float lane[32]{};for(uint32_t i=0;i<32;++i)for(uint32_t chunk=i;chunk<1280;chunk+=32){float acc=0;for(uint32_t k=0;k<8;++k)acc=native_hc_normal(std::fma(native_hc_value(uint32_t(wi[row*10240+chunk*8+k])<<16),out.xn[token*10240+chunk*8+k],acc));lane[i]=native_hc_normal(lane[i]+acc);}
   for(uint32_t offset=16;offset;offset>>=1){float old[32];std::memcpy(old,lane,sizeof lane);for(int i=0;i<32;++i)lane[i]=native_hc_normal(old[i]+old[i^offset]);}out.inject[token*4+row]=lane[0];}
 }
 return out;
}
}
