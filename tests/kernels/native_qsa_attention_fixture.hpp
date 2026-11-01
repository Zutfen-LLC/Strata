#pragma once
#include "native_qsa_select_fixture.hpp"
#include "native_qsa_attention_layout.hpp"
#include "native_gdn_domain.hpp"
#include "ggml.h"
#include <cfloat>
namespace native_qsa_attn {
using namespace native_hc_fixture;
using namespace strata::core::portable_detail;
constexpr uint32_t canary=0x4acaf00d;
inline std::array<size_t,13> counts(uint32_t cells){size_t pages=(cells+3)/4;return {{pages*1024,pages*1024,6144,pages,2051,4,12288,204400,101376,6984,6144,12288,6144}};}
struct Fixture{uint32_t cells=256;std::array<std::vector<uint32_t>,7>in;};
inline std::array<int32_t,4> step(const Fixture&f){std::array<int32_t,4>s{};std::memcpy(s.data(),f.in[5].data(),16);return s;}
inline float half(const std::vector<uint32_t>&v,size_t i){return ggml_fp16_to_fp32(ggml_fp16_t((v[i/2]>>((i%2)*16))&65535));}
inline void set_half(std::vector<uint32_t>&v,size_t i,float x){uint32_t shift=(i%2)*16,h=ggml_fp32_to_fp16(x);v[i/2]=(v[i/2]&~(65535u<<shift))|(h<<shift);}
inline void validate(const Fixture&f){auto n=counts(f.cells);for(size_t i=0;i<7;++i)native_hc_need(f.in[i].size()==n[i],"QSA attention exact fixture spans");auto s=step(f);native_qsa_attention_geometry(f.cells,s);for(auto b:f.in[2])native_hc_normal(value(b));for(size_t h=0;h<24;++h)for(size_t d=0;d<256;++d)native_gdn_sigmoid_input(value(f.in[6][h*512+256+d]));for(auto page:f.in[3])native_hc_need(int32_t(page)>=-1&&int32_t(page)<int32_t(n[3]),"QSA attention page bounds");for(int i=0;i<s[3];++i){uint32_t cell=f.in[4][i];native_hc_need(cell<uint32_t(s[1])&&(i==0||cell>f.in[4][i-1]),"QSA attention ascending ID bounds");int page=int32_t(f.in[3][cell/4]);if(page<0)continue;for(size_t h=0;h<2;++h)for(size_t d=0;d<256;++d)for(size_t side=0;side<2;++side)native_hc_normal(half(f.in[side],((page*2+h)*4+cell%4)*256+d));}}
inline float add(float a,float b){return native_hc_normal(a+b);}inline float mul(float a,float b){return native_hc_normal(a*b);}inline float fma(float a,float b,float c){return native_hc_normal(std::fma(a,b,c));}
inline float maximum(float a,float b){return a==0&&b==0?value(bits(a)&bits(b)):std::max(a,b);}
inline float sum32(std::array<float,32>a){for(size_t o:{16,8,4,2,1}){auto old=a;for(size_t i=0;i<32;++i)a[i]=add(old[i],old[i^o]);}return a[0];}
inline float max32(std::array<float,32>a){for(size_t o:{16,8,4,2,1}){auto old=a;for(size_t i=0;i<32;++i)a[i]=maximum(old[i],old[i^o]);}return a[0];}
struct Control{std::array<std::vector<uint32_t>,6>out;std::vector<float>chunk_args,merge_args,gate_args;};
// Supplied primitive values are local controls, not replacement execution.
// Original CUDA attention/gate launchers remain unchanged in the native probe.
inline Control control(const Fixture&f,const std::vector<float>&exp_chunk={},const std::vector<float>&exp_merge={},const std::vector<float>&exp_gate={},const std::vector<float>&sigmoid={}){
 validate(f);native_hc_need((exp_chunk.empty()||exp_chunk.size()==50688)&&(exp_merge.empty()||exp_merge.size()==792)&&(exp_gate.empty()||exp_gate.size()==6144)&&(sigmoid.empty()||sigmoid.size()==6144),"QSA primitive control spans");Control r;auto n=counts(f.cells);for(size_t i=0;i<6;++i)r.out[i].assign(n[i+7],canary);r.chunk_args.assign(50688,0);r.merge_args.assign(792,0);r.gate_args.assign(6144,0);auto&s=r.out[0];auto&aux=r.out[1];auto st=step(f);
 for(size_t kv=0;kv<2;++kv)for(size_t chunk=0;chunk<33;++chunk){int here=std::min(64,st[3]-int(chunk*64));size_t slot=kv*33+chunk;
  if(here<=0){for(size_t h=0;h<12;++h){s[202752+slot*12+h]=bits(-FLT_MAX);s[203544+slot*12+h]=0;}continue;}
  std::array<int32_t,64>row;row.fill(-1);for(int c=0;c<here;++c){uint32_t cell=f.in[4][chunk*64+c];int32_t page=int32_t(f.in[3][cell/4]);if(page>=0)row[c]=(page*2+int(kv))*4+int(cell%4);}
  for(size_t h=0;h<12;++h){size_t head=kv*12+h;std::array<float,64>logits{},prob{};
   for(size_t c=0;c<64;++c){float score=-FLT_MAX;if(c<size_t(here)&&row[c]>=0){std::array<float,32>lane{};for(size_t l=0;l<32;++l){size_t d=l*8,at=head*256+d;float v=mul(half(f.in[0],row[c]*256+d),value(f.in[2][at]));for(size_t j=1;j<8;++j)v=fma(half(f.in[0],row[c]*256+d+j),value(f.in[2][at+j]),v);lane[l]=v;}score=mul(sum32(lane),.0625f);}logits[c]=score;aux[head*2112+chunk*64+c]=bits(score);}
   std::array<float,32>maxpair{},sumpair{};for(size_t l=0;l<32;++l)maxpair[l]=maximum(logits[l],logits[l+32]);float m=max32(maxpair);
   for(size_t c=0;c<64;++c){size_t at=head*2112+chunk*64+c;if(c<size_t(here)&&row[c]>=0){r.chunk_args[at]=add(logits[c],-m);prob[c]=native_hc_normal(exp_chunk.empty()?::expf(r.chunk_args[at]):exp_chunk[at]);}aux[50688+at]=bits(prob[c]);}
   for(size_t l=0;l<32;++l)sumpair[l]=add(prob[l],prob[l+32]);s[202752+slot*12+h]=bits(m);s[203544+slot*12+h]=bits(sum32(sumpair));
   for(size_t d=0;d<256;++d){float acc=0;for(int c=0;c<here;++c)if(row[c]>=0)acc=fma(prob[c],half(f.in[1],row[c]*256+d),acc);s[(slot*12+h)*256+d]=bits(acc);}
  }
 }
 for(size_t h=0;h<24;++h){size_t kv=h/12,hl=h%12;float M=-FLT_MAX;for(size_t c=0;c<33;++c)M=maximum(M,value(s[202752+(kv*33+c)*12+hl]));std::array<float,33>w{};float L=0;
  for(size_t c=0;c<33;++c){size_t at=h*33+c,slot=kv*33+c;float m=value(s[202752+slot*12+hl]);if(m==-FLT_MAX)continue;r.merge_args[at]=add(m,-M);w[c]=native_hc_normal(exp_merge.empty()?::expf(r.merge_args[at]):exp_merge[at]);r.out[2][at]=bits(w[c]);L=fma(value(s[203544+slot*12+hl]),w[c],L);}
  r.out[2][792+h]=bits(M);r.out[2][816+h]=bits(L);
  for(size_t d=0;d<256;++d){float acc=0;for(size_t c=0;c<33;++c){size_t slot=kv*33+c;if(value(s[202752+slot*12+hl])==-FLT_MAX)continue;acc=fma(value(s[(slot*12+hl)*256+d]),w[c],acc);}size_t i=h*256+d;r.out[2][840+i]=bits(acc);float attn=L>0?native_hc_normal(acc/L):0;r.out[3][i]=bits(attn);float raw=value(f.in[6][h*512+256+d]);r.gate_args[i]=-raw;float ev=exp_gate.empty()?::expf(-raw):exp_gate[i];float sig=sigmoid.empty()?1.f/(1.f+ev):sigmoid[i];native_hc_normal(sig);r.out[4][i]=bits(ev);r.out[4][6144+i]=bits(sig);r.out[5][i]=bits(mul(attn,sig));}
 }
 return r;
}
inline Fixture synthetic(unsigned mode){native_hc_need(mode<7,"QSA attention fixture mode");Fixture f;auto n=counts(f.cells);for(size_t i=0;i<7;++i)f.in[i].assign(n[i],0);const uint32_t lengths[]={1,1,65,129,256,63,64};uint32_t nk=lengths[mode];f.in[5]={nk-1,nk,nk/4,nk};for(size_t i=0;i<f.in[3].size();++i)f.in[3][i]=mode==4?uint32_t(f.in[3].size()-1-i):uint32_t(i);if(mode==5)std::fill(f.in[3].begin(),f.in[3].end(),uint32_t(-1));if(mode==4)f.in[3][3]=uint32_t(-1);std::fill(f.in[4].begin(),f.in[4].end(),canary);for(size_t i=0;i<nk;++i)f.in[4][i]=uint32_t(i);
 for(size_t page=0;page<f.in[3].size();++page)for(size_t h=0;h<2;++h)for(size_t c=0;c<4;++c)for(size_t d=0;d<256;++d){size_t at=((page*2+h)*4+c)*256+d;float k=0,v=0;if(mode==1||mode==2)v=float(int(d%17)-8)/16;if(mode==3||mode==4||mode==6){k=float(int((at*7)%31)-15)/16;v=float(int((at*13)%37)-18)/16;}set_half(f.in[0],at,k);set_half(f.in[1],at,v);}
 for(size_t i=0;i<6144;++i){float q=mode>=3?float(int((i*17)%29)-14)/32:0;f.in[2][i]=bits(q);float gate=mode==4?(i%3==0?128.f:i%3==1?-128.f:0.f):mode==3?float(int(i%19)-9):0;f.in[6][(i/256)*512+256+i%256]=bits(gate);}
 if(mode==6){std::fill(f.in[2].begin(),f.in[2].end(),0);for(size_t h=0;h<24;++h){const float q[]={0x1p24f,1,-0x1p24f,1,0,0,0,0};for(size_t j=0;j<8;++j)f.in[2][h*256+j]=bits(q[j]);}for(size_t page=0;page<f.in[3].size();++page)for(size_t h=0;h<2;++h)for(size_t c=0;c<4;++c)for(size_t j=0;j<8;++j)set_half(f.in[0],((page*2+h)*4+c)*256+j,1);}
 if(mode==1)set_half(f.in[1],17,0x1p-24f);
 validate(f);return f;
}
inline void write(const std::string&p,const Fixture&f){validate(f);native_hc_need(!std::filesystem::exists(p),"QSA attention fresh fixture");std::ofstream o(p,std::ios::binary);for(uint32_t w:{0x3141514eu,1u,f.cells})word(o,w);for(auto&v:f.in)for(auto w:v)word(o,w);native_hc_need(bool(o),"QSA attention fixture write");}
inline Fixture read(const std::string&p){std::ifstream i(p,std::ios::binary);native_hc_need(bool(i)&&word(i)==0x3141514e&&word(i)==1,"QSA attention header");Fixture f;f.cells=word(i);native_hc_need(f.cells>=1&&f.cells<=256,"QSA attention capacity");auto n=counts(f.cells);for(size_t j=0;j<7;++j){f.in[j].resize(n[j]);for(auto&w:f.in[j])w=word(i);}native_hc_need(i.peek()==EOF,"QSA attention fixture span");validate(f);return f;}
inline const std::array<const char*,6> names{{"scratch.f32le","chunk.f32le","merge.f32le","attn.f32le","gate.f32le","gated.f32le"}};
inline void contract(const Fixture&f,const std::string&p){validate(f);native_hc_need(!std::filesystem::exists(p),"QSA attention fresh contract");std::ofstream o(p);o<<"{\"fields\":{";auto n=counts(f.cells);for(size_t i=0;i<6;++i){if(i)o<<',';o<<'"'<<names[i]<<"\":"<<n[7+i]*4;}o<<"},\"qualified\":false}\n";native_hc_need(bool(o),"QSA attention contract write");}
inline std::array<std::vector<uint32_t>,6> outputs(const Fixture&f,const std::filesystem::path&d){auto n=counts(f.cells);std::array<std::vector<uint32_t>,6>v;for(size_t i=0;i<6;++i)v[i]=native_qsa_ss::words(d/names[i],n[7+i]);return v;}
inline void check(const Fixture&f,const std::filesystem::path&d){auto v=outputs(f,d);std::vector<float>ec(50688),em(792),eg(6144),sig(6144);for(size_t i=0;i<ec.size();++i)ec[i]=value(v[1][50688+i]);for(size_t i=0;i<em.size();++i)em[i]=value(v[2][i]);for(size_t i=0;i<eg.size();++i){eg[i]=value(v[4][i]);sig[i]=value(v[4][6144+i]);}auto ref=control(f,ec,em,eg,sig);for(size_t i=0;i<6;++i)native_hc_need(v[i]==ref.out[i],"QSA attention exact source-order/own-primitive/inactive fields differ");}
inline void save(const std::filesystem::path&d,const Control&r){for(size_t i=0;i<6;++i)native_qsa_ss::put(d/names[i],r.out[i]);}
}
