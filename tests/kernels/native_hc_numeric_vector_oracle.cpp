// Reproduce pinned generic vectors independently of the Python validator.
// c++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off this.cpp -o oracle
// ./oracle /fresh/output-directory
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
constexpr int D=10240, H=2560, T=3;
uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
float value(uint32_t u){float x;std::memcpy(&x,&u,4);return x;}
void word(std::ofstream& f,uint32_t u,int bytes=4){for(int i=0;i<bytes;++i)f.put(char(u>>(8*i)));}
void save(const std::filesystem::path& p,const std::vector<float>& a){std::ofstream f(p,std::ios::binary);for(float x:a)word(f,bits(x));if(!f)throw std::runtime_error("write");}
int main(int argc,char**argv){
 static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);
 if(argc!=2||!std::filesystem::create_directory(argv[1]))return 1;
 const std::filesystem::path out=argv[1];uint32_t state=0x1539;
 auto next=[&](){state=state*1664525u+1013904223u;return state;};
 std::vector<float>x(T*D),w(D),wi(4*D),weighted(T*D),xn(T*D),sums,inject;
 for(int t=0;t<T;++t)for(int j=0;j<D;++j){uint32_t s=next();x[t*D+j]=std::ldexp(value((0x3e000000u+(s&0x3ffffffu))|(s&0x80000000u)),t-1);}
 for(float& v:w)v=float(64+next()%67)/128.f;
 for(float& v:wi)v=float(int(next()%129)-64)/256.f;
 std::ofstream inputs(out/"inputs.bin",std::ios::binary);
 for(float v:x)word(inputs,bits(v));for(float v:w)word(inputs,bits(v));for(float v:wi)word(inputs,bits(v)>>16,2);inputs.close();
 const std::array<float,12>rs{.5f,1.f,2.f,4.f,.25f,.5f,1.f,2.f,.125f,.25f,.5f,1.f};
 for(int t=0;t<T;++t){float p[4][256]{};
  for(int tid=0;tid<256;++tid)for(int j=tid*4;j<D;j+=1024){const float*r=x.data()+t*D+j;float q=r[1]*r[1];q=std::fma(r[0],r[0],q);q=std::fma(r[2],r[2],q);q=std::fma(r[3],r[3],q);p[j/H][tid]+=q;}
  for(int offset:{16,8,4,2,1}){float old[4][256];std::memcpy(old,p,sizeof p);for(int c=0;c<4;++c)for(int i=0;i<256;++i)p[c][i]=old[c][i]+old[c][i^offset];}
  for(int c=0;c<4;++c){float s=0;for(int warp=0;warp<8;++warp)s+=p[c][warp*32];sums.push_back(s);}
  for(int j=0;j<D;++j){weighted[t*D+j]=x[t*D+j]*w[j];xn[t*D+j]=weighted[t*D+j]*rs[t*4+j/H];}
  for(int row=0;row<4;++row){float lane[32]{};for(int i=0;i<32;++i)for(int chunk=i;chunk<1280;chunk+=32){float a=0;for(int k=0;k<8;++k)a=std::fma(wi[row*D+chunk*8+k],xn[t*D+chunk*8+k],a);lane[i]+=a;}for(int offset:{16,8,4,2,1}){float old[32];std::memcpy(old,lane,sizeof lane);for(int i=0;i<32;++i)lane[i]=old[i]+old[i^offset];}inject.push_back(lane[0]);}
 }
 save(out/"sums.bin",sums);save(out/"weighted.bin",weighted);save(out/"normalized.bin",xn);save(out/"inject.bin",inject);
 for(float v:sums)std::printf("%08x ",bits(v));std::puts("");for(float v:inject)std::printf("%08x ",bits(v));std::puts("");
}
