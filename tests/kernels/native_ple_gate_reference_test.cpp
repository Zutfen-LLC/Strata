// Frozen arithmetic controls for native_ple_postops.cu::gate_kernel.
// The sigmoid's platform-specific exp/div result is not a CUDA parity oracle.
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t columns=2560, lanes=512, rows=4;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
float add(float a,float b){volatile float r=a+b;return r;}
float mul(float a,float b){volatile float r=a*b;return r;}
uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
float from_bits(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
float row_sum(const float* key,const float* query){
    std::array<float,lanes> values{};
    for(size_t tid=0;tid<lanes;++tid){
        std::array<float,8> products{};
        for(size_t j=0;j<8;++j){size_t d=tid+j*lanes;float p=d<columns?mul(key[d],query[d]):0.0f;products[j]=add(0.0f,p);}
        float sum=0.0f;for(float p:products)sum=add(sum,p);values[tid]=sum;
    }
    for(unsigned delta=16;delta;delta>>=1){auto before=values;for(size_t tid=0;tid<lanes;++tid)values[tid]=add(before[tid],before[tid^delta]);}
    std::array<float,32> warp{};for(size_t i=0;i<16;++i)warp[i]=values[i*32];
    for(unsigned delta=16;delta;delta>>=1){auto before=warp;for(size_t lane=0;lane<32;++lane)warp[lane]=add(before[lane],before[lane^delta]);}
    return warp[0];
}
float host_gate(float sum){
    float score=std::fma(from_bits(0x3ca1e89b),sum,0.0f);
    float mag=std::sqrt(std::fmax(std::fabs(score),1e-6f));
    float sign=float((score>0.f)-(score<0.f));
    return 1.0f/(1.0f+std::exp(-mul(sign,mag)));
}
}
int main(){try{
    require(std::fesetround(FE_TONEAREST)==0,"rounding mode");
    std::vector<float> k(rows*columns,1.0f),q(rows*columns,0.0f);
    for(size_t i=0;i<columns;++i){q[i]=1.0f;q[2*columns+i]=-1.0f;}
    for(size_t tid=0;tid<lanes;++tid){q[columns+tid]=16777216.0f;q[columns+tid+lanes]=1.0f;q[columns+tid+2*lanes]=-16777216.0f;q[columns+tid+3*lanes]=1.0f;}
    q[3*columns]=16777216.0f;q[3*columns+1]=1.0f;q[3*columns+2]=-16777216.0f;q[3*columns+3]=1.0f;
    const std::array<float,rows> expected{{2560.0f,512.0f,-2560.0f,2.0f}};
    for(size_t r=0;r<rows;++r)require(bits(row_sum(k.data()+r*columns,q.data()+r*columns))==bits(expected[r]),"source-order reduction literal differs");
    require(bits(row_sum(k.data(),std::vector<float>(columns,0.0f).data()))==0,"zero reduction differs");
    require(host_gate(0.0f)==0.5f && host_gate(512.0f)>0.5f && host_gate(-2560.0f)<0.5f,"host primitive classification differs");
    require(bits(from_bits(0x3ca1e89b))==0x3ca1e89b,"native host scale constant differs");
    std::puts("Four independent literal PLE gate reductions and host primitive classifications passed; cross-backend intrinsics unqualified.");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"PLE gate reference: %s\n",e.what());return 1;}}
