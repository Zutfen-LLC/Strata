#include "native_gdn_ab_fixture.hpp"
#include "strata_llvm_log1p_oracle.hpp"
#include "native_gdn_log1p_domain.hpp"
using namespace native_gdn_ab;
int main(int argc,char**argv){try{native_hc_need(std::fesetround(FE_TONEAREST)==0,"GDN ab host RNE");
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],native_gdn_ab::synthetic(std::stoul(argv[2])));return 0;}
 if(argc==3&&std::string(argv[1])=="--contract"){contract(argv[2]);return 0;}
 if(argc==4&&std::string(argv[1])=="--oracle"){
  auto f=read(argv[2]);auto d=dot(f);std::filesystem::path dir=argv[3];native_hc_need(std::filesystem::create_directory(dir),"GDN ab fresh CPU control");std::filesystem::copy_file(argv[2],dir/"fixture.bin");std::array<std::vector<float>,9>v;for(auto&a:v)a.resize(H);v[0]=d[0];v[1]=d[1];for(size_t i=0;i<H;++i){v[2][i]=d[0][i]+f.dt[i];if(v[2][i]>20){v[3][i]=0;v[4][i]=v[2][i];}else{float e=std::exp(v[2][i]);if(std::fpclassify(e)==FP_SUBNORMAL)e=0;v[3][i]=e;v[4][i]=strata_llvm_upstream::math::log1pf(e);}v[5][i]=std::exp(-d[1][i]);v[6][i]=1.f+v[5][i];v[7][i]=v[4][i]*f.a[i];v[8][i]=1.f/v[6][i];}for(size_t i=0;i<9;++i)floats((dir/(std::string(fields[i])+".f32le")).string(),v[i]);return 0;
 }
 if(argc==5&&std::string(argv[1])=="--check"){
  auto f=read(argv[2]);auto d=dot(f);std::array<std::vector<float>,9>v;for(size_t i=0;i<9;++i){std::ifstream in(std::filesystem::path(argv[3])/(std::string(fields[i])+".f32le"),std::ios::binary);native_hc_need(bool(in),"GDN ab missing output");v[i].resize(H);for(float&x:v[i])x=value(word(in));native_hc_need(in.peek()==EOF,"GDN ab output shape");}exact(d[0],v[0],"GDN ab continuous FMA/alpha reduction");exact(d[1],v[1],"GDN ab beta row/reduction");
  for(size_t h=0;h<H;++h){float x=native_hc_normal(d[0][h]+f.dt[h]);native_hc_need(bits(x)==bits(v[2][h]),"GDN ab dt association");if(x>20){native_hc_need(bits(v[3][h])==0&&bits(v[4][h])==bits(x),"GDN ab strict >20 bypass");}else if(std::string(argv[4])=="vulkan"){native_gdn_log1p_input(v[3][h]);native_hc_need(bits(v[4][h])==bits(strata_llvm_upstream::math::log1pf(v[3][h])),"GDN ab LLVM original source");}
   float gate=native_hc_normal(v[4][h]*f.a[h]);native_hc_need(bits(gate)==bits(v[7][h]),"GDN ab gate multiplication/order");volatile float den=1.f+v[5][h];native_hc_need(bits(den)==bits(v[6][h]),"GDN ab sigmoid denominator association");native_hc_normal(v[8][h]);}
  return 0;
 }
 auto f=native_gdn_ab::synthetic(3);auto d=dot(f);native_hc_need(bits(d[0][0])==0&&d[0][1]==1.f&&bits(d[0][2])==0x34010000,"GDN ab independent continuous-FMA/XOR/fused vectors");auto branches=native_gdn_ab::synthetic(1);d=dot(branches);native_hc_need(bits(d[0][4]+branches.dt[4])==0x419fffff&&bits(d[0][5]+branches.dt[5])==0x41a00000&&bits(d[0][6]+branches.dt[6])==0x41a00001,"GDN ab strict threshold vectors");f.wa[0]=0x7f80;bool refused=false;try{validate(f);}catch(...){refused=true;}native_hc_need(refused,"GDN ab nonfinite BF16 accepted");puts("GDN ab pinned layout/continuous FMA/XOR/threshold and invalid controls pass");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
