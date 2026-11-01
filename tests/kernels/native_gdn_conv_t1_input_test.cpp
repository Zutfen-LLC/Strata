#include "native_gdn_conv_fixture.hpp"
using namespace native_gdn_conv;
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],native_gdn_conv::synthetic(unsigned(std::stoul(argv[2]))));return 0;}
 if(argc==4&&std::string(argv[1])=="--check"){
  auto f=read(argv[2]);std::filesystem::path d=argv[3];auto get=[&](const char*name,size_t n){std::ifstream in(d/(std::string(name)+".f32le"),std::ios::binary);native_hc_need(bool(in),"GDN conv result missing");std::vector<float>a(n);for(float&x:a)x=value(word(in));native_hc_need(in.peek()==EOF,"GDN conv result shape");return a;};auto ref=control(f,get("exp",10240),get("rs",32),true);
  exact(ref.history,get("next-history",30720),"native history commit order");exact(ref.pre,get("pre",10240),"native convolution contraction/order");exact(ref.silu,get("silu",10240),"native own-exp SiLU");exact(ref.arg,get("norm-arg",32),"native q/k reduction/epsilon");exact(ref.h,get("h",10240),"native q/k normalization and v bypass");puts("GDN conv exact source-order own-primitive/history check pass");return 0;
 }
 native_hc_need(argc==1,"GDN conv host arguments");auto f=native_gdn_conv::synthetic(1);std::vector<float>ex(10240,1),rs(32,1);auto out=control(f,ex,rs,true),verify=control(f,ex,rs,false);
 for(size_t c:{size_t(0),size_t(4095),size_t(4096),size_t(10239)}){native_hc_need(out.pre[c]==10&&out.silu[c]==5&&out.h[c]==5,"independent four-tap conv/head-boundary vector");native_hc_need(out.history[c*3]==2&&out.history[c*3+1]==3&&out.history[c*3+2]==4,"independent native history shift");}
 native_hc_need(bits(out.arg[0])==0x41c80001&&bits(out.arg[31])==0x41c80001&&bits(out.arg[1])==0x358637bd,"independent q/k square/reduction/epsilon vectors");
 auto scaled_rs=rs;scaled_rs[0]=.25f;scaled_rs[31]=.5f;auto scaled=control(f,ex,scaled_rs,true);native_hc_need(scaled.h[0]==1.25f&&scaled.h[4095]==2.5f&&scaled.h[4096]==5.f,"independent q/k normalization versus v bypass");
 exact(verify.history,f[0],"verify history changed");exact(verify.h,out.h,"commit changed arithmetic");
 // Contraction-sensitive pair: y*y rounded first, x*x fused. This is an
 // explicit source-control candidate; actual CUDA launcher must confirm it.
 f[0][0]=value(0x40330828);f[0][1]=value(0x402e4d67);f[0][2]=f[1][0]=0;f[2][0]=f[0][0];f[2][1]=f[0][1];f[2][2]=f[2][3]=0;
 auto sensitive=control(f,ex,rs,true);native_hc_need(bits(sensitive.pre[0])==0x4173e1b8,"independent rounded-second/FMA-first convolution vector");
 auto bad=sensitive.h;bad[0]=1;bool rejected=false;try{exact(bad,sensitive.h,"wrong shared result");}catch(...){rejected=true;}native_hc_need(rejected,"incorrect common result accepted");
 f[0][0]=INFINITY;rejected=false;try{validate(f);}catch(...){rejected=true;}native_hc_need(rejected,"nonfinite history accepted");puts("GDN conv pinned order/layout/verify/commit/invalid controls pass");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
