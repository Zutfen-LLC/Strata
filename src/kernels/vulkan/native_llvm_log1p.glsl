// Derived from LLVM libc at 4497566c7ca6b1d958fe27678f541308b948f6c2.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See third_party/llvm-libc-log1p/NOTICE.md and LICENSE.TXT.
// Existing accurate, non-FMA configuration; positive normal/zero <=2^29, RNE.
// No synthesized coefficients. All constants retain original binary64 bits.
#ifndef STRATA_LLVM_LOG1P
#define STRATA_LLVM_LOG1P
const uvec2 llvm_R_LOG[64] = uvec2[64](
uvec2(0x00000000u,0x3ff00000u),
uvec2(0x00000000u,0x3fef8000u),
uvec2(0x00000000u,0x3fef0000u),
uvec2(0x00000000u,0x3fee8000u),
uvec2(0x00000000u,0x3fee0000u),
uvec2(0x00000000u,0x3fed8000u),
uvec2(0x00000000u,0x3fed4000u),
uvec2(0x00000000u,0x3fecc000u),
uvec2(0x00000000u,0x3fec4000u),
uvec2(0x00000000u,0x3fec0000u),
uvec2(0x00000000u,0x3feb8000u),
uvec2(0x00000000u,0x3feb4000u),
uvec2(0x00000000u,0x3feac000u),
uvec2(0x00000000u,0x3fea8000u),
uvec2(0x00000000u,0x3fea4000u),
uvec2(0x00000000u,0x3fe9c000u),
uvec2(0x00000000u,0x3fe98000u),
uvec2(0x00000000u,0x3fe94000u),
uvec2(0x00000000u,0x3fe90000u),
uvec2(0x00000000u,0x3fe88000u),
uvec2(0x00000000u,0x3fe84000u),
uvec2(0x00000000u,0x3fe80000u),
uvec2(0x00000000u,0x3fe7c000u),
uvec2(0x00000000u,0x3fe78000u),
uvec2(0x00000000u,0x3fe74000u),
uvec2(0x00000000u,0x3fe70000u),
uvec2(0x00000000u,0x3fe6c000u),
uvec2(0x00000000u,0x3fe68000u),
uvec2(0x00000000u,0x3fe64000u),
uvec2(0x00000000u,0x3fe60000u),
uvec2(0x00000000u,0x3fe5c000u),
uvec2(0x00000000u,0x3fe58000u),
uvec2(0x00000000u,0x3fe54000u),
uvec2(0x00000000u,0x3fe50000u),
uvec2(0x00000000u,0x3fe4c000u),
uvec2(0x00000000u,0x3fe4c000u),
uvec2(0x00000000u,0x3fe48000u),
uvec2(0x00000000u,0x3fe44000u),
uvec2(0x00000000u,0x3fe40000u),
uvec2(0x00000000u,0x3fe3c000u),
uvec2(0x00000000u,0x3fe3c000u),
uvec2(0x00000000u,0x3fe38000u),
uvec2(0x00000000u,0x3fe34000u),
uvec2(0x00000000u,0x3fe30000u),
uvec2(0x00000000u,0x3fe30000u),
uvec2(0x00000000u,0x3fe2c000u),
uvec2(0x00000000u,0x3fe28000u),
uvec2(0x00000000u,0x3fe28000u),
uvec2(0x00000000u,0x3fe24000u),
uvec2(0x00000000u,0x3fe20000u),
uvec2(0x00000000u,0x3fe20000u),
uvec2(0x00000000u,0x3fe1c000u),
uvec2(0x00000000u,0x3fe1c000u),
uvec2(0x00000000u,0x3fe18000u),
uvec2(0x00000000u,0x3fe14000u),
uvec2(0x00000000u,0x3fe14000u),
uvec2(0x00000000u,0x3fe10000u),
uvec2(0x00000000u,0x3fe10000u),
uvec2(0x00000000u,0x3fe0c000u),
uvec2(0x00000000u,0x3fe0c000u),
uvec2(0x00000000u,0x3fe08000u),
uvec2(0x00000000u,0x3fe08000u),
uvec2(0x00000000u,0x3fe04000u),
uvec2(0x00000000u,0x3fe00000u));
const uvec2 llvm_C_LOG[64] = uvec2[64](
uvec2(0x00000000u,0x00000000u),
uvec2(0x00000000u,0xbf300000u),
uvec2(0x00000000u,0xbf500000u),
uvec2(0x00000000u,0xbf620000u),
uvec2(0x00000000u,0xbf700000u),
uvec2(0x00000000u,0xbf790000u),
uvec2(0x00000000u,0xbf300000u),
uvec2(0x00000000u,0xbf6b0000u),
uvec2(0x00000000u,0xbf7c0000u),
uvec2(0x00000000u,0xbf600000u),
uvec2(0x00000000u,0xbf7a0000u),
uvec2(0x00000000u,0xbf610000u),
uvec2(0x00000000u,0xbf7e0000u),
uvec2(0x00000000u,0xbf6e0000u),
uvec2(0x00000000u,0xbf300000u),
uvec2(0x00000000u,0xbf7b8000u),
uvec2(0x00000000u,0xbf700000u),
uvec2(0x00000000u,0xbf560000u),
uvec2(0x00000000u,0x3f500000u),
uvec2(0x00000000u,0xbf7d0000u),
uvec2(0x00000000u,0xbf760000u),
uvec2(0x00000000u,0xbf700000u),
uvec2(0x00000000u,0xbf660000u),
uvec2(0x00000000u,0xbf5c0000u),
uvec2(0x00000000u,0xbf500000u),
uvec2(0x00000000u,0xbf400000u),
uvec2(0x00000000u,0xbf300000u),
uvec2(0x00000000u,0xbf300000u),
uvec2(0x00000000u,0xbf400000u),
uvec2(0x00000000u,0xbf500000u),
uvec2(0x00000000u,0xbf5c0000u),
uvec2(0x00000000u,0xbf660000u),
uvec2(0x00000000u,0xbf700000u),
uvec2(0x00000000u,0xbf760000u),
uvec2(0x00000000u,0xbf7d0000u),
uvec2(0x00000000u,0x3f690000u),
uvec2(0x00000000u,0x3f500000u),
uvec2(0x00000000u,0xbf560000u),
uvec2(0x00000000u,0xbf700000u),
uvec2(0x00000000u,0xbf7b8000u),
uvec2(0x00000000u,0x3f680000u),
uvec2(0x00000000u,0xbf300000u),
uvec2(0x00000000u,0xbf6e0000u),
uvec2(0x00000000u,0xbf7e0000u),
uvec2(0x00000000u,0x3f600000u),
uvec2(0x00000000u,0xbf610000u),
uvec2(0x00000000u,0xbf7a0000u),
uvec2(0x00000000u,0x3f660000u),
uvec2(0x00000000u,0xbf600000u),
uvec2(0x00000000u,0xbf7c0000u),
uvec2(0x00000000u,0x3f600000u),
uvec2(0x00000000u,0xbf6b0000u),
uvec2(0x00000000u,0x3f760000u),
uvec2(0x00000000u,0xbf300000u),
uvec2(0x00000000u,0xbf790000u),
uvec2(0x00000000u,0x3f630000u),
uvec2(0x00000000u,0xbf700000u),
uvec2(0x00000000u,0x3f720000u),
uvec2(0x00000000u,0xbf620000u),
uvec2(0x00000000u,0x3f788000u),
uvec2(0x00000000u,0xbf500000u),
uvec2(0x00000000u,0x3f7d0000u),
uvec2(0x00000000u,0xbf300000u),
uvec2(0x00000000u,0xbf800000u));
const uvec2 llvm_LOG_R[64] = uvec2[64](
uvec2(0x00000000u,0x00000000u),
uvec2(0x58935847u,0x3f902056u),
uvec2(0x89e74444u,0x3fa0415du),
uvec2(0x149fb343u,0x3fa894aau),
uvec2(0xb59e3a07u,0x3fb08598u),
uvec2(0x5d207eacu,0x3fb4d311u),
uvec2(0x0aeac0e1u,0x3fb700d3u),
uvec2(0x8dad5b1cu,0x3fbb6ac8u),
uvec2(0x39dbd566u,0x3fbfe891u),
uvec2(0x8227e47cu,0x3fc1178eu),
uvec2(0xb0159016u,0x3fc365fcu),
uvec2(0x8333b561u,0x3fc4913du),
uvec2(0x8b756abcu,0x3fc6f012u),
uvec2(0x6551a3c2u,0x3fc823c1u),
uvec2(0xdcf7017fu,0x3fc95a5au),
uvec2(0x383bd8adu,0x3fcbd087u),
uvec2(0xf2655e7bu,0x3fcd1037u),
uvec2(0xffe71012u,0x3fce530eu),
uvec2(0x6cb3b379u,0x3fcf991cu),
uvec2(0x8227e47cu,0x3fd1178eu),
uvec2(0x635a6b95u,0x3fd1bf99u),
uvec2(0x1134db92u,0x3fd26962u),
uvec2(0xe1d35ce4u,0x3fd314f1u),
uvec2(0x77333184u,0x3fd3c252u),
uvec2(0xc271c41bu,0x3fd4718du),
uvec2(0x0738a3d8u,0x3fd522aeu),
uvec2(0xdf595f30u,0x3fd5d5bdu),
uvec2(0x3e9c6a14u,0x3fd68ac8u),
uvec2(0x76c67bb1u,0x3fd741d8u),
uvec2(0x3bd8151cu,0x3fd7fafau),
uvec2(0xa88b2df5u,0x3fd8b639u),
uvec2(0x431356aeu,0x3fd973a3u),
uvec2(0x0224fa79u,0x3fda3344u),
uvec2(0x5248cdd0u,0x3fdaf529u),
uvec2(0x1b80e2fbu,0x3fdbb961u),
uvec2(0x1b80e2fbu,0x3fdbb961u),
uvec2(0xc74554c9u,0x3fdc7ff9u),
uvec2(0x46defa6bu,0x3fdd4902u),
uvec2(0x1a2726ceu,0x3fde148au),
uvec2(0x56b413e5u,0x3fdee2a1u),
uvec2(0x56b413e5u,0x3fdee2a1u),
uvec2(0xaf7a4884u,0x3fdfb358u),
uvec2(0xbe7603adu,0x3fe04360u),
uvec2(0xe2d054fau,0x3fe0ae76u),
uvec2(0xe2d054fau,0x3fe0ae76u),
uvec2(0x23c75aa8u,0x3fe11af8u),
uvec2(0x40f23ca6u,0x3fe188eeu),
uvec2(0x40f23ca6u,0x3fe188eeu),
uvec2(0x5fc61659u,0x3fe1f863u),
uvec2(0x1134db92u,0x3fe26962u),
uvec2(0x1134db92u,0x3fe26962u),
uvec2(0x57b0df43u,0x3fe2dbf5u),
uvec2(0x57b0df43u,0x3fe2dbf5u),
uvec2(0xad9d8c86u,0x3fe35028u),
uvec2(0x0c36bfb5u,0x3fe3c608u),
uvec2(0x0c36bfb5u,0x3fe3c608u),
uvec2(0xf2f923c5u,0x3fe43d9fu),
uvec2(0xf2f923c5u,0x3fe43d9fu),
uvec2(0x6f970c1fu,0x3fe4b6fdu),
uvec2(0x6f970c1fu,0x3fe4b6fdu),
uvec2(0x26867857u,0x3fe5322eu),
uvec2(0x26867857u,0x3fe5322eu),
uvec2(0x5c3649e0u,0x3fe5af40u),
uvec2(0x00000000u,0x00000000u));
const uvec2 llvm_SMALL[7] = uvec2[7](
uvec2(0x00000000u,0xbfe00000u),
uvec2(0x55556aadu,0x3fd55555u),
uvec2(0x0000181au,0xbfd00000u),
uvec2(0x8998124eu,0x3fc99999u),
uvec2(0x452e2a2bu,0xbfc55555u),
uvec2(0x8cde4aa7u,0x3fc24adbu),
uvec2(0xb915ef6fu,0xbfc0019du));
const uvec2 llvm_LARGE[6] = uvec2[6](
uvec2(0xfffffe03u,0xbfdfffffu),
uvec2(0x555395f9u,0x3fd55555u),
uvec2(0x01be5329u,0xbfd00000u),
uvec2(0x1bf8c3afu,0x3fc9999cu),
uvec2(0xba9cee4bu,0xbfc554f0u),
uvec2(0xd56b72d7u,0x3fc1d94cu));
double llvm_ma(double x,double y,double z){precise double product=x*y;precise double result=product+z;return result;}
double llvm_log_eval(double x){
 uvec2 bits=unpackDouble2x32(x);uint index=(bits.y>>14u)&63u;
 double r=packDouble2x32(llvm_R_LOG[index]);int exponent=int((bits.y+0x4000u)>>20u)-1023;
 precise double r_hi=llvm_ma(double(exponent),packDouble2x32(uvec2(0xfefa39efu,0x3fe62e42u)),packDouble2x32(llvm_LOG_R[index]));
 uvec2 mantissa=uvec2(bits.x,(bits.y&0xfffffu)|0x3ff00000u);
 double m=packDouble2x32(mantissa);double c=packDouble2x32(uvec2(0u,mantissa.y&0x3fffc000u));
 precise double delta=m-c;
 precise double dx=llvm_ma(r,delta,packDouble2x32(llvm_C_LOG[index]));
 precise double dx2=dx*dx;
 precise double c1=llvm_ma(dx,packDouble2x32(llvm_LARGE[1]),packDouble2x32(llvm_LARGE[0]));
 precise double c2=llvm_ma(dx,packDouble2x32(llvm_LARGE[3]),packDouble2x32(llvm_LARGE[2]));
 precise double c3=llvm_ma(dx,packDouble2x32(llvm_LARGE[5]),packDouble2x32(llvm_LARGE[4]));
 precise double dx4=dx2*dx2;precise double base=r_hi+dx;
 precise double d1=llvm_ma(dx2,c1,base);precise double d2=llvm_ma(dx2,c3,c2);
 return llvm_ma(dx4,d2,d1);
}
float strata_llvm_log1p(float x,out double wide,out uint path){
 uint b=floatBitsToUint(x);double xd=double(x);path=0u;
 // RNE-only special returns are identical to upstream round_result_slightly_*.
 if(b==0x35400003u){wide=double(uintBitsToFloat(0x353fffffu));path=2u;return float(wide);}
 if(b==0x3710001bu){wide=double(uintBitsToFloat(0x370ffff3u));path=2u;return float(wide);}
 if(b==0x3efd81adu){wide=double(uintBitsToFloat(0x3ecdeee1u));path=2u;return float(wide);}
 if((b&0x7fffffffu)<=0x3c800000u){
  precise double xsq=xd*xd;
  precise double c0=llvm_ma(xd,packDouble2x32(llvm_SMALL[1]),packDouble2x32(llvm_SMALL[0]));
  precise double c1=llvm_ma(xd,packDouble2x32(llvm_SMALL[3]),packDouble2x32(llvm_SMALL[2]));
  precise double c2=llvm_ma(xd,packDouble2x32(llvm_SMALL[5]),packDouble2x32(llvm_SMALL[4]));
  precise double x4=xsq*xsq;
  precise double d0=llvm_ma(xsq,c1,c0);precise double d1=llvm_ma(xsq,packDouble2x32(llvm_SMALL[6]),c2);
  precise double d2=llvm_ma(x4,d1,d0);wide=llvm_ma(xsq,d2,xd);
 }else{precise double sum=xd+1.0LF;wide=llvm_log_eval(sum);path=1u;}
 return float(wide);
}
#endif
