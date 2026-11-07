// Shared by the decode dot and the baseline product-input gate.
 if(is_beta){float e=exp(-a);precise float den=1.0+e;precise float b=1.0/den;
  bool positive=a>uintBitsToFloat(STRATA_HC_MIX_ACTIVATION_LIMIT_BITS),negative=a<=-89.0;
  if(positive){if(floatBitsToUint(e)>0x33800000u||den!=1.0||floatBitsToUint(b)!=0x3f800000u)atomicOr(status[p.off[8]],2u);}
  else if(negative){if(floatBitsToUint(e)!=0x7f800000u||floatBitsToUint(den)!=0x7f800000u||floatBitsToUint(b)!=0u)atomicOr(status[p.off[8]],2u);}
  else{if((floatBitsToUint(a)&0x7fffffffu)>STRATA_HC_MIX_ACTIVATION_LIMIT_BITS)atomicOr(status[p.off[8]],2u);ck(e);ck(den);ck(b);if(!(e>0.0)||b==0.0||den>uintBitsToFloat(STRATA_HC_MIX_MAX_DENOMINATOR_BITS))atomicOr(status[p.off[8]],2u);}
  beta[p.off[6]+token*48u+r]=b;diag[p.off[7]+token*336u+48u+r]=a;diag[p.off[7]+token*336u+240u+r]=e;diag[p.off[7]+token*336u+288u+r]=den;
 }else{precise float v=ck(a+dt[p.off[3]+r]);float e=0.0,sp;
  if(v>20.0)sp=v;
  else{e=exp(v);uint eb=floatBitsToUint(e);if(eb==0u||(eb>=0x800000u&&eb<=0x4e000000u)){double wide;uint kind;sp=strata_llvm_log1p(e,wide,kind);}else{atomicOr(status[p.off[8]],4u);sp=uintBitsToFloat(0x7fc00000u);}}
  precise float g=ck(ck(sp)*ssa[p.off[4]+r]);if(sp!=0.0&&ssa[p.off[4]+r]!=0.0&&g==0.0)atomicOr(status[p.off[8]],8u);gate[p.off[5]+token*48u+r]=g;diag[p.off[7]+token*336u+r]=a;diag[p.off[7]+token*336u+96u+r]=v;diag[p.off[7]+token*336u+144u+r]=e;diag[p.off[7]+token*336u+192u+r]=sp;
 }
