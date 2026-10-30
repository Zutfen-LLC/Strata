#ifndef STRATA_NATIVE_F32_DIV_RN
#define STRATA_NATIVE_F32_DIV_RN
// Integer significand division with guard/round/sticky and nearest-even packing.
// Prior art: Berkeley SoftFloat 3e f32_div's direct quotient/remainder path and
// compiler-rt fp_div_impl.inc's IEEE classification/normalization. See
// third_party/softfloat/NOTICE. Restoring division keeps all operations uint32;
// no shaderFloat64, shaderInt64, floating reciprocal, FMA or division is needed.
// NaNs: quiet and preserve the first NaN operand; invalid 0/0 and inf/inf use
// 0x7fc00000. IEEE value classes are preserved; native NaN payload equivalence
// and floating exception flags are not claimed by this diagnostic.
uint strata_f32_div_rn_bits(uint a,uint b) {
    uint sign=(a^b)&0x80000000u,aa=a&0x7fffffffu,bb=b&0x7fffffffu;
    if(aa>0x7f800000u)return a|0x00400000u;
    if(bb>0x7f800000u)return b|0x00400000u;
    if(aa==0x7f800000u)return bb==0x7f800000u?0x7fc00000u:sign|0x7f800000u;
    if(bb==0x7f800000u)return sign;
    if(aa==0u)return bb==0u?0x7fc00000u:sign;
    if(bb==0u)return sign|0x7f800000u;
    int ea=int(aa>>23)-127,eb=int(bb>>23)-127;
    uint sa=aa&0x007fffffu,sb=bb&0x007fffffu;
    if((aa>>23)==0u){ea=-126;while((sa&0x00800000u)==0u){sa<<=1;--ea;}}
    else sa|=0x00800000u;
    if((bb>>23)==0u){eb=-126;while((sb&0x00800000u)==0u){sb<<=1;--eb;}}
    else sb|=0x00800000u;
    int exponent=ea-eb;
    if(sa<sb){sa<<=1;--exponent;}
    if(exponent>127)return sign|0x7f800000u;
    if(exponent< -150)return sign;
    // Exact 31-bit normalized quotient plus nonzero remainder (sticky). Each
    // remainder is <2*sb<2^25, so no intermediate needs more than 32 bits.
    uint quotient=0u,remainder=sa;
    for(int bit=0;bit<31;++bit){
        quotient<<=1;
        if(remainder>=sb){remainder-=sb;quotient|=1u;}
        if(bit!=30)remainder<<=1;
    }
    uint shift=7u;
    if(exponent< -126)shift+=uint(-126-exponent); // at most 31; includes underflow ties
    uint significand=quotient>>shift;
    uint guard=(quotient>>(shift-1u))&1u;
    bool sticky=(quotient&((1u<<(shift-1u))-1u))!=0u || remainder!=0u;
    if(guard!=0u && (sticky || (significand&1u)!=0u))++significand;
    if(exponent< -126)return sign|significand; // may round up to minimum normal
    if(significand==0x01000000u){significand>>=1;++exponent;}
    if(exponent>127)return sign|0x7f800000u;
    return sign|(uint(exponent+127)<<23)|(significand&0x007fffffu);
}
#endif
