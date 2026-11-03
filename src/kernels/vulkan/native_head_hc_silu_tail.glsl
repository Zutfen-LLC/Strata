#ifndef STRATA_NATIVE_HEAD_HC_SILU_TAIL
#define STRATA_NATIVE_HEAD_HC_SILU_TAIL
// The head-only negative overflow class is admitted only when the actual
// unchanged exp, denominator and division have the native signed-zero form.
bool strata_head_silu_overflow_candidate(uint input_bits) {
    uint magnitude=input_bits&0x7fffffffu;
    return (input_bits&0x80000000u)!=0u && magnitude>=0x42b20000u && magnitude<0x7f800000u;
}
bool strata_head_silu_overflow_valid(uint input_bits,uint exp_bits,uint denominator_bits,uint output_bits) {
    return strata_head_silu_overflow_candidate(input_bits) && exp_bits==0x7f800000u &&
           denominator_bits==0x7f800000u && output_bits==0x80000000u;
}
#endif
