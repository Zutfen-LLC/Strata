// Shared native Fmt<8>/row_dot machinery for packed gate/up and down.
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
#extension GL_EXT_spirv_intrinsics : require
// SPV_KHR_fma: FMAKHR=6030, OpFmaKHR=4427. Compiled and validated,
// never inserted into a binary by the host.
spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
float native_q8_strict_fma(float a,float b,float c);
#endif
layout(buffer_reference,std430,buffer_reference_align=4) readonly buffer WeightWords {uint w[];};
shared float partial[256];
uint weight_word(uint64_t byte_offset) {
    WeightWords words=WeightWords(byte_offset & ~uint64_t(3));
    uint shift=uint(byte_offset%4)*8;
    if(shift==0)return words.w[0];
    return (words.w[0]>>shift)|(words.w[1]<<(32-shift));
}
float weight_scale(uint64_t byte_offset) {
    WeightWords words=WeightWords(byte_offset & ~uint64_t(3));
    return unpackHalf2x16(words.w[0]>>(uint(byte_offset%4)*8)).x;
}
float q8_row_dot(uint64_t base,uint entry,uint nb,uint tid) {
    uint lane=tid%32;
    float sum=0.0;
    for(uint k=lane;k<nb*4;k+=32) {
        uint block=k/4,iqs=2*(k%4);
        uint64_t wx=base+uint64_t(block)*34+2+4*iqs;
        uint xx=p.xb+(entry*nb+block)*9;
        int dot=dotPacked4x8EXT(int(weight_word(wx)),int(x[xx+1+iqs]));
        dot+=dotPacked4x8EXT(int(weight_word(wx+4)),int(x[xx+2+iqs]));
        // sm_86: FMUL(d0,d1), I2FP(sumi), FFMA(scale,sumi,sum).
        precise float scale=weight_scale(base+block*34)*unpackHalf2x16(x[xx]).x;
#ifdef STRATA_NATIVE_Q8_STRICT_FMA
        sum=native_q8_strict_fma(scale,float(dot),sum);
#else
        sum=fma(scale,float(dot),sum);
#endif
    }
    partial[tid]=sum;barrier();
    for(uint offset=16;offset>0;offset>>=1) {
        // The first native FADD lists the shuffled operand first.
        precise float next=offset==16 ? partial[(tid^offset)]+partial[tid]
                                      : partial[tid]+partial[(tid^offset)];
        barrier();partial[tid]=next;barrier();
    }
    return partial[tid];
}
