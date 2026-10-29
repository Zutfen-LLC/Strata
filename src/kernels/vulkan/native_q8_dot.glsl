// Shared native Fmt<8>/row_dot machinery for packed gate/up and down.
shared float partial[256];
uint weight_word(uint byte_offset) {
    uint at=p.wb+byte_offset/4,shift=(byte_offset%4)*8;
    if(shift==0)return w[at];
    return (w[at]>>shift)|(w[at+1]<<(32-shift));
}
float weight_scale(uint byte_offset) {
    return unpackHalf2x16(w[p.wb+byte_offset/4]>>((byte_offset%4)*8)).x;
}
float q8_row_dot(uint base,uint entry,uint nb,uint tid) {
    uint lane=tid%32;
    float sum=0.0;
    for(uint k=lane;k<nb*4;k+=32) {
        uint block=k/4,iqs=2*(k%4),wx=base+block*34+2+4*iqs;
        uint xx=p.xb+(entry*nb+block)*9;
        int dot=dotPacked4x8EXT(int(weight_word(wx)),int(x[xx+1+iqs]));
        dot+=dotPacked4x8EXT(int(weight_word(wx+4)),int(x[xx+2+iqs]));
        // sm_86: FMUL(d0,d1), I2FP(sumi), FFMA(scale,sumi,sum).
        precise float scale=weight_scale(base+block*34)*unpackHalf2x16(x[xx]).x;
        sum=fma(scale,float(dot),sum);
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
