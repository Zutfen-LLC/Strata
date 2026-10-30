#pragma once
#include "native_q8_fixture.hpp"
namespace native_q8_fixture::consumer {
constexpr size_t ELEMENTS=832, H=256, F=32, BLOCK_BYTES=36, PREFIX_BYTES=8*BLOCK_BYTES;
// Same fixed paired fixture as native_expert_grouped: hq follows three
// 256-byte-aligned F32 planes. Capture only the live blocks, not padding.
constexpr size_t ENTRIES=2, FLOAT_PLANE_BYTES=(ENTRIES*F*4+255)&~size_t(255);
constexpr size_t HQ_OFFSET=3*FLOAT_PLANE_BYTES, HQ_BYTES=ENTRIES*(F/32)*BLOCK_BYTES;
static_assert(FLOAT_PLANE_BYTES==256 && HQ_OFFSET==768 && HQ_BYTES==72);
inline size_t checked_hq_offset(size_t scratch_bytes) {
    need(HQ_OFFSET<=scratch_bytes && HQ_BYTES<=scratch_bytes-HQ_OFFSET,"consumer: hq exceeds native scratch");
    return HQ_OFFSET;
}
inline void write_hq(const std::string& path,const Bytes& hq) {
    need(hq.size()==HQ_BYTES,"consumer: hq sidecar size mismatch");
    std::ofstream out(path,std::ios::binary);need(bool(out),"consumer: cannot create hq sidecar");
    out.write(reinterpret_cast<const char*>(hq.data()),hq.size());need(bool(out),"consumer: hq sidecar write failed");
}
inline Result read_capture(const std::string& path,bool observation) {
    std::ifstream in(path,std::ios::binary);need(bool(in),"consumer: cannot read capture");
    need(word(in)==(observation?0x3252384e:0x3152384e),"consumer: wrong capture role/magic");
    if(observation)need(word(in)==1,"consumer: observation sentinel status failed");
    auto r=blank(ELEMENTS);
    for(auto* b:{&r.input,&r.hidden,&r.separate,&r.fused,&r.empty}){
        need(word(in)==b->size(),"consumer: incorrect section size/order");
        in.read(reinterpret_cast<char*>(b->data()),b->size());need(bool(in),"consumer: truncated section");}
    need(in.peek()==EOF,"consumer: trailing capture data");return r;
}
inline void validate_pair(const Result& native,const Result& observation) {
    const auto shape=blank(ELEMENTS);
    for(const auto* r:{&native,&observation}) {
        need(r->input.size()==shape.input.size() && r->hidden.size()==shape.hidden.size() &&
            r->separate.size()==shape.separate.size() && r->fused.size()==shape.fused.size() &&
            r->empty.size()==shape.empty.size(),"consumer: capture shape mismatch");
        for(uint8_t x:r->empty)need(x==0xa5,"consumer: empty-interval sentinel failed");
        for(size_t i=0;i<r->fused.size();++i)if(i<4*36 || i>=18*36)
            need(r->fused[i]==0xa5,"consumer: inactive-region sentinel failed");
    }
    for(size_t i=0;i<native.input.size();++i){
        if(i>=184 && i<216)need(native.input[i]==127 && observation.input[i]==0,"consumer: expected block5 code difference absent");
        else need(native.input[i]==observation.input[i],"consumer: unexpected input-Q8 difference");}
    for(size_t i=180;i<184;++i)need(native.input[i]==0 && observation.input[i]==0,"consumer: block5 ds must be +0/+0");
    need(native.hidden==observation.hidden && native.separate==observation.separate &&
        native.fused==observation.fused && native.empty==observation.empty,"consumer: unexpected other-stage difference");
    need(std::equal(native.separate.begin()+4*36,native.separate.begin()+18*36,native.fused.begin()+4*36),
        "consumer: active native separate/fused mismatch");
    bool signal=false;
    for(size_t b=0;b<8;++b){const size_t base=b*36;
        uint16_t scale=uint16_t(native.input[base])|(uint16_t(native.input[base+1])<<8);
        need((scale&0x7c00)!=0x7c00,"consumer: nonfinite prefix activation scale");
        if(b!=5 && (scale&0x7fff))for(size_t i=4;i<36;++i)signal|=native.input[base+i]!=0;
    }
    need(signal,"consumer: common nonzero signal blocks required");
}
// A single finite synthetic Q8_0 expert. No model weights or dequantization.
// The signed rows and zero-scale rows exercise output signs; positive rows keep
// the common nonzero input signal from degenerating into an all-zero test.
inline Bytes weights(size_t h=H,size_t f=F) {
    need((h==H && f==F) || (h==2560 && f==640),"consumer: unsupported synthetic geometry");
    Bytes out((2*f*h/32+h*f/32)*34);
    size_t at=0;
    for(int matrix=0;matrix<3;++matrix){const size_t rows=matrix==2?h:f,columns=matrix==2?f:h;
        for(size_t row=0;row<rows;++row)for(size_t block=0;block<columns/32;++block){
            const uint16_t scale=row%4==2?0x8000:0x1000; // -0 or +2^-11, both finite half
            out[at]=uint8_t(scale);out[at+1]=uint8_t(scale>>8);
            for(size_t i=0;i<32;++i){const int q=row%4==1?-1:row%4==3?(i%2?-1:1):1;
                out[at+2+i]=uint8_t(q);}
            at+=34;}}
    need(at==out.size() && out.size()==(h==H?26112:5222400),"consumer: weight layout mismatch");
    for(size_t i=0;i<out.size();i+=34)need(((uint16_t(out[i])|(uint16_t(out[i+1])<<8))&0x7c00)!=0x7c00,
        "consumer: nonfinite weight scale");
    return out;
}
}
