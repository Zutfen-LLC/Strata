#include "strata/worker/packed_activation.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>

namespace strata::worker {
using namespace strata::kernels::cpu;
std::string default_packed_qualification() {
#ifdef STRATA_CPU_PACKED_QUALIFICATION
    return STRATA_CPU_PACKED_QUALIFICATION;
#else
    return {};
#endif
}
bool packed_qualification_valid(const std::string& s) {
    return s.size()==16 && s.find_first_not_of("0123456789abcdef")==std::string::npos &&
           s.find_first_not_of('0')!=std::string::npos;
}
bool packed_format_supported(const NativeFmt& f) {
#if defined(__linux__) && defined(__x86_64__)
    static_assert(CHAR_BIT==8 && sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(block_q8_0)==34 && offsetof(block_q8_0,d)==0 && offsetof(block_q8_0,qs)==2);
    return native_experts_available() && f.gu_type==8 && f.d_type==8 && f.gu_act==8 && f.d_act==8 &&
        f.n_embd==2560 && f.n_ff==640 && f.act_bytes==2720 && f.h_bytes==680 && f.bytes==5222400 &&
        !f.had_block && !f.had_gu && !f.had_d && !f.gu_signs && !f.d_signs && !f.tail_off &&
        ggml_blck_size(GGML_TYPE_Q8_0)==32 && ggml_type_size(GGML_TYPE_Q8_0)==34;
#else
    (void)f;return false;
#endif
}
std::shared_ptr<const PackedActivation> pack_activation(const NativeFmt& f,const float* x,
    const uint8_t* active,int T,std::string& reason,uint64_t* quantized_rows) {
    if(quantized_rows)*quantized_rows=0;
    if(!packed_format_supported(f)){reason="unsupported_format_or_build";return {};}
    if(T<1 || T>2){reason="unsupported_token_count";return {};}
    if(!x || !active){reason="invalid_input";return {};}
    for(int i=0;i<T*2560;++i)if(!std::isfinite(x[i])){reason="nonfinite_input";return {};}
    auto a=std::make_shared<PackedActivation>();a->T=(uint8_t)T;
    a->active.assign(active,active+T);a->fields.assign((size_t)T*kPackedRowBytes,0);
    std::vector<uint8_t> scratch(kNativeActBytes,0);
    for(int t=0;t<T;++t) {
        if(active[t]>1){reason="invalid_mask";return {};}
        if(!active[t])continue;
        std::fill(scratch.begin(),scratch.end(),0);
        native_quant_act(f,x+(size_t)t*2560,scratch.data());
        if(quantized_rows)++*quantized_rows;
        auto* row=a->fields.data()+(size_t)t*kPackedRowBytes;
        for(size_t b=0;b<80;++b) {
            block_q8_0 block;std::memcpy(&block,scratch.data()+b*sizeof(block),sizeof(block));
            if((block.d&0x7c00)==0x7c00){reason="unsupported_scale";return {};}
            row[b*34]=(uint8_t)block.d;row[b*34+1]=(uint8_t)(block.d>>8);
            std::memcpy(row+b*34+2,block.qs,32);
        }
    }
    reason.clear();return a;
}
bool restore_activation(const NativeFmt& f,const PackedActivation& a,std::vector<uint8_t>& scratch) {
    if(!packed_format_supported(f) || !a.T || a.T>2 || a.active.size()!=a.T ||
       a.fields.size()!=(size_t)a.T*kPackedRowBytes)return false;
    scratch.assign((size_t)a.T*kNativeActBytes,0);
    for(size_t t=0;t<a.T;++t) {
        if(a.active[t]>1)return false;
        if(!a.active[t])continue;
        const auto* row=a.fields.data()+t*kPackedRowBytes;
        for(size_t b=0;b<80;++b) {
            block_q8_0 block{};block.d=(uint16_t)row[b*34]|(uint16_t)row[b*34+1]<<8;
            if((block.d&0x7c00)==0x7c00)return false;
            std::memcpy(block.qs,row+b*34+2,32);
            std::memcpy(scratch.data()+t*kNativeActBytes+b*sizeof(block),&block,sizeof(block));
        }
    }
    return true;
}
}
