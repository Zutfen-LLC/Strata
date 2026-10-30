#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#ifndef _WIN32
#include <sys/stat.h>
#endif
namespace strata::core {
// Native state is [key][head][value]; GGML state is [head][value][key].
// A diagnostic byte permutation only, never a numerical reconstruction.
inline void layer0_native_state_view(const void* source,void* destination,size_t count) {
    if(!source || !destination || source==destination || count!=786432)
        throw std::runtime_error("layer0 observer: invalid state view");
    auto* dst=static_cast<uint8_t*>(destination);const auto* src=static_cast<const uint8_t*>(source);
    for(size_t head=0;head<48;++head)for(size_t value=0;value<128;++value)for(size_t key=0;key<128;++key)
        std::memcpy(dst+((head*128+value)*128+key)*4,src+((key*48+head)*128+value)*4,4);
}
// One diagnostic image. No computed replacement for an unavailable native value.
class Layer0Observer {
    FILE* file_=nullptr;
    std::array<bool,19> seen_{};
    bool active_=false,finished_=false,reduced_helper_=false;
    size_t bytes_=0;
    void write(const void* p,size_t n) {
        if(n>limit-bytes_ || (n && (!p || fwrite(p,1,n,file_)!=n)))throw std::runtime_error("layer0 observer: bounded write failed");
        bytes_+=n;
    }
public:
    enum Field {embedding,state_before,history_before,pending_residual,moe_input,router_scores,
        router_ids,router_weights,shared_output,expert_rows,combined_output,inject,
        state_after,history_after,primary_ownership,token,materialized_post_residual,helper_ownership,helper_reduction};
    std::array<int32_t,512> primary{},helper{}; // allocated only with this opt-in owner
    static constexpr size_t limit=8u<<20,host_bound=8u<<20;
    static constexpr std::array<size_t,19> counts{{10240,786432,30720,10240,2560,512,10,10,2560,25600,2560,4,786432,30720,512,1,10240,512,1}};
    static bool requested() {const char* p=std::getenv("STRATA_LAYER0_OBSERVER");return p && *p;}
    Layer0Observer() {
        if(!requested())return;
        const char* binding=std::getenv("STRATA_LAYER0_BINDING");
        if(!binding || strlen(binding)!=64 || strspn(binding,"0123456789abcdef")!=64)
            throw std::runtime_error("layer0 observer: SHA256 input/ownership binding required");
        const uint16_t endian=1;
        if(*reinterpret_cast<const uint8_t*>(&endian)!=1)throw std::runtime_error("layer0 observer: little endian required");
        file_=fopen(std::getenv("STRATA_LAYER0_OBSERVER"),"wbx");
        if(!file_)throw std::runtime_error("layer0 observer: exclusive output creation failed");
#ifndef _WIN32
        if(fchmod(fileno(file_),0600)!=0){fclose(file_);file_=nullptr;throw std::runtime_error("layer0 observer: private output failed");}
#endif
        const char magic[8]={'S','T','R','L','0','0','2','\0'};
        try {write(magic,8);write(binding,64);}catch(...){fclose(file_);file_=nullptr;throw;}
    }
    ~Layer0Observer(){if(file_)fclose(file_);}
    Layer0Observer(const Layer0Observer&)=delete;
    Layer0Observer& operator=(const Layer0Observer&)=delete;
    bool active() const{return active_;}
    bool finished() const{return finished_;}
    void begin(int32_t value,bool reduced_helper=false) {
        if(!file_ || finished_)return;
        if(active_)throw std::runtime_error("layer0 observer: repeated begin");
        if(value<0)throw std::runtime_error("layer0 observer: invalid token");
        reduced_helper_=reduced_helper;
        active_=true;put(token,&value,1);
        const int32_t mode=reduced_helper?1:0;put(helper_reduction,&mode,1);
    }
    void put(Field field,const void* data,size_t count) {
        if(!active_)return;
        const size_t id=size_t(field);
        if(id>=counts.size() || seen_[id] || count!=counts[id] || (field==expert_rows && reduced_helper_))throw std::runtime_error("layer0 observer: duplicate/invalid shape");
        const uint32_t header[]{uint32_t(id),uint32_t(count)};write(header,8);write(data,count*4);seen_[id]=true;
    }
    void finish() {
        if(!active_)return;
        for(size_t i=0;i<seen_.size();++i)if(!seen_[i] && i!=state_after && i!=history_after && i!=materialized_post_residual && !(i==expert_rows && reduced_helper_))
            throw std::runtime_error("layer0 observer: required boundary missing");
        for(size_t i=0;i<seen_.size();++i)if(!seen_[i]){const uint32_t h[]{uint32_t(i),0};write(h,8);}
        const uint32_t end[]{UINT32_MAX,0};write(end,8);
        if(fflush(file_)!=0)throw std::runtime_error("layer0 observer: flush failed");
        active_=false;finished_=true;
    }
};
}
