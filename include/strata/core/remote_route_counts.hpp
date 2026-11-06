#pragma once
// Opt-in first-serve-request aggregate. No activations, tokens or weights.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace strata::core {
inline bool remote_route_counts_supported(bool enabled,bool serve,bool remote_cpu,int batch,
                                          bool vision,bool first_trace,bool cpu_capture) {
    return !enabled || (serve && remote_cpu && batch==0 && !vision && !first_trace && !cpu_capture);
}
class RemoteRouteCounts {
    static constexpr size_t max_patterns_=65536;
    std::FILE* file_=nullptr;
    bool attempted_=false;
    std::atomic<bool> active_{false},complete_{true};
    std::mutex mutex_;
    uint64_t calls_=0;
    int32_t layers_=0,experts_=0;
    std::map<std::vector<int32_t>,uint64_t> patterns_;
public:
    ~RemoteRouteCounts() { if(file_) std::fclose(file_); }
    void begin(int64_t layers,int64_t experts) {
        if(attempted_) return;
        attempted_=true;
        const char* path=std::getenv("STRATA_REMOTE_ROUTE_COUNTS");
        if(!path || !*path) return;
        if(layers<1 || layers>1024 || experts<1 || experts>65536) {std::fprintf(stderr,"remote route counts: unsupported geometry\n");return;}
        layers_=int32_t(layers);experts_=int32_t(experts);
        file_=std::fopen(path,"wbx");
#ifndef _WIN32
        if(file_ && fchmod(fileno(file_),0600)!=0) {std::fclose(file_);file_=nullptr;}
#endif
        if(!file_) std::fprintf(stderr,"remote route counts: cannot create output\n");
        active_.store(file_!=nullptr);
    }
    void record(int64_t layer,const int32_t* ids,int64_t T,int64_t K) noexcept {
        if(!active_.load())return;
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if(!file_ || !complete_.load())return;
            if(!ids || layer<0 || layer>=layers_ || T<1 || T>8 || K<1 || K>10 || calls_==UINT64_MAX) {complete_=false;return;}
            std::vector<int32_t> key;key.reserve(size_t(T*K)+3);
            key.push_back(int32_t(layer));key.push_back(int32_t(T));key.push_back(int32_t(K));
            for(int64_t i=0;i<T*K;++i) {if(ids[i]<0 || ids[i]>=experts_){complete_=false;return;}key.push_back(ids[i]);}
            auto it=patterns_.find(key);
            if(it==patterns_.end()) {
                if(patterns_.size()>=max_patterns_){complete_=false;return;}
                it=patterns_.emplace(std::move(key),0).first;
            }
            if(it->second==UINT64_MAX){complete_=false;return;}
            ++it->second;++calls_;
        } catch(...) {complete_.store(false);}
    }
    void finish(int64_t generated,const char* reason) noexcept {
        if(!active_.exchange(false))return;
        std::lock_guard<std::mutex> lock(mutex_);
        bool ok=std::fprintf(file_,"STRATA_REMOTE_ROUTE_COUNTS_V1 complete=%d calls=%llu generated=%lld finish=%s patterns=%zu layers=%d experts=%d\n",
            complete_.load()?1:0,(unsigned long long)calls_,(long long)generated,reason?reason:"unknown",patterns_.size(),layers_,experts_)>=0;
        for(const auto& [key,count]:patterns_) {
            ok=std::fprintf(file_,"%d %d %d %llu",key[0],key[1],key[2],(unsigned long long)count)>=0 && ok;
            for(size_t i=3;i<key.size();++i)ok=std::fprintf(file_," %d",key[i])>=0 && ok;
            ok=std::fputc('\n',file_)!=EOF && ok;
        }
        ok=std::fclose(file_)==0 && ok;file_=nullptr;patterns_.clear();
        if(!ok || !complete_.load())std::fprintf(stderr,"remote route counts: incomplete output\n");
    }
};
} // namespace strata::core
