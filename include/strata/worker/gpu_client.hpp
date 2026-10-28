#pragma once
#include "strata/worker/gpu_wire.hpp"
#include <functional>

namespace strata::worker {
// Fixed opt-in GPU tier: CUDA default, Vulkan hybrid loopback only. A failure latches it dead. The engine
// must abort the affected request and may start a new CPU-only session explicitly.
class GpuWorkerTier {
public:
    struct Open {
        std::string host="127.0.0.1";
        uint16_t port=7432;
        int deadline_ms=10000,setup_ms=600000;
        GpuPlan plan;
        bool vulkan=false; // opt-in hybrid contract, requires 127.0.0.1 and indexed rows
        bool entry_rows=false; // negotiated; unsupported peers fail without weighted fallback
        std::function<bool(int64_t,int64_t,uint8_t*,uint32_t)> fetch;
        std::function<bool()> cancelled;
    };
    GpuWorkerTier()=default;
    ~GpuWorkerTier();
    GpuWorkerTier(const GpuWorkerTier&)=delete;
    GpuWorkerTier& operator=(const GpuWorkerTier&)=delete;
    bool open(const Open&,std::string&);
    void close();
    bool holds(int64_t layer,int32_t expert)const{return plan_.owns((int)layer,expert);}
    bool ready()const{return ready_;}
    bool dead()const{return dead_;}
    const std::string& dead_why()const{return why_;}
    bool post(int64_t,const float*,const int32_t*,const float*,const uint8_t*,int,int,std::string&);
    bool finish(std::string&);
    bool posted()const{return posted_;}
    bool served()const{return served_;}
    const float* partials()const{return reply_.partials.data();}
    size_t partial_count()const{return reply_.partials.size();}
    const EntryRowsReply& entry_rows() const { return rows_reply_; }
    bool status(GpuCounters&,std::string&);
    void report(std::string&)const;
    struct Stats {
        uint64_t session=0,assigned_bytes=0,setup_network_bytes=0,cache_reused_bytes=0;
        uint64_t host_staging_peak=0,requests=0,entries=0,failures=0;
        double setup_ms=0,request_ms=0;
    };
    const Stats& stats()const{return stats_;}
private:
    bool exchange(uint16_t,const std::vector<uint8_t>&,uint16_t,WireFrame&,std::string&);
    bool fail(const std::string&,std::string&);
    uintptr_t fd_=~(uintptr_t)0;
    int deadline_ms_=10000;
    uint64_t session_=0,sequence_=0,setup_end_=0;
    bool ready_=false,dead_=false,posted_=false,served_=false;
    std::function<bool()> cancelled_;
    std::string why_;
    GpuPlan plan_;
    InferReply reply_;
    EntryRowsReply rows_reply_;
    bool rows_=false,vulkan_=false;
    Stats stats_;
    GpuCounters remote_;
};
} // namespace strata::worker
