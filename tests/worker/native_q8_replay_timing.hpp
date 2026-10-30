#pragma once
#include "vulkan_native_q8_layout.hpp"
#include "../../src/core/portable_expert_timer.hpp"
#include <array>
#include <chrono>
#include <cstring>
namespace strata::worker::vulkan_detail {
inline bool native_q8_same_allocations(const GpuCounters& a,const GpuCounters& b) {
    return a.gpu_allocated_bytes==b.gpu_allocated_bytes && a.gpu_weight_bytes==b.gpu_weight_bytes &&
        a.gpu_scratch_bytes==b.gpu_scratch_bytes && a.gpu_allocation_count==b.gpu_allocation_count;
}
// The captured down tensor is in packed group order; the public reply must
// restore the original request indices AND their complete row contents.
inline void native_q8_check_restored_rows(const EntryRowsReply& reply,const GpuExecutionTrace& trace,
    const std::vector<float>& packed) {
    require(trace.original.size()==reply.indices.size() && packed.size()==reply.rows.size() &&
        packed.size()==trace.original.size()*H,"native Q8 restored row shape");
    for(size_t row=0;row<trace.original.size();++row) {
        const auto it=std::lower_bound(reply.indices.begin(),reply.indices.end(),trace.original[row]);
        require(it!=reply.indices.end() && *it==trace.original[row],"native Q8 restored row index absent");
        const size_t restored=it-reply.indices.begin();
        require(!std::memcmp(packed.data()+row*H,reply.rows.data()+restored*H,H*sizeof(float)),
            "native Q8 restored row contents differ from captured down output");
    }
}
struct NativeQ8ReplayTiming {
    std::array<double,16> samples_ms{};
    double minimum_ms=0,median_ms=0,maximum_ms=0;
    GpuCounters allocations;
    EntryRowsReply first_reply;
};
// Fixed bounded observation only. Existing executor owns synchronization.
// Capture-once happens during warmup 0; reply/counter checks are outside timing.
template<class Clock=std::chrono::steady_clock,class Infer,class Snapshot>
NativeQ8ReplayTiming native_q8_time_replay(Infer infer,Snapshot snapshot,
    typename Clock::time_point deadline,std::string& error) {
    NativeQ8ReplayTiming result;result.allocations=snapshot();
    const auto check_deadline=[&]{require(Clock::now()<deadline,"native Q8 replay exceeded 60-second overall deadline");};
    for(size_t call=0;call<18;++call) {
        check_deadline();EntryRowsReply reply;bool ok;
        if(call<2)ok=infer(reply);
        else {
            uint64_t calls=0;
            strata::core::portable_detail::ScopedExpertTimer<Clock> timer(true,result.samples_ms[call-2],calls);
            ok=infer(reply);
        }
        require(ok,error.empty()?"native Q8 timing inference failed":error);check_deadline();
        require(native_q8_same_allocations(result.allocations,snapshot()),"native Q8 owned allocation counters changed");
        if(call==0){
            require(reply.session && reply.sequence && reply.sequence<=UINT64_MAX-17 && !reply.indices.empty() && reply.rows.size()==reply.indices.size()*H,"native Q8 timing reply shape");
            result.first_reply=std::move(reply);
        } else {
            const auto& first=result.first_reply;
            require(reply.session==first.session && reply.sequence==first.sequence+call && reply.indices==first.indices &&
                reply.rows.size()==first.rows.size() && !std::memcmp(reply.rows.data(),first.rows.data(),reply.rows.size()*sizeof(float)),
                "native Q8 timing changed repeated-request identity or output bits");
        }
    }
    auto sorted=result.samples_ms;
    for(double v:sorted)require(std::isfinite(v) && v>=0,"native Q8 timing invalid sample");
    std::sort(sorted.begin(),sorted.end());result.minimum_ms=sorted.front();result.maximum_ms=sorted.back();
    result.median_ms=(sorted[7]+sorted[8])/2;check_deadline();return result;
}
}
