#include "native_q8_replay_timing.hpp"
#include <iostream>
#include <sstream>
#include "../../src/core/native_q8_subgroup.hpp"
using namespace strata::worker;
using namespace strata::worker::vulkan_detail;
struct FakeClock {
    using duration=std::chrono::milliseconds;
    using time_point=std::chrono::time_point<FakeClock,duration>;
    inline static int64_t ms=0;
    static time_point now(){return time_point(duration(ms));}
};
int main(){try {
    {
        std::array<NativeQ8StageTiming,16> samples{};
        for(size_t i=0;i<16;++i)for(size_t j=0;j<7;++j)samples[i].milliseconds[j]=double(i+j);
        std::ostringstream out;print_native_q8_stage_timing(out,samples);
        require(out.str().find("stage=input_upload samples_ms=0,1,2,3")!=std::string::npos &&
                out.str().find("stage=row_restore samples_ms=6,7,8,9")!=std::string::npos &&
                out.str().find("includes_waits=true gpu_kernel_only=false")!=std::string::npos,"stage timing format/accounting");
        samples[2].milliseconds[3]=-1;bool rejected=false;
        try{print_native_q8_stage_timing(out,samples);}catch(const std::exception&){rejected=true;}
        require(rejected,"invalid stage duration accepted");
    }
    {
        using strata::core::portable_detail::native_q8_subgroup_supported;
        for(unsigned size:{16u,32u,64u,128u})for(int flags=0;flags<8;++flags)
            require(native_q8_subgroup_supported(flags&1,flags&2,flags&4,size)==(flags==7 && (size==32 || size==64)),"subgroup capability admission");
        for(unsigned width:{32u,64u}) {
            std::array<float,256> shared{},shuffled{};
            for(size_t i=0;i<256;++i)shared[i]=shuffled[i]=(i%4==0?1.0e20f:i%4==1?1.0f:i%4==2?-1.0e20f:float(i%31));
            for(unsigned offset=16;offset;offset>>=1) {
                auto old=shared,prev=shuffled;
                for(unsigned tid=0;tid<256;++tid) {
                    shared[tid]=offset==16?old[tid^offset]+old[tid]:old[tid]+old[tid^offset];
                    const unsigned subgroup=tid/width,lane=tid%width,partner=subgroup*width+(lane^offset);
                    shuffled[tid]=offset==16?prev.at(partner)+prev[tid]:prev[tid]+prev.at(partner);
                }
            }
            require(!std::memcmp(shared.data(),shuffled.data(),sizeof(shared)),"subgroup logical XOR tree mismatch");
        }
    }
    GpuCounters baseline;baseline.gpu_allocated_bytes=1024;baseline.gpu_weight_bytes=512;
    baseline.gpu_scratch_bytes=512;baseline.gpu_allocation_count=4;
    std::string error;
    for(int failure=0;failure<5;++failure) {
        FakeClock::ms=0;size_t calls=0;bool stopped=false;
        auto infer=[&](EntryRowsReply& r){
            ++calls;r.session=1;r.sequence=calls;r.indices={0};r.rows.assign(H,1.f);
            // First warmup simulates capture I/O; snapshots simulate bookkeeping.
            FakeClock::ms+=calls==1?5000:calls==2?10:int64_t(calls-2);
            if(failure==1 && calls==5)r.rows[0]=2.f;
            if(failure==2 && calls==5)r.sequence++;
            if(failure==3 && calls==1)FakeClock::ms=60000;
            return failure!=4 || calls!=5;
        };
        auto snapshot=[&]{FakeClock::ms+=100;return baseline;};
        NativeQ8ReplayTiming result;
        try {result=native_q8_time_replay<FakeClock>(infer,snapshot,FakeClock::time_point(FakeClock::duration(60000)),error);}
        catch(const std::runtime_error&) {stopped=true;}
        require(stopped==(failure!=0),"timing failure handling");
        if(failure==0){
            require(calls==18 && result.minimum_ms==1 && result.median_ms==8.5 && result.maximum_ms==16,
                "warmups or out-of-scope I/O entered timed samples");
            for(size_t i=0;i<16;++i)require(result.samples_ms[i]==double(i+1),"timed sample order/count");
        }
    }
    for(int changed=0;changed<4;++changed) {
        FakeClock::ms=0;size_t calls=0;bool stopped=false;
        auto infer=[&](EntryRowsReply& r){++calls;r.session=1;r.sequence=calls;r.indices={0};r.rows.assign(H,1.f);return true;};
        auto snapshot=[&]{auto c=baseline;if(calls==3){
            if(changed==0)++c.gpu_allocated_bytes;
            if(changed==1)++c.gpu_weight_bytes;
            if(changed==2)++c.gpu_scratch_bytes;
            if(changed==3)++c.gpu_allocation_count;
        }return c;};
        try{native_q8_time_replay<FakeClock>(infer,snapshot,FakeClock::time_point(FakeClock::duration(60000)),error);}catch(const std::runtime_error&){stopped=true;}
        require(stopped,"owned allocation growth/change was accepted");
    }
    {
        GpuExecutionTrace trace;EntryRowsReply reply;std::vector<float> packed(80*H);
        for(int i=0;i<80;++i)trace.original.push_back(i<40?i*2:(i-40)*2+1);
        reply.rows.resize(80*H);
        for(int i=0;i<80;++i){reply.indices.push_back(i);std::fill_n(packed.begin()+i*H,H,float(i+1));
            std::fill_n(reply.rows.begin()+trace.original[i]*H,H,float(i+1));}
        native_q8_check_restored_rows(reply,trace,packed);
        reply.rows.back()=-1;bool rejected=false;
        try{native_q8_check_restored_rows(reply,trace,packed);}catch(const std::exception&){rejected=true;}
        require(rejected,"wrong restored row contents accepted");
    }
    std::cout<<"fixed warmups/calls, timing scope, exact repeats, deadline, failures and fresh allocation counters checked; no GPU execution\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
