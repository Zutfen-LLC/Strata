#pragma once
// Saved-input group-wrapper discriminator. Never owns a worker or loads an arena.
#include "strata/worker/cpu_capture.hpp"
#include "strata/worker/cpu_group.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace strata::worker::diagnostic {
inline bool unchanged_resident(const WorkerCacheInfo& a,const WorkerCacheInfo& b){
    return b.complete && !b.session && b.idle_remaining_ms==kResidencyNoExpiry &&
        a.resident_bytes==b.resident_bytes && a.loaded_bytes==b.loaded_bytes &&
        a.verified_bytes==b.verified_bytes && a.arena_allocations==b.arena_allocations &&
        a.peak_arena_bytes==b.peak_arena_bytes && a.expirations==b.expirations && a.unloads==b.unloads;
}
struct SingleGroupReport {
    uint32_t checked=0,active=0,zero=0;
    bool cache_hit=false,resident_unchanged=false;
};
inline bool replay_single_group(const CpuCaptureFile& f,const LocalManifest& local,const char* host,uint16_t port,
    const WorkerCacheInfo& baseline,const CpuReplayReport& validated,SingleGroupReport& report,std::string& err){
    report={};
    if(f.records.empty() || validated.full_checked!=f.records.size() || validated.grouped_checked!=f.records.size() ||
       !validated.entry_order_bits || !validated.peer_bits || !validated.detail_envelope){
        err="single group requires completed full-reference validation";return false;
    }
    ExpertWorkerClient check;WorkerCacheInfo before,after;
    if(!baseline.complete || baseline.session || baseline.idle_remaining_ms!=kResidencyNoExpiry ||
       !local.matches(f.setup) || local.identity!=f.identity || local.index_digest!=f.index ||
       !check.connect_local(host,port,5000,{},err) || !check.cache_status(before,err) ||
       !unchanged_resident(baseline,before)){
        err="single group idle resident preflight refused: "+err;return false;
    }
    check.close(); // the server admits one session; never overlap status and group connections
    ExpertWorkerGroup group;bool ok=false;
    try {
        const auto start=std::chrono::steady_clock::now();
        auto expired=[&]{return std::chrono::steady_clock::now()-start>std::chrono::seconds(180);};
        ExpertWorkerGroup::Open option;option.host=host;option.port=port;option.local=&local;
        option.resident=true;option.resident_reuse_only=true;option.local_fallback=false;
        option.setup_ms=10000;option.deadline_ms=10000;option.cancelled=expired;
        option.model_name=f.setup.model_name;option.gu_type=f.setup.gu_type;option.d_type=f.setup.d_type;
        option.n_layers=f.setup.n_layers;option.n_expert=f.setup.n_expert;
        option.n_embd=f.setup.n_embd;option.n_ff=f.setup.n_ff;option.partition=f.setup.partition;
        ok=group.open({option},f.setup.partition,err);
        if(ok){
            const auto& stats=group.peer_stats(0);
            report.cache_hit=stats.cache_hit && stats.reused_bytes==baseline.resident_bytes &&
                stats.setup_blobs==f.setup.owned_count() && stats.setup_weight_bytes==baseline.resident_bytes &&
                !stats.setup_network_bytes && !stats.local_bytes && !stats.local_hits && !stats.local_misses;
            if(!report.cache_hit){err="single group requires exact reuse without uploads";ok=false;}
        }
        for(size_t i=0;ok && i<f.records.size();++i){
            if(expired()){err="single group total deadline";ok=false;break;}
            const auto& r=f.records[i];const auto& q=r.request;
            std::vector<int32_t> ids(q.ids.begin(),q.ids.end());
            if(!group.post(q.layer,q.activations.data(),ids.data(),q.weights.data(),q.mask.data(),q.T,q.K,err) ||
               !group.finish(err)){ok=false;break;}
            const size_t count=(size_t)q.T*f.setup.n_embd;
            // Zero-work windows expose no partial pointer; compare an explicit positive-zero row.
            std::vector<float> zero;
            if(!group.served())zero.assign(count,0.f);
            const float* got=group.served()?group.partials():zero.data();
            if(group.served()!=r.sent || r.partials.size()!=count || !got ||
               !std::all_of(got,got+count,[](float v){return std::isfinite(v);}) ||
               std::memcmp(got,r.partials.data(),count*sizeof(float))){
                err="single group bit mismatch at record "+std::to_string(i);ok=false;break;
            }
            ++report.checked;if(r.sent)++report.active;else ++report.zero;
        }
        if(ok && group.peer_stats(0).requests!=report.active){err="single group request accounting mismatch";ok=false;}
    } catch(const std::exception& e){err=std::string("single group exception: ")+e.what();ok=false;}
    const auto primary=err;
    group.close(); // drain/join socket owner, then disconnect; no STOP/UNLOAD
    std::string cache_error;
    report.resident_unchanged=check.connect_local(host,port,5000,{},cache_error) &&
        check.cache_status(after,cache_error) && unchanged_resident(baseline,after);
    check.close();
    if(ok && report.resident_unchanged &&
       (after.cache_hits!=before.cache_hits+1 || after.reused_bytes!=before.reused_bytes+baseline.resident_bytes)){
        err="single group cache-hit accounting mismatch";return false;
    }
    if(!ok)err=primary;
    else if(!report.resident_unchanged)err="single group final resident verification failed: "+cache_error;
    return ok && report.resident_unchanged;
}
}
