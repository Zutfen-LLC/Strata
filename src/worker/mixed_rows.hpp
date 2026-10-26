#pragma once
// Shared pure seam for the group and bounded offline diagnostics. No socket/session state.
#include "strata/worker/expert_worker.hpp"
#include <algorithm>
#include <cmath>

namespace strata::worker {
struct MixedRowsPeer {
    const EntryRowsReply& reply;
    const std::vector<uint8_t>& mask;
};
inline bool reduce_mixed_rows(int H,int T,int K,const std::vector<float>& cpu_rows,
    const std::vector<uint8_t>& cpu_mask,const std::vector<uint8_t>& expected,const float* weights,
    const std::vector<MixedRowsPeer>& peers,std::vector<float>& partials,std::string& err) {
    partials.clear();auto refuse=[&](const char* why){err=why;return false;};
    const size_t n=(size_t)T*K;
    if(H<1 || H>kMaxNEmbd || T<1 || T>kMaxT || K<1 || K>kMaxK || !weights ||
       expected.size()!=n || cpu_mask.size()!=n || cpu_rows.size()!=n*H)
        return refuse("ordered mixed row state/shape mismatch");
    std::vector<float> rows(cpu_rows);std::vector<uint8_t> seen(cpu_mask);
    for(size_t i=0;i<n;++i) {
        if(expected[i]>1 || cpu_mask[i]>1 || cpu_mask[i]>expected[i] || !std::isfinite(weights[i]))
            return refuse("ordered mixed CPU ownership/weights mismatch");
        if(cpu_mask[i])for(int h=0;h<H;++h)if(!std::isfinite(rows[i*H+h]))return refuse("nonfinite mixed CPU row");
    }
    for(const auto& peer:peers) {
        const auto& reply=peer.reply;
        if(reply.rows.size()!=reply.indices.size()*(size_t)H || peer.mask.size()!=n)
            return refuse("ordered mixed GPU row shape mismatch");
        std::vector<uint8_t> peer_seen(n,0);
        for(size_t j=0;j<reply.indices.size();++j) {
            const size_t i=reply.indices[j];
            if(i>=n || !expected[i] || !peer.mask[i] || seen[i])return refuse("ordered mixed row overlap/foreign index");
            for(int h=0;h<H;++h)if(!std::isfinite(reply.rows[j*H+h]))return refuse("nonfinite mixed GPU row");
            seen[i]=peer_seen[i]=1;std::copy_n(reply.rows.data()+j*H,H,rows.data()+i*H);
        }
        if(peer_seen!=peer.mask)return refuse("ordered mixed GPU rows missing");
    }
    if(seen!=expected)return refuse("ordered mixed row coverage gap");
    std::vector<float> sum((size_t)T*H);
    ExpertWorkerService::accumulate_partial(T,K,expected.data(),weights,rows.data(),H,sum.data());
    if(!std::all_of(sum.begin(),sum.end(),[](float v){return std::isfinite(v);}))return refuse("ordered mixed sum nonfinite");
    partials=std::move(sum);return true;
}
} // namespace strata::worker
