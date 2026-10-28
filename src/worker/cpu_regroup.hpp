#pragma once
// Diagnostic reductions on one full resident session; no worker/protocol or arena changes.
#include "strata/worker/expert_wire.hpp"
#include "strata/worker/expert_worker.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace strata::worker {
template<class Exchange>
bool ordered_cpu_request(const InferRequest& input, uint16_t H, uint64_t& sequence, Exchange exchange,
                         InferReply& output, uint64_t& calls, uint64_t& row_bytes, std::string& err) {
    output.partials.clear();calls=row_bytes=0;
    const size_t entries=(size_t)input.T*input.K, count=(size_t)input.T*H;
    if(input.T<1 || input.T>kMaxT || input.K<1 || input.K>kMaxK || H<1 || H>kMaxNEmbd ||
       input.mask.size()!=entries || input.ids.size()!=entries || input.weights.size()!=entries ||
       input.activations.size()!=count) {err="ordered input bounds/shape";return false;}
    if(std::find(input.mask.begin(),input.mask.end(),1)==input.mask.end()) {
        output.partials.assign(count,0.f);return true;
    }
    auto finite=[&](const InferReply& reply) {
        return reply.partials.size()==count &&
            std::all_of(reply.partials.begin(),reply.partials.end(),[](float v){return std::isfinite(v);});
    };
    InferReply shadow, row;
    ++calls;
    if(!exchange(input,shadow,err))return false;
    if(!finite(shadow)) {err="ordered shadow shape/nonfinite";return false;}
    row_bytes=entries*H*sizeof(float); // <=8*10*2560*4 =819200, checked before allocating
    std::vector<float> rows(entries*H,0.f), exact(count,0.f);
    InferRequest request=input;
    std::fill(request.weights.begin(),request.weights.end(),1.f);
    for(int rank=0;rank<input.K;++rank) {
        std::fill(request.mask.begin(),request.mask.end(),0);
        bool active=false;
        for(int token=0;token<input.T;++token) {
            const size_t i=(size_t)token*input.K+rank;
            request.mask[i]=input.mask[i];active=active || request.mask[i];
        }
        if(!active)continue;
        if(sequence==std::numeric_limits<uint64_t>::max()) {err="ordered sequence exhausted";return false;}
        request.sequence=++sequence;++calls;
        if(!exchange(request,row,err))return false;
        if(!finite(row)) {err="ordered row shape/nonfinite";return false;}
        for(int token=0;token<input.T;++token) {
            const size_t i=(size_t)token*input.K+rank;
            if(request.mask[i])std::copy_n(row.partials.data()+(size_t)token*H,H,rows.data()+i*H);
        }
    }
    ExpertWorkerService::accumulate_partial(input.T,input.K,input.mask.data(),input.weights.data(),
                                            rows.data(),H,exact.data());
    if(!std::all_of(exact.begin(),exact.end(),[](float v){return std::isfinite(v);}) ||
       std::memcmp(exact.data(),shadow.partials.data(),count*sizeof(float))) {
        err="ordered bit gate mismatch at layer "+std::to_string(input.layer);return false;
    }
    output.session=shadow.session;output.sequence=shadow.sequence;output.partials=std::move(exact);
    return true; // no shadow or reconstructed partial is exposed before the exact bit gate
}

template<class Exchange>
bool regroup_cpu_request(const InferRequest& input, const std::vector<int8_t>& owners,
                         uint16_t experts, uint16_t H, uint64_t& sequence, Exchange exchange,
                         InferReply& output, uint64_t (&calls)[2], std::string& err) {
    output.partials.clear();
    InferRequest request=input;
    InferReply combined, part;
    bool first=true;
    for(int peer=0;peer<2;++peer) {
        bool active=false;
        for(size_t i=0;i<input.mask.size();++i) {
            request.mask[i]=input.mask[i] && owners.at((size_t)input.layer*experts+input.ids[i])==peer;
            active=active || request.mask[i];
        }
        if(!active)continue; // no frame and no addition of a zero row (including signed zero)
        if(!first) {
            if(sequence==std::numeric_limits<uint64_t>::max()) {err="regroup sequence exhausted";return false;}
            request.sequence=++sequence;
        }
        ++calls[peer];
        if(!exchange(request,part,err))return false; // each exchange drains before the next starts
        if(part.partials.size()!=(size_t)input.T*H ||
           !std::all_of(part.partials.begin(),part.partials.end(),[](float v){return std::isfinite(v);})) {
            err="regroup reply shape/nonfinite";return false;
        }
        if(first) {combined=std::move(part);first=false;}
        else {
            for(size_t i=0;i<combined.partials.size();++i) {
                combined.partials[i]=combined.partials[i]+part.partials[i];
                if(!std::isfinite(combined.partials[i])) {err="regroup sum nonfinite";return false;}
            }
        }
    }
    if(first)combined.partials.assign((size_t)input.T*H,0.f);
    output=std::move(combined); // publish only after every active peer succeeds
    return true;
}
} // namespace strata::worker
