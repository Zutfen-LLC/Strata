#pragma once
// Diagnostic only: the original two peer masks, evaluated on one full resident session.
#include "strata/worker/expert_wire.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace strata::worker {
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
