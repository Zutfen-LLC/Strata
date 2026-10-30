#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
namespace strata::core::portable_detail {
inline void capacity_need(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
struct NativeCapacityRoom {
    size_t dense,limit,available,total,reserve,tensor_ceiling,workspace;
    size_t tensor_room() const {
        capacity_need(total && limit<=total && available<=total && reserve && workspace && tensor_ceiling,"native capacity: invalid memory evidence/budgets");
        capacity_need(dense<=limit && workspace<=limit-dense,"native capacity: no ledger room for runtime");
        capacity_need(reserve<available && workspace<=available-reserve,"native capacity: no physical room for runtime/reserve");
        return std::min({tensor_ceiling,limit-dense-workspace,available-reserve-workspace});
    }
};
using NativeCapacityEstimate=std::function<size_t(size_t)>;
inline size_t native_fit_count(size_t maximum,size_t room,const NativeCapacityEstimate& estimate,size_t multiple=1) {
    capacity_need(multiple && maximum<=SIZE_MAX/multiple,"native capacity: slot count overflow");
    size_t lo=0,hi=maximum;
    while(lo<hi) {
        const size_t mid=lo+(hi-lo)/2+1;
        if(estimate(mid*multiple)<=room)lo=mid;else hi=mid-1;
    }
    return lo;
}
inline size_t native_primary_quota(size_t requested_slots,size_t experts,const std::vector<size_t>& strides,
                                   size_t room,const NativeCapacityEstimate& estimate) {
    capacity_need(requested_slots && experts && !strides.empty(),"native capacity: invalid primary request");
    size_t sum=0,largest=0;
    for(size_t n:strides){capacity_need(n && n<=SIZE_MAX-sum,"native capacity: stride sum overflow");sum+=n;largest=std::max(largest,n);}
    capacity_need(requested_slots<=SIZE_MAX/largest,"native capacity: requested byte ceiling overflow");
    const size_t upper=std::min(experts,requested_slots*largest/sum);
    const size_t q=native_fit_count(upper,room,estimate,strides.size());
    capacity_need(q,"native capacity: no complete per-layer quota fits");return q;
}
struct NativeRankedSelection {std::vector<std::pair<int32_t,int32_t>> primary,remaining;};
inline NativeRankedSelection native_ranked_selection(const std::vector<std::pair<int32_t,int32_t>>& ranked,
                                                     size_t layers,size_t experts,size_t quota) {
    capacity_need(layers && experts && quota && quota<=experts,"native capacity: invalid quota/geometry");
    NativeRankedSelection s;std::vector<size_t> counts(layers);std::set<std::pair<int32_t,int32_t>> seen;
    for(auto pair:ranked) {
        capacity_need(pair.first>=0 && size_t(pair.first)<layers && pair.second>=0 && size_t(pair.second)<experts,
                      "native capacity: profile pair outside geometry");
        if(!seen.insert(pair).second)continue;
        if(counts[pair.first]<quota){s.primary.push_back(pair);++counts[pair.first];}
        else s.remaining.push_back(pair);
    }
    capacity_need(!s.primary.empty() && !s.remaining.empty(),"native capacity: empty primary or unclaimed helper profile");
    return s;
}
}
