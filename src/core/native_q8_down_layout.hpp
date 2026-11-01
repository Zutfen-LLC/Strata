#pragma once
#include "native_q8_gu_layout.hpp"
namespace strata::core::portable_detail {
struct Q8DownLayout {
    uint32_t h,f,entries,destinations,grid_groups;
    // Offsets point to each group's down matrix, not to the expert blob start.
    std::vector<uint64_t> weight_offsets;
    std::vector<int32_t> starts,destination_ids;
};
inline Q8GroupMetadata q8_down_metadata(const Q8DownLayout& l,const std::vector<size_t>& weight_bytes) {
    q8_gu_need(l.h && l.h<=2560 && l.h%256==0 && l.f && l.f<=640 && l.f%32==0,
        "Q8 down: bounded native dimensions required");
    q8_gu_need(l.entries && l.entries<=q8_entry_capacity && l.destinations && l.destinations<=q8_entry_capacity && l.grid_groups && l.grid_groups<=q8_entry_capacity,
        "Q8 down: bounded capacities/grid required");
    const size_t groups=l.weight_offsets.size(),down_bytes=size_t(l.h)*(l.f/32)*34;
    q8_gu_need(groups<=q8_entry_capacity && l.starts.size()==groups+1 && l.destination_ids.size()==l.entries &&
        weight_bytes.size()==groups,"Q8 down: metadata/weight shape mismatch");
    for(size_t i=0;i<l.starts.size();++i)q8_gu_need(l.starts[i]>=0 && uint32_t(l.starts[i])<=l.entries &&
        (!i || l.starts[i]>=l.starts[i-1]),"Q8 down: invalid group interval");
    for(size_t g=0;g<groups;++g)if(l.starts[g]!=l.starts[g+1])
        q8_gu_need(l.weight_offsets[g]%4==0 && weight_bytes[g]>0 && weight_bytes[g]%4==0 && l.weight_offsets[g]<=weight_bytes[g] &&
            down_bytes<=weight_bytes[g]-l.weight_offsets[g],"Q8 down: active weight interval exceeds buffer");
    for(int32_t e=l.starts.front();e<l.starts.back();++e)
        q8_gu_need(l.destination_ids[e]>=0 && uint32_t(l.destination_ids[e])<l.destinations,"Q8 down: active destination out of range");
    // Direct stores retain native semantics. Do not invent atomic accumulation,
    // sorting, or a resolution policy for destinations shared across groups.
    Q8GroupMetadata metadata{};metadata[0]=int32_t(groups);
    for(size_t i=0;i<groups;++i)q8_store_address(metadata,i,l.weight_offsets[i]);
    for(size_t i=0;i<l.starts.size();++i)metadata[STRATA_Q8_START_WORD+i]=l.starts[i];
    for(size_t i=0;i<l.destination_ids.size();++i)metadata[STRATA_Q8_ID_WORD+i]=l.destination_ids[i];
    return metadata;
}
inline Q8GroupMetadata q8_down_metadata(const Q8DownLayout& l,size_t weight_bytes) {
    return q8_down_metadata(l,std::vector<size_t>(l.weight_offsets.size(),weight_bytes));
}

}
