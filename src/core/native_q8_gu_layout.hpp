#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strata::core::portable_detail {
inline void q8_gu_need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
// Keep the native offsets: count[0], weights[1..24], starts[25..49], IDs[50..129].
inline constexpr size_t q8_entry_capacity=80, q8_metadata_words=50+q8_entry_capacity;
using Q8GroupMetadata=std::array<int32_t,q8_metadata_words>;
inline std::array<int32_t,32> q8_stage_metadata(const int32_t* starts,size_t count) {
    q8_gu_need(starts && count>=1 && count<=25,"native stages: bounded group starts required");
    std::array<int32_t,32> result{};
    for(size_t i=0;i<count;++i)result[i]=starts[i];
    result[31]=int32_t(count-1);return result;
}
// native_gu_kernel<8> metadata, replacing pointers with offsets into the
// existing packed weight buffer. Inactive entries are not interpreted.
struct Q8GuLayout {
    uint32_t h,f,entries,tokens,grid_groups;
    std::vector<uint32_t> weight_offsets;
    std::vector<int32_t> starts,token_ids;
};
inline Q8GroupMetadata q8_gu_metadata(const Q8GuLayout& l,size_t weight_bytes) {
    q8_gu_need(l.h && l.h<=2560 && l.h%256==0 && l.f && l.f<=640 && l.f%32==0,
        "Q8 GU: bounded native dimensions required");
    q8_gu_need(l.entries && l.entries<=q8_entry_capacity && l.tokens && l.tokens<=64 && l.grid_groups && l.grid_groups<=24,
        "Q8 GU: bounded capacities/grid required");
    const size_t groups=l.weight_offsets.size(),gu_bytes=2*size_t(l.f)*(l.h/32)*34;
    q8_gu_need(groups<=24 && l.starts.size()==groups+1 && l.token_ids.size()==l.entries &&
        weight_bytes>0 && weight_bytes<=128u*1024*1024 && weight_bytes%4==0,
        "Q8 GU: metadata/weight shape mismatch");
    for(size_t i=0;i<l.starts.size();++i)q8_gu_need(l.starts[i]>=0 && uint32_t(l.starts[i])<=l.entries &&
        (!i || l.starts[i]>=l.starts[i-1]),"Q8 GU: invalid group interval");
    for(size_t g=0;g<groups;++g)if(l.starts[g]!=l.starts[g+1])
        q8_gu_need(l.weight_offsets[g]%4==0 && l.weight_offsets[g]<=weight_bytes &&
            gu_bytes<=weight_bytes-l.weight_offsets[g],"Q8 GU: active weight interval exceeds buffer");
    for(int32_t e=l.starts.front();e<l.starts.back();++e)
        q8_gu_need(l.token_ids[e]>=0 && uint32_t(l.token_ids[e])<l.tokens,"Q8 GU: active token out of range");
    Q8GroupMetadata metadata{};metadata[0]=int32_t(groups);
    for(size_t i=0;i<groups;++i)metadata[1+i]=int32_t(l.weight_offsets[i]);
    for(size_t i=0;i<l.starts.size();++i)metadata[25+i]=l.starts[i];
    for(size_t i=0;i<l.token_ids.size();++i)metadata[50+i]=l.token_ids[i];
    return metadata;
}
}
