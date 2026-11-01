#pragma once
#include <array>
#include "../kernels/vulkan/native_q8_abi.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strata::core::portable_detail {
inline void q8_gu_need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
// Count, 80 pairs of address words, 81 starts, 80 entry IDs.
inline constexpr size_t q8_entry_capacity=STRATA_Q8_GROUP_CAPACITY, q8_metadata_words=STRATA_Q8_METADATA_WORDS;
using Q8GroupMetadata=std::array<int32_t,q8_metadata_words>;
inline void q8_store_address(Q8GroupMetadata& m,size_t group,uint64_t address) {
    q8_gu_need(group<q8_entry_capacity,"Q8 address group exceeds capacity");
    m[STRATA_Q8_ADDRESS_WORD+2*group]=int32_t(uint32_t(address));
    m[STRATA_Q8_ADDRESS_WORD+2*group+1]=int32_t(uint32_t(address>>32));
}
inline std::array<int32_t,STRATA_Q8_STAGE_WORDS> q8_stage_metadata(const int32_t* starts,size_t count) {
    q8_gu_need(starts && count>=1 && count<=q8_entry_capacity+1,"native stages: bounded group starts required");
    std::array<int32_t,STRATA_Q8_STAGE_WORDS> result{};
    for(size_t i=0;i<count;++i)result[i]=starts[i];
    result[STRATA_Q8_STAGE_COUNT_WORD]=int32_t(count-1);return result;
}
// Host layouts contain checked per-owner byte offsets. Dispatch converts them
// to BDA references; tensor.data is never interpreted as a GPU address.
struct Q8GuLayout {
    uint32_t h,f,entries,tokens,grid_groups;
    std::vector<uint64_t> weight_offsets;
    std::vector<int32_t> starts,token_ids;
};
inline Q8GroupMetadata q8_gu_metadata(const Q8GuLayout& l,const std::vector<size_t>& weight_bytes) {
    q8_gu_need(l.h && l.h<=2560 && l.h%256==0 && l.f && l.f<=640 && l.f%32==0,
        "Q8 GU: bounded native dimensions required");
    q8_gu_need(l.entries && l.entries<=q8_entry_capacity && l.tokens && l.tokens<=64 && l.grid_groups && l.grid_groups<=q8_entry_capacity,
        "Q8 GU: bounded capacities/grid required");
    const size_t groups=l.weight_offsets.size(),gu_bytes=2*size_t(l.f)*(l.h/32)*34;
    q8_gu_need(groups<=q8_entry_capacity && l.starts.size()==groups+1 && l.token_ids.size()==l.entries &&
        weight_bytes.size()==groups,
        "Q8 GU: metadata/weight shape mismatch");
    for(size_t i=0;i<l.starts.size();++i)q8_gu_need(l.starts[i]>=0 && uint32_t(l.starts[i])<=l.entries &&
        (!i || l.starts[i]>=l.starts[i-1]),"Q8 GU: invalid group interval");
    for(size_t g=0;g<groups;++g)if(l.starts[g]!=l.starts[g+1])
        q8_gu_need(l.weight_offsets[g]%4==0 && weight_bytes[g]>0 && weight_bytes[g]%4==0 && l.weight_offsets[g]<=weight_bytes[g] &&
            gu_bytes<=weight_bytes[g]-l.weight_offsets[g],"Q8 GU: active weight interval exceeds buffer");
    for(int32_t e=l.starts.front();e<l.starts.back();++e)
        q8_gu_need(l.token_ids[e]>=0 && uint32_t(l.token_ids[e])<l.tokens,"Q8 GU: active token out of range");
    Q8GroupMetadata metadata{};metadata[0]=int32_t(groups);
    for(size_t i=0;i<groups;++i)q8_store_address(metadata,i,l.weight_offsets[i]);
    for(size_t i=0;i<l.starts.size();++i)metadata[STRATA_Q8_START_WORD+i]=l.starts[i];
    for(size_t i=0;i<l.token_ids.size();++i)metadata[STRATA_Q8_ID_WORD+i]=l.token_ids[i];
    return metadata;
}
inline Q8GroupMetadata q8_gu_metadata(const Q8GuLayout& l,size_t weight_bytes) {
    return q8_gu_metadata(l,std::vector<size_t>(l.weight_offsets.size(),weight_bytes));
}

}
