#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
namespace strata::core {
// Original native STRP-v1 reader. Optional caller bound prevents untrusted
// counts from allocating unbounded host storage; native callers retain default.
bool read_expert_profile(const std::string&,int64_t,int64_t,
    std::vector<std::pair<int32_t,int32_t>>&,int64_t&,std::string&);
bool read_expert_profile_bounded(const std::string&,int64_t,int64_t,
    std::vector<std::pair<int32_t,int32_t>>&,int64_t&,std::string&,size_t max_ranked);
}
