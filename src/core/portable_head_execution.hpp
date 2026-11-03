#pragma once
#include "strata/core/portable_model.hpp"
#include <utility>
namespace strata::core::portable_detail {
// The caller supplies the existing paid head operation. Elision does not
// touch it and cannot expose a stale vector from a preceding position.
template<class Run>
bool model_optional_head(bool project,PortableModelWork& work,std::vector<float>& logits,Run&& run){
    logits.clear();
    if(!project){++work.heads_skipped;return true;}
    ++work.heads_executed;return std::forward<Run>(run)();
}
}
