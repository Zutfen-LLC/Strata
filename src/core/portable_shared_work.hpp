#pragma once
#include "strata/core/portable_ffn.hpp"
#include <stdexcept>
namespace strata::core::portable_detail {
// Stage actual shared-execution tags for one model forward. Publish only after its last
// layer/head succeeds, so a poisoned/partial forward emits no completion credit.
class SharedParticipation {
    bool required_;
    uint64_t calls_=0,rows_=0;
public:
    explicit SharedParticipation(bool required):required_(required){}
    void observe(const PortableRoute& route) {
        if(route.native_shared!=required_)throw std::runtime_error("portable model: shared participation differs from requested mode");
        if(route.native_shared){++calls_;++rows_;}
    }
    void observe(const std::vector<PortableRoute>& routes) {
        if(routes.empty())throw std::runtime_error("portable model: empty route participation");
        for(const auto& route:routes)if(route.native_shared!=required_)
            throw std::runtime_error("portable model: shared participation differs from requested mode");
        if(required_){++calls_;rows_+=routes.size();}
    }
    template<class Work> void commit(Work& work) const {
        work.native_shared_calls+=calls_;work.native_shared_rows+=rows_;
    }
};
}
