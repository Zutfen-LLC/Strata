#pragma once
#include "strata/core/portable_model.hpp"
#include <algorithm>
#include <stdexcept>
namespace strata::core::portable_detail {
// Serial model-layer ledger, one slot per unique physical backend. Retained
// buffers/private allowances add; token-local schedulers use one peak. The
// temporary peak remains additive with resident expert tensor/runtime budgets.
inline void account_device_memory(PortableModelReport& report, size_t die, size_t cap,
                                  const PortableMemoryUsage& m) {
    if (report.device_admission.at(die) > cap)
        throw std::runtime_error("portable model: existing die admission exceeds cap");
    (void) report.device_state.at(die);
    (void) report.device_weights.at(die); // validate ledger shape before any commit
    const size_t old_peak=report.device_token_temporary_bound.at(die);
    const size_t peak=std::max(old_peak,m.device_token_temporary_bound);
    const size_t old_conversion=report.device_shared_conversion_bound.at(die);
    const size_t conversion=std::max(old_conversion,m.device_shared_conversion_bound);
    size_t required=0;
    const size_t remaining=cap-report.device_admission[die];
    for(size_t bytes:{m.device_state,m.device_weights,m.device_graph,m.device_extra_bound,peak-old_peak,conversion-old_conversion}) {
        if(bytes>remaining-required)
            throw std::runtime_error("portable model: separate die "+std::to_string(die)+" admission cap exceeded");
        required+=bytes;
    }
    // Commit only after all guards, including peak growth, pass.
    report.device_admission[die]+=required;
    report.device_scheduler[die]+=m.device_graph;
    report.device_state.at(die)+=m.device_state;
    report.device_weights.at(die)+=m.device_weights;
    report.device_workspace_bound[die]+=m.device_extra_bound;
    report.device_token_temporary_bound[die]=peak;
    report.device_shared_conversion_bound[die]=conversion;
}
} // namespace
