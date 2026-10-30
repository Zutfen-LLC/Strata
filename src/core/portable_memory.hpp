#pragma once
#include "strata/core/portable_model.hpp"
#include <algorithm>
#include <stdexcept>
namespace strata::core::portable_detail {
// State can be allocated inside a layer constructor before its actual ledger
// is available. This estimate is a preflight requirement only, never a charge.
inline size_t layer_state_preflight_bytes(const PortableModelOptions &options, size_t layer,
                                         ggml_backend *backend) {
    if (!backend)
        return 0;
    if (layer % 4 == 3)
        return options.qsa_production ? PortableQsaLayer::device_state_allocation_size(backend, options.context) : 0;
    return options.gdn_resident_state ? PortableGdnOptions{}.resident_state_budget : 0;
}
struct DeviceAdmissionContext {
    size_t slot, layer;
    const char* module;
    size_t policy_cap, free_reserve;
};
inline void preflight_device_memory(size_t used, size_t cap, size_t next,
                                    const DeviceAdmissionContext* context = nullptr) {
    if (used > cap || next > cap - used) {
        // Diagnostics only: keep the exact overflow-safe admission predicate.
        // Optional API callers may lack the CLI's policy/reserve evidence.
        throw std::runtime_error(std::string("portable model: next module exceeds separate die admission cap") +
            " slot=" + (context ? std::to_string(context->slot) : "unknown") +
            " layer=" + (context ? std::to_string(context->layer) : "unknown") +
            " module=" + (context ? context->module : "unknown") +
            " admitted_bytes=" + std::to_string(used) +
            " next_estimated_bytes=" + std::to_string(next) +
            " policy_cap_bytes=" + (context && context->policy_cap ? std::to_string(context->policy_cap) : "unknown") +
            " effective_limit_bytes=" + std::to_string(cap) +
            " reserve_bytes=" + (context && context->free_reserve ? std::to_string(context->free_reserve) : "unknown"));
    }
}
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
