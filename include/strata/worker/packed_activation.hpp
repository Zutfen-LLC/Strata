#pragma once
#include "strata/worker/expert_wire.hpp"
namespace strata::worker {
// Qualification is a manually supplied, build-pair-specific 16 lowercase hex ID. No automatic eligibility.
bool packed_qualification_valid(const std::string&);
bool packed_format_supported(const strata::kernels::cpu::NativeFmt&);
std::shared_ptr<const PackedActivation> pack_activation(const strata::kernels::cpu::NativeFmt&,
    const float*, const uint8_t* token_active, int T, std::string& reason, uint64_t* quantized_rows=nullptr);
bool restore_activation(const strata::kernels::cpu::NativeFmt&, const PackedActivation&,
    std::vector<uint8_t>& scratch);
}
