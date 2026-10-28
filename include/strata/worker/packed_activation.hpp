#pragma once
#include "strata/worker/expert_wire.hpp"
namespace strata::worker {
// A qualified build carries its verified build-pair ID. Empty means no recorded eligibility.
std::string default_packed_qualification();
bool packed_qualification_valid(const std::string&);
bool packed_format_supported(const strata::kernels::cpu::NativeFmt&);
std::shared_ptr<const PackedActivation> pack_activation(const strata::kernels::cpu::NativeFmt&,
    const float*, const uint8_t* token_active, int T, std::string& reason, uint64_t* quantized_rows=nullptr);
bool restore_activation(const strata::kernels::cpu::NativeFmt&, const PackedActivation&,
    std::vector<uint8_t>& scratch);
}
