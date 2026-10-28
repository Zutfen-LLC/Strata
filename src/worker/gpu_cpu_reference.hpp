#pragma once
// Slow diagnostic for the qualified Q8_0 AVX2/non-VNNI CPU baseline only.
#include "strata/kernels/cpu/native_expert.hpp"
#include <vector>
namespace strata::worker {
bool q8_cpu_reference_format(strata::kernels::cpu::NativeFmt&, std::string&);
void q8_cpu_reference_input(const strata::kernels::cpu::NativeFmt&, const float*, int T, std::vector<uint8_t>&);
void q8_cpu_reference_hidden(const strata::kernels::cpu::NativeFmt&, const float* gate, const float* up,
                             int entries, std::vector<uint8_t>&);
// CUDA: each row follows eight CPU accumulators, sequential blocks, explicit RN FMA and CPU hsum order.
bool q8_avx2_reference_dot(const unsigned long long* entry_weights, const int32_t* input_row,
                           const uint8_t* q8_0, int n_in, int n_out, int entries, size_t weight_offset,
                           float* out, void* stream);
}
