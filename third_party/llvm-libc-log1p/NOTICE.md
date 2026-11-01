# LLVM libc log1pf source and bounded Vulkan adapter

Upstream: https://github.com/llvm/llvm-project/tree/4497566c7ca6b1d958fe27678f541308b948f6c2/libc
Commit: `4497566c7ca6b1d958fe27678f541308b948f6c2`.
License: Apache-2.0 WITH LLVM-exception; full text in LICENSE.TXT.
The three upstream math headers are retained verbatim, with original notices.
Their hashes are recorded in provenance.json. No upstream coefficients are fitted
or regenerated. The adapter is a modified work, not an upstream Vulkan guarantee.

The GLSL adapter selects upstream's accurate non-FMA configuration, with separately
rounded binary64 multiply/add and its original C_LOG range-reduction compensation,
R_LOG/LOG_R tables and small/large polynomials. Constants are encoded as exact
binary64 words. Integer extraction uses the two 32-bit halves rather than requiring
shaderInt64. It requires core shaderFloat64, actual logical enablement, RTE64 and
signed-zero/Inf/NaN preservation. It does not require FP64 FMA. No floating log or
division approximation is used by this candidate.

Supported experiment: RNE, signed zero or positive normal F32 <=2^29. Negative,
subnormal, nonfinite and larger inputs refuse before submission. All upstream
positive hard-case corrections inside that domain are retained; RNE's
round_result_slightly_up/down returns the given RNE value. Negative/special-value
branches and hard cases above 2^43 are unreachable and omitted from GLSL. errno,
floating exception flags and other rounding modes are not claimed. No outer
Strata softplus/exp/threshold/gate multiplication is changed or integrated.

The host oracle uses original algorithm bodies after removing only include
lines. Minimal test shims provide bit transport, separate multiply/add and
RNE-only special returns; constexpr/exception plumbing is not runtime arithmetic.
The separately compiled GLSL adapter must match this oracle exactly; the independent
Decimal reference additionally enforces the unchanged one-adjacent-step gate.
