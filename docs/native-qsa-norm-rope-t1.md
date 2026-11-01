# Native QSA normalization and RoPE: ordinary T1

The specification is `Verifier::record_window()`'s `norm_rope_on` / `norm_rope`
branches, `native_qsa.cu::norm<256>` and `native_rope.cu::norm_rope_kernel`.
This default-disabled standalone component ports those operations, with a
separate normalized output for the weighted-RMS method. It does not port QSA
projection GEMVs, pooled indexer arithmetic, selection/attention, state commit,
window scheduling or other RoPE modes.

Supported geometry is text-only T1, widths 128/256, one to 24 rows, input stride
one or two widths, 64 rotated channels, epsilon 1e-6, unscaled frequency base 1e7
and positions 0..255. These are the current bounded model target's existing
settings. Table/MRoPE, scaling/YaRN, other widths, large context and nonfinite or
preserved-subnormal local arithmetic refuse explicitly; no mode is substituted.

For these widths each of native's 256 threads accumulates at most one square.
The first square/FMA-with-zero has the same RNE result. Preserve the XOR 16/8/4/2/1
warp reduction, eight-warp padded second reduction, division by width, epsilon,
rsqrt and `(scale*x)*gamma`. In RoPE preserve host F32 theta_scale, device Pow,
position multiplication and the native NEOX pair layout. `b*sin` and `b*cos`
round before the explicit two rotation FMAs. The suffix beyond channel 63 is
exactly the normalized output. Strided padding is never a norm operand.

Acceptance is declared before hardware results. Independent source-order
controls require exact reduction/mean/epsilon, weighted multiplication and
own-coefficient rotation. The existing GDN rsqrt policy is reused. Native fast
Pow derives from exp2f(y*__log2f(x)); for this theta in [.5,2], use CUDA's published
2^-22 absolute log2 error, RNE product rounding and 2-ULP exp2 bound, with the
existing conservative 0.5-ULP policy slack. This is a qualification requirement
for the Vulkan adapter, not a claim that GLSL itself guarantees that bound.

For trig in [-pi,pi], require CUDA's published 2^-21.41 absolute intrinsic bound.
CUDA explicitly does not give that same tight bound outside this interval.
For the supported larger angles, reuse **Strata's existing**
`rope_parity.cpp` check 5: at most 3e-3 absolute error divided by the maximum
amplitude of that row's pre-rotation input (floor 1e-30), against the unscaled
FP64-table reference. The same unit-basis criterion bounds emitted sin/cos
independently, including zero-input cases. It remains deliberately distinct from
local operation checks and from full-model acceptance. Row-amplitude scaling
handles cancellation/near-zero outputs without a fitted relative epsilon.
Zero-position sin/cos identities and the unrotated suffix remain exact.

The CUDA probe calls the unchanged production weighted norm, in-place RoPE and
fused norm/RoPE launchers. It requires fused output bit-equal to the original pair
and an independent source-control kernel to reproduce the actual outputs.
Vulkan computes separate candidate outputs, preserving inputs; invalid/aliased
preflight must not dispatch or write, and repeat/reset/retry must agree exactly.
All effective variants require strict FMA, 32-bit RTE/signed-zero controls and
protected basic operations. No cross-backend bit identity, whole-model quality,
performance or general intrinsic equivalence follows from a component pass.

References: [CUDA floating-point functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html),
[GLSL extended instructions](https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html),
and the repository's native `src/kernels/rope_parity.cpp` checks 5/6.
