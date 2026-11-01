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

## Default-disabled model seam

The separate model scope uses `--qsa-norm-rope-resident-weights PREFIX` in both
control/on and adds `--native-qsa-norm-rope-t1` only to on. It applies the same
three methods to ordinary QSA layers 3/7/.../47, without changing upstream
projections, HC/GDN choices, pooled-key normalization or attention/selection.
Actual completed raw Q-full (stride512), K and indexer-query tensors are retained
as graph outputs. Native candidates run before readback and existing consumers;
the unused GGML position leaf is not uploaded in the native path.

Only the three F32 normalization roles move to the resident bank; their original
host allocations are replaced. Both arms reserve the same aligned candidate,
position/diagnostic scratch and component status storage, plus a conservative
2 MiB host allowance per QSA layer. The existing ledger and native planner decide
fit. No cap, reserve, quota or placement rule changes. Production/device-authority
QSA, chunks, table RoPE or explicitly disabled native fusion refuse in this scope.

Initial layer3 Q/K/indexer-query own-input fixtures and component outputs are
recorded. Accepted first-key FP16 bytes bind the K candidate to the existing KV
commit. Later failures retain current input/status and do not publish stale
candidate outputs. Paid calls are counted even if later FFN work refuses; eager
KV/indexer publication still follows successful FFN. Per-request QSA state digests
and actual call deltas support exact within-arm reset replay, not cross-backend
state equality. The default path allocates or dispatches none of this storage.

Local coordinator tests use the real GGML projections/consumers/state with only
the component arithmetic mocked; they establish plumbing and noncommit, not GPU
numerics. The actual standalone native/Vulkan component gate must pass before
the bounded same-image fresh-planning/model comparison. Neither gate establishes
native QSA projection, pooled indexer, attention/selection, accepted-prefix/window,
quality or performance parity.

## Bounded model result

The comparison at `eab5f489c7bf8f72ae95899c547190f9aabe6e64` passes fresh
admission and matched ownership, two reset-separated requests per arm, initial
layer3 own-input controls, accepted FP16 KV-key linkage and 24 new module uses.
On runs 1,008 paid component calls per request; control runs none. Within-arm
state/head/routes reset exactly; both arms generate the same three IDs. Routes
and logits differ (maximum retained matching-history absolute difference
1.3833283185958862), which remains diagnostic. All owned processes exit/reap and
restore the measured baseline across three cleanup observations. Guard samples
are discrete, not continuous coverage. This is bounded integration evidence,
not full native QSA, accepted-prefix/window, quality or performance parity;
qualified=false and defaults remain unchanged.
