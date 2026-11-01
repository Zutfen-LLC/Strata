# Native GDN positive log1p dependency probe

This standalone experiment compares CUDA `log1pf(x)` with Strata's existing
Metal compensated positive expression from `metal/kernels/gdn.metal`:

```cpp
u = 1.0f + x;
y = u == 1.0f ? x : (log(u) * x) / (u - 1.0f);
```

`Verifier::record_window()` calls `gdn_ab_multi()`; its CUDA specification in
`src/kernels/cuda/verify_kernels.cu` uses `v > 20 ? v : log1pf(__expf(v))`.
The probe covers the positive log1p dependency only. It neither changes the native
outer branch nor adds an extra exponential to the log gate. It does not enable
an a/b or model path. Production defaults and existing native kernels are unchanged.
The diagnostic CUDA compensated expression is not a replacement for CUDA log1pf.

## Criteria fixed before hardware execution

Each fixture has 256 inputs, signed zero or positive normal F32 no greater than
2^29. This ceiling conservatively contains exp(v) for v <= 20, including the
existing exp error allowance. Negative nonzero, subnormal, nonfinite and larger
inputs explicitly refuse. No clipping or saturation is performed.

The primary result must be within one adjacent positive F32 representable step
of correctly rounded RNE log1p(x). This deliberately unambiguous binade-aware
criterion uses CUDA's documented 1 ULP log1pf accuracy as the native requirement;
there is no extra half-ULP policy slack. Zero signs are exact. On Vulkan, the
existing Metal u==1 branch must return x exactly. This is a bounded dependency
qualification, not a universal accuracy proof or full-model parity claim.

The offline oracle computes ln(1+x) with independently converged 120/180-digit
Decimal arithmetic and rounds by exact rational distances. Tests pin known
vectors independently of the compensated computation, reject common erroneous
outputs on both backends and test a primitive-valid but fidelity-invalid case.
Rounded addition, multiplication association and subtraction are exact local
checks. Division is checked against its observed numerator/denominator with a
2.5 ULP bound, not forced to host-RNE. Both operands and intermediates are exposed.
CUDA's auxiliary logf has its documented 1 ULP bound; the probe target does not
use --use_fast_math. Vulkan log has absolute error <2^-21 on [0.5,2], and 3 ULP
outside that interval. Passing these primitive checks alone does not pass the
primary log1p requirement. CUDA compensated accuracy is reported diagnostically.

The near-one Vulkan guarantee is weak enough to permit a logarithm that produces
roughly three times the correct result at x=2^-23. Therefore source algebra and
minimum Vulkan guarantees do not establish native fidelity. Hardware failure of
the tighter primary criterion is a real unsupported dependency, not permission to
loosen the target. This probe exists to make that decision before a/b integration.

Sources: [CUDA mathematical functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html),
[Vulkan shader precision](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/appendices/spirvenv.adoc).

## Execution and limits

Three deterministic, locally generated fixtures cover signed zeros/small x,
branch/binade/exp(20) boundaries, and ordinary varied mantissas. The host executable
also generates the output-field contract used by runners, so producer names and
byte counts are checked before numerical validation. No fixture payload transfer
is necessary. Vulkan tests immutable inputs/canaries, disjoint spans, invalid-input
refusal, retry and bitwise repeat. Effective shader variants require four protected
basic operations, zero FMA and the established float-control transformation.
The tensor bank and component metadata each have a 64 KiB ceiling. Existing device
feature/refusal and strict arithmetic plumbing is reused. No state is committed.

Build the host control and applicable reference/component target after explicitly
regenerating the existing CMake build with unchanged options. Local CUDA compilation
and hardware execution must be reported separately; CPU and SPIR-V checks are not
CUDA qualification. The private operator recipe runs native first, then Vulkan
with the established resource/lifecycle/cleanup policy. `qualified=false` and
`model_integration=false` remain until a separately scoped integration is justified.
