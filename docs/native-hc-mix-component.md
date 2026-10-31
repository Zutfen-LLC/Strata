# Standalone T1 native HC down/up and mixing

This component extends the native-method investigation after PR98's partial
norm/injection seam. It is not called by `PortableModelSession`, adds no model
flag, and does not change any existing shader, native launch selection or default.
The existing norm/injection v1 contract remains unchanged. Hardware qualification
of this component is pending; `qualified=false` remains explicit.

The fixed geometry is HC4/H2560/LR320/T1, no pending write, normal-or-zero F32
normalized input and raw BF16 down/up rows. Conversion from an approved Q8 role
must use the original `iq_pack.py --compat-bf16` dequantize/RNE contract already
used by norm/injection. The synthetic fixture is versioned `HCMIX01` and contains
no model data. Other shapes, nonfinite/subnormal arithmetic, foreign/overlapping
spans, unsupported strict-FMA features and HC Q8/REQ8/V3 mode substitutions refuse.
Activation preinputs outside [-16,16] refuse; that is a bounded component domain,
not a proposed change to the native engine's supported model values.

## Original method and isolated operands

`verify.cpp:1329` binds BF16 down/up/injection and calls `fused_gr_read_multi`.
`cuda/fused_gr.cu` computes eight ordered FMAs per weight chunk, adds chunks in
each logical lane, then XOR-reduces offsets 16,8,4,2,1. Down divides its result
by four and applies `x/(1+__expf(-x))`. Up starts with its first chunk, adds a
second only in lanes 0..7, uses the same XOR, and gates the normalized residual
by `1/(1+__expf(-x))`. Mixing starts at +0, adds streams 0..3 and divides by four.
The Vulkan shader preserves those associations, using strict OpFmaKHR and
explicit shared-memory logical lanes rather than the physical subgroup width.

The standalone CUDA probe calls existing plain and staged down kernels plus the
existing T1 up kernel. It supplies the *same already normalized input* to down
and to up's residual path, with norm and RS exactly +1; it bypasses normalization.
It requires the selected staged variant3 and actual fixture plain/staged/repeat
bit equality. It does not replace any production launcher or reconstruct a
pending residual. CUDA compilation/execution and HIP compilation remain separate
unrun gates where that toolchain is unavailable. The probe's added entry point
is diagnostic only; no model execution calls it.

## Bounded arithmetic acceptance

`native_hc_mix_numeric.py` reuses independent integer-dyadic FP32 primitives,
not host float GEMM. Vulkan scaled down preactivation must match source-order
RNE dots exactly. Up must match source-order dots against that backend's actual
low output. Vulkan mixed must match exact ordered operations against its actual
gates, including zero signs. Thus an intrinsic allowance cannot hide a wrong
projection or summation association. Pinned controls cover dense/sparse lane,
chunk, row and stream boundaries and reject identical erroneous captures.

Exp and division are separate admitted operations. For input z in [-16,16], the
Vulkan exp interval uses `(3+2*abs(z))` local ULP against 100-digit mathematical
exp, following [GLSL precision](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#precision-and-precision-qualifiers).
Native `__expf` uses `2+floor(abs(1.173*z))` ULP against correctly rounded RNE,
following the [CUDA intrinsic table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
NVIDIA calls these maximum observed errors, not guarantees. These policies must
be checked on actual outputs; they are not universal hardware equivalence.
The 100-digit evaluation is an offline reference, not a software-exp runtime.

The denominator is one correctly rounded addition of actual exp and +1. Each
division must lie within 2.5 times the larger local ULP around its exact rational
quotient; all denominators here are normal. This uses GLSL's division allowance
conservatively for both backends, and does not assert NVIDIA needs that slack.
Vulkan records exp explicitly, so its activation is checked both against the
intrinsic interval and its *own actual exp*. Native kernels expose low/mixed;
their low is checked by interval propagation, and up/gate/mixed bounds use that
actual native low. Missing native preactivation/gate instrumentation is explicit:
those local native intermediates are source-derived bounds, not captured exact
checks. Cross-backend subtraction is a diagnostic, never a fitted epsilon.

Signed endpoint propagation through each stream product/add handles cancellation
and near-zero mixed values without a relative-error denominator. Exact local
Vulkan checks remain mandatory even where that propagated interval is wide.
No final-logit tolerance, full-HC, quality or full-model inference follows.

## Next gate

Build and run zero, sparse boundary and held-out dense synthetic fixtures through
both tiny probes sequentially, preserving exact fixture/source/binary/feature
bindings and existing resource/lifecycle guards. Independently check each backend
against its own intermediate arithmetic; retain exact cross-backend differences.
Do not integrate until those gates pass and remaining intrinsic discrepancies
are explained under this contract. Real layer-0 down/up role extraction remains
separately pending authorization; existing norm/injection operands do not cover
those bytes. Pending writes, FFN HC, T8, production integration and performance
remain separate work.
