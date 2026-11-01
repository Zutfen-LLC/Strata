# Experimental native T1 GDN convolution/L2

This standalone component ports `Verifier::record_window()` →
`gdn_conv_l2_multi()` / `gdn_conv_l2_body()` in
`src/kernels/cuda/verify_kernels.cu`, for C10240, S128, 32 q/k heads, T1,
`t_begin=0` and epsilon `1e-6f`. No model option or default path changes.
The separate `gdn_ab_multi()` BF16 projections/sigmoid/softplus are not ported by
this slice. Native CUDA uses these separate launchers: `fuse_convab_on()` is
false outside HIP. This is not the HIP opt-in fused conv/a-b variant.

Native history is three values per channel. Four convolution taps feed the
original SiLU `sum / (1 + __expf(-sum))`. For q/k heads only, y*y reduces by
XOR16/8/4/2/1 within each logical warp and a left-associated four-warp sum;
`rsqrt(sum + eps)` scales y. The remaining v heads bypass normalization.
Verify leaves history unchanged; the one-token commit shifts history to
`[hist1,hist2,qkv]`. Vulkan reads immutable history and produces this candidate
in a separate admitted buffer. Only a future existing layer transaction may
commit it; the component does not create new state authority or scheduling.

The convolution expression's CUDA contraction is a qualification condition.
The explicit candidate rounds the second product first, then uses FMA for the
first product/add and each later tap. Independent pinned host vectors and strict
Vulkan operations describe this candidate. The reference invokes the unchanged
production CUDA launcher and requires its actual h/history to reproduce exactly
from its own exp/rsqrt controls. If that fails, preserve outputs and inspect
compiled lowering; do not change compiler flags, math or tolerance to force a pass.
Native diagnostic pre/SiLU/argument fields are source-control intermediates, not
instrumented values copied out of the production kernel. Their interpretation
requires the actual-launcher output check to pass.

The exact local control checks convolution association, SiLU using the actual
exp value, reduction/epsilon, output using the actual rsqrt, and history commit.
A separate primitive validator uses the existing primary-spec exp/division and
2.5-local-ULP RNE-relative rsqrt policy (including NVIDIA's conservative 0.5-ULP
policy slack). No blanket h tolerance or near-zero relative error is used.
SiLU positive tails require the unchanged denominator to round to one and output
to equal the numerator. Negative overflow tails require actual +Inf exp and
negative-zero division output. No clamp, saturation assignment, FTZ
execution mode or alternate sigmoid is introduced. Normal/zero inputs and basic
results are supported; subnormal arithmetic and the negative transition between
-89 and the existing derived normal limit explicitly refuse. Unsupported
geometry/modes are not approximated. Cross-backend h differences are diagnostic.

The API uses 11 descriptor slots, a 128-invocation workgroup and 512 bytes of
shared storage; existing logical strict-FMA and FP32 controls remain mandatory.
Exact contiguous same-device disjoint spans are checked, including history versus
candidate. The probe admits at most 2 MiB of tensor buffers plus bounded component
metadata. It checks immutable inputs/canaries, verify/commit, reset/repeat, invalid
input noncommit and retry. Synthetic zero/sparse/dense fixtures are generated
locally by the pinned host control; the handoff verifies their exact hashes.
There is no new raw weight extraction, model capture or full-model execution.

The smallest remaining a/b dependency is faithful `log1pf(__expf(v))` for native
softplus when v<=20. Vulkan GLSL provides log, not log1p. Silently using rounded
`log(1+x)` would lose the native near-zero behavior. A separately justified
backend intrinsic adapter and its own native comparison are needed before a/b
or full-prefix/model integration can be claimed. Existing GGML a/b remains
unchanged; this component is not complete GDN prefix or native window parity.

Sources: [GLSL standard functions](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html),
[CUDA intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html),
[CUDA mathematical-function accuracy](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html),
and [Vulkan floating point](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).
