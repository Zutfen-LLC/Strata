# Native HC sigmoid tails

The native `sigmoidf_` in `src/kernels/cuda/fused_gr.cu` evaluates
`1 / (1 + __expf(-v))`. The Vulkan up stage retains that exp/add/divide expression.
It does not clamp v, assign saturated outputs, rescale production inputs or switch
to a numerically different sigmoid. Down SiLU, projections, reductions, norm,
pending/write gates and final mixed summation retain their existing contracts.

The previous symmetric normal-only domain was sufficient but excluded finite
inputs whose auxiliary exponential overflows or underflows even though the
sigmoid output remains finite. This extension admits two source-defined classes:

- For positive v above the normal-domain limit, exp must be nonnegative, finite
  and no greater than 2^-24. The actual rounded denominator must equal 1 and the
  actual quotient must equal 1 exactly. The mathematical exponential at the old
  limit is already below 2^-24 by many orders of magnitude. Auxiliary subnormals
  or underflow zero do not affect this exact rounded identity.
- For v <= -89, the actual exponential and denominator must both be positive
  infinity and the actual quotient must be positive zero. This is evaluation of
  the original operations, not an assigned zero. The natural integer threshold
  is ceil(log(FLT_MAX))=89. exp(89) exceeds FLT_MAX by more than 31%, far beyond the
  specified normal-range intrinsic error policies. It is not fitted to a model.

The [SPIR-V exp definition](https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html)
and [Vulkan floating-point rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#precision-and-operation-of-spir-v-instructions)
permit overflow infinities and underflow/FTZ behavior; overflow may also return
maximum finite. The [CUDA intrinsic reference](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
defines `__expf` as the native exponential approximation. We do not claim every
implementation returns the admitted auxiliary classes: the actual values must
satisfy the checks. Existing strict RTE and signed-zero/infinity/NaN controls are
required. Vulkan division's normal-denominator ULP bound is not extrapolated to
infinity; the admitted tail quotient is checked as an exact IEEE identity.

The finite-exp/subnormal-reciprocal transition between the negative normal limit
and -89 is still unqualified and explicitly refused. Nonfinite inputs, unexpected
NaNs, invalid auxiliary signs, invalid quotient identities and subnormal basic
projection/mixed arithmetic also refuse. This is a bounded capability extension,
not universal finite-domain or native hardware equivalence.

`native-hc-mix-sigmoid-tails-v3` retains the previous pointwise exp/division policies
inside the normal domain. Positive-tail auxiliary exp observations additionally
allow underflow zero when the admitted intrinsic interval reaches the subnormal
range; nonzero observations retain the primary ULP policy. The propagated gate
is exactly 1. The qualified negative overflow class requires actual +infinity and
propagates exactly +0. There is no relative epsilon near zero or final-output
logit tolerance. Normal native low inputs still control up-dot attribution.

A test-only preactivation entry in the same Vulkan shader invokes the production
activation function directly. The native probe calls the already existing
`fused_gr_pending_gate_probe`, which invokes the same `sigmoidf_`. It uses exact
power-of-two input/output scaling, verifies round-trip bits, and observes exp and
sigmoid without another numerical implementation. Its supported test inputs must
survive that scaling exactly; it is not a production method. Both probes retain
canaries, immutable inputs and repeated outputs. No new CMake target is required.

CPU independent boundary/adversarial tests and strict effective-module checks
are distinct from physical qualification. The tiny matched-preactivation native
and Vulkan gate must run before another model attempt. No defaults, model
placement, memory caps/reserves, scheduling or state lifetime change here.
