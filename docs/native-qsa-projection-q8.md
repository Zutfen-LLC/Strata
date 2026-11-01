# Standalone QSA projection-input Q8_1 experiment

This standalone check covers the 6,144 float values produced by the T1 QSA
attention gate before the output projection. It neither implements the output
projection nor connects this check to model execution. The native oracle calls
`native_quantize_q8_1` from the existing `strata_kernels` target; it does not
copy or modify the production CUDA kernel. The Vulkan candidate calls the
existing mode 0 stage through its explicit RN observation owner. The candidate
changes neither the original shader nor the default path.

Each 32-value block has 36 bytes: little-endian binary16 scale and raw sum,
then 32 signed bytes. The fixed blocks independently pin positive and negative
zero, positive and negative halfway integer rounding, XOR32 sum order,
cancellation, both signs of half-sum ties, a half-scale tie and finite scale and
sum clamps. Every generated input starts with those fixed blocks. The rest of
its 192 blocks form zero, sparse, dense, order-sensitive or adversarial data.
Input length, nonfinite inputs, subnormal inputs, subnormal or nonfinite XOR
sums, scales and ratios are rejected. This is a conservative diagnostic domain;
it is not a claim about native treatment of all other numerical classes.

The CUDA source is configured with fast math; the existing RN candidate uses
explicit IEEE binary32 nearest-even division for scale and ratio. Previous
retained-input agreement does not establish equivalence for all values. The
binary32 operation used by the actual native build, including division and FTZ,
must be inspected before physical qualification. Preserve the actual CUDA
compile command and inspect the selected `native_quantize_q8_1_kernel` in the
existing native object. Do not change its flags or introduce a division
approximation. Native and Vulkan must each pass the fixed controls and all 6,912
bytes must agree exactly; any physical difference is a stop. No tolerance or
post hoc input filtering is allowed.

The host trace records amax, raw XOR sum, raw scale, clamped scale, ratio,
rounded float and clamped code as binary32 bits or integers for every lane. It
is a host RNE diagnostic; it is not a snapshot of native CUDA or Vulkan
intermediates. Both actual producers retain the input, packed bytes, repeat and
canary status. The Vulkan producer also retains its effective SPIR-V module,
and tests invalid shape and alias refusals before dispatch. Existing RN division
and effective-module controls remain required.

Five fixtures are generated locally using the compiled control:

| Mode | Input | SHA-256 |
| --- | --- | --- |
| 0 | Zero | `0d9098e4e6802a38c768751d0961966c21c17686bba0dfc52d2c021b7e949413` |
| 1 | Sparse | `a4edc6eb16a60b47a49e0f5d4b9dfbc9299c445b94a4737568d0cee19e8be5e0` |
| 2 | Dense | `f970f19c01da8e815227df7bc14129f14ac58c0df16ec85f44b66cf544c07688` |
| 3 | XOR order | `24add7ab099be5fde0072f44f68b2f6228ea967f165fa33b8f3135e42b14758a` |
| 4 | Adversarial | `4203c42967b515536d69ecafcae8e6fd5fd498b90011ecffbae0cdd1bb60b055` |

Generate with `native_qsa_projection_q8_input_test --synthetic MODE FILE` and
check the hashes before using the existing guarded native-then-Vulkan workflow.
Both producer binaries offer `--contract FIXTURE OUTPUT` without initializing a
backend. The output contract requires `qx.bin` of exactly 6,912 bytes. Compare
results with `tools/native_qsa_projection_q8_numeric.py --fixture FIXTURE
--fixture-sha256 FROZEN_HASH --validator CONTROL --validator-sha256 CONTROL_HASH
--native NATIVE_DIRECTORY --vulkan VULKAN_DIRECTORY`. The checker hashes the
fixture and validator and invokes the compiled control to validate a copy of the
exact fixture bytes before comparing outputs. Both actual producer exit
statuses, canary/repeat reports, effective module, fixture identity and guarded
runner lifecycle must also pass. Preserve raw outputs and failure receipts.

This component is experimental and `qualified=false`. A pass applies only to
these fixed inputs and the tested native compiler and Vulkan stack. It does not
establish WO matrix parity, integrated attention parity or model behavior.
