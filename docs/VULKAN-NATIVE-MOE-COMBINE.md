# Standalone native K10 Vulkan combine

This explicit strict-FMA diagnostic component computes one missing boundary:
original-order routed expert rows and routing weights, followed by the already
gated shared-expert output. It does not run a model or replace the portable
FFN combine. Default portable execution, routing, placement, cache capacities
and all earlier comparison failures stay unchanged.

The component is compiled only with `STRATA_VULKAN_NATIVE_STAGES=ON` and
`STRATA_VULKAN_NATIVE_Q8_STRICT_FMA=ON`. Calling the standalone executable is
its runtime opt-in. It retains `observation_only=true`, `qualified=false`,
`quantizer_qualified=false` and unspecified FP32 denorm mode.

## Source and reuse

The native specification is `combine_k10_vec4` in
[`src/kernels/cuda/native_moe.cu`](../src/kernels/cuda/native_moe.cu), selected by
`native_moe_combine` / `native_moe_combine_multi` for K10, a non-null shared row
and aligned vec4 spans. It performs a separately rounded first product, then
one FMA for experts 1 through 9 in router order, then a separate shared add.
There is no tree reduction, renormalization, extra shared gating, sorting or
atomic accumulation. Four FP32 lanes retain independent arithmetic.

The shader [`native_moe_combine.comp`](../src/kernels/vulkan/native_moe_combine.comp)
uses a 128-invocation workgroup like that native branch. First multiply and final
add are `precise`; the intervening operations use strict `OpFmaKHR`. The
existing feature request/creation receipt, float-control handling, final-module
validator, same-device tensor checks and pipeline/resource owner are reused.
Unsupported strict FMA refuses device admission; there is no ordinary-FMA or
software fallback. The reused owner has the existing small metadata allocation,
which is reported even though this component needs no group metadata.

[PR80](https://github.com/Zutfen-LLC/Strata/pull/80), checked at
`fe1a9baa7450d2ab8b54257c20ffa7b59d11a43c`, combines QSA attention gating,
projection and HC residual expressions in an active attention graph. That is
a different operation; it contains no reusable native K10 expert weighted
reduction. This slice reuses existing component/view ownership and admission,
without importing that independent branch or duplicating an attention graph.

The current portable host combine remains in
[`PortableFfn::step` / `step_batch`](../src/core/portable_ffn.cpp). A later
integration must supply the complete original router-order rows, including
native CPU misses, and the existing gated shared row. The grouped provider's
packed order is not the combine's input order. This component does not invent
that handoff or change its ownership.

## Interface and bounds

The API is in
[`strata_vk_native_moe_combine.hpp`](../src/core/strata_vk_native_moe_combine.hpp).
Parts are contiguous F32 `[T][10][2560]`, weights `[T][10]`, shared and output
`[T][2560]`, with T1–8. Parts/shared/output views must be 16-byte aligned;
weights require 4-byte alignment. All tensors belong to the same selected
Vulkan backend. Output is disjoint from every input; inputs are read-only.
Tensor view biases are applied in the shader in the correct vec4/scalar units.
No resident/group/entry capacity or byte-address limit is widened.

Configuration and every shader rebuild compile and validate strict SPIR-V for
Vulkan 1.2. Effective-module validation preserves every input instruction
through the pinned helper's FP16 metadata insertions. The existing GPU Q8
provider and replay do not call this component yet.

## Fixture and checks

Retained expert captures contain indexed output rows and routing data but lack
the already gated shared-expert row required for a complete native combine
oracle. No real inputs are reconstructed or published. The bounded fixture is
therefore synthetic: T2/H2560/K10, finite normal/zero inputs, signed zero,
zero weights, distinct lane data and four arithmetic-order witnesses.

`native_moe_combine_input_test` checks exact RNE host results using a rounded
first product, `std::fma` for experts 1–9 and a rounded shared add. Deliberately
fusing the wrong product, separating multiply/add, reversing expert order, or
adding shared early all change fixture output bits. This host arithmetic is
an oracle only, not a software runtime FMA path.

`native_moe_combine_shader_test` checks the actual SPIR-V dataflow: first-product
seed, ordered 1..9 loop, strict FMA carry and shared add after the loop. It
checks four FP32 lanes, contraction controls, absence of legacy Fma/fast-math
and every pinned FP16 helper combination, with an FMA-operand mutation negative
control. `native_moe_combine_receipt_test` checks exact bits, output bit-flip
failure, immutable inputs, guard sentinels, binary fixture roundtrip, overwrite
refusal and trailing-data refusal without a GPU.

The Vulkan runner performs one combine dispatch on offset tensor views and
retains every output word, expected host word, complete padded input/output
views, effective SPIR-V and a JSON report. It checks all input words and both
output guards. The native reference runner calls the existing
`native_moe_combine_multi` launcher; it does not copy or replace the CUDA kernel.
It uses the identical fixture and padded views. A native comparison is still
required before claiming hardware equivalence.

## Build and separately authorized component recipe

Use the existing pinned source and verified build configurations. The Vulkan
build needs the same strict-FMA SDK/ICD support as the existing native-Q8
components. No driver install or fleet execution is part of this change.

```sh
set -euo pipefail
cmake --build VULKAN_BUILD --target native_moe_combine_component \
  native_moe_combine_input_test native_moe_combine_receipt_test \
  native_moe_combine_shader_test
ctest --test-dir VULKAN_BUILD --output-on-failure \
  -R '^(native_moe_combine_.*_test|native_q8_.*strict_fma.*_test|portable_logits_guard_build_test)$'
```

The generated T2 fixture is 225,380 bytes; its SHA256 is
`424335ee303b882b1fa3efcb22cc9d858010f0d926c71d1c8f2aeaf8d87a234d`. A changed fixture must not be mixed into that comparison.

The native reference requires a CUDA/HIP-capable build of the existing kernel
library; it does not require Vulkan:

```sh
set -euo pipefail
cmake --build NATIVE_BUILD --target native_moe_combine_reference \
  native_moe_combine_input_test
```

After separate execution approval, create one new fixture with the host test,
verify its SHA256 on both hosts, and use that exact file in both commands. Use
existing verified device identity and process-local ICD/library settings; keep
the external timeout and record the command's actual status, including through
any logging wrapper. These are one-component probes, not full-model runs.

```sh
set -euo pipefail
VULKAN_BUILD/native_moe_combine_input_test --write-fixture NEW_FIXTURE
sha256sum NEW_FIXTURE
# Native host, after the same fixture hash is verified:
timeout 60s NATIVE_BUILD/native_moe_combine_reference \
  --device VERIFIED_NATIVE_INDEX --expect-device VERIFIED_NATIVE_DESCRIPTION \
  --fixture VERIFIED_FIXTURE --output NEW_NATIVE_PREFIX
# Vulkan host under the already verified private ICD environment:
timeout 60s VULKAN_BUILD/native_moe_combine_component \
  --device VERIFIED_VULKAN_INDEX --expect-device VERIFIED_VULKAN_DESCRIPTION \
  --fixture VERIFIED_FIXTURE --output NEW_VULKAN_PREFIX
spirv-val --target-env vulkan1.2 NEW_VULKAN_PREFIX.effective.spv
cmp NEW_NATIVE_PREFIX.output.f32 NEW_VULKAN_PREFIX.output.f32
```

Each runner refuses existing result paths and exits 1 for different output bits,
nonfinite output or a sentinel/input mutation. Raw receipts remain on failure.
Read the JSON as well: it must report all 5,120 words, zero differing bits,
zero nonfinite words and `sentinels_ok=true`. Preserve exact CUDA/Vulkan module,
executable, fixture and output hashes. Do not edit an earlier failure or infer
qualification, end-to-end parity or performance from this fixture.
