# Standalone native K10 Vulkan combine

This explicit strict-FMA diagnostic component computes one missing boundary:
original-order routed expert rows and routing weights, followed by the already
gated shared-expert output. It does not run a model by itself. An explicit FFN/model option connects it
to the existing expert-row boundary (see below). Default execution, routing, placement, cache capacities
and all earlier comparison failures stay unchanged.

The component is compiled only with `STRATA_VULKAN_NATIVE_STAGES=ON` and
`STRATA_VULKAN_NATIVE_Q8_STRICT_FMA=ON`. Calling the standalone executable or setting the explicit FFN/model flag is
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

The default portable host combine remains in
[`PortableFfn::step` / `step_batch`](../src/core/portable_ffn.cpp). The opt-in
integration supplies the complete original router-order rows, including
native CPU misses, and the existing gated shared row. The grouped provider's
packed order is not the combine's input order. The existing provider owns
that handoff; the connection preserves its ownership.

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
through the pinned helper's FP16 metadata insertions. The standalone GPU Q8
provider replay does not call this component; the opt-in FFN caller does.

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

## Opt-in FFN / model connection

`PortableFfnOptions::native_moe_combine` now connects the same component to
`PortableFfn::step` and `step_batch`. The model exposes it as
`--native-moe-combine`, requiring `--expert-native-q8-observation`, explicit
expert plans and strict accelerator execution. Both flags remain off by default.
The strict-FMA build and its feature receipt are mandatory; an unsupported build
or device refuses the request. This remains an unqualified observation.

The pre graph still computes `down(silu(gate(mixed))*up(mixed))` and multiplies
it by `sigmoid(shared_gate(mixed))` exactly once. That already gated row is
passed to the combine, which performs no gating. `LocalExperts::run/run_batch`
restores each indexed GPU reply to its original router slot and fills cold
slots with the existing native CPU expert pool. The complete token-major,
rank-order host rows and corresponding weights are uploaded without sorting.
The component result directly supplies `post_block`; the host weighted sum is
skipped. The existing post graph applies HC injection/residual as before.

This deliberately retains the existing host expert-row boundary and gated
shared-row readback. It adds no placement or scheduler policy. A separate
bounded staging buffer and component owner belong to each FFN execution shape.
`PortableFfnOptions::native_moe_combine_receipt_prefix` is mandatory. The model
uses `RECEIPT_PREFIX.combine` from its native-Q8 flag; the FFN adds `.layer-L`
and atomically reserves an unused `.t-T.owner-N` directory, storing the normal
component receipt as `effective.spv` inside it. Existing directories are skipped,
including after restart or replanning, and old receipts remain untouched.
The parent directory must already exist and be writable. Empty prefixes,
filesystem failures or namespace exhaustion refuse construction. No fallback
or change to the component's receipt/validation contract is introduced.
Their actual aligned bytes count toward the existing aggregate graph cap and
`device_graph`; a 16 KiB metadata context counts toward the host bound. Shared
T1/chunk weights still count once, while both combine buffers count together.
Failure leaves the FFN residual/routes unpublished; the enclosing session's
existing poison behavior applies. No CPU/GPU expert cache capacity changes.

`portable_ffn_native_combine_test` compiles the real FFN coordinator against a
mocked component boundary. T1/T2/T8 checks cover exact row/weight uploads,
nontrivial ranked IDs, token association, the gated shared row, one consumption
by the post graph, absence of a redundant host sum, unchanged legacy selection,
budget/shape/fallback refusal, chunk accounting and exception/nonfinite rollback.
The component mock enforces the real mandatory nonempty shader-receipt contract;
with this check the previous null-path connection fails before dispatch.
This is a host integration regression, not a GPU arithmetic result.

The explicit-device executable runs the same synthetic FFN fixture with the
real component, pre graph and post graph:

```sh
cmake --build "$BUILD" --target portable_model portable_ffn_native_combine_vulkan
# Use the approved private ICD / SDK-loader environment from the rootless
# prerequisite recipe. INDEX and DESCRIPTION must identify the intended device.
timeout 60s "$BUILD/portable_ffn_native_combine_vulkan" \
  --device "$INDEX" --expect-device "$DESCRIPTION" \
  --receipt-prefix "$NEW_PRIVATE_FFN_RECEIPT_PREFIX"
```

It checks finite T1/T2/T8 FFN residuals with a fixed 4e-6 absolute integration
tolerance against the simple synthetic shared/injection graph. That tolerance
covers graph arithmetic; it is not a raw-bit native/CUDA qualification. The
standalone paired K10 fixture remains the raw-bit arithmetic check. This FFN
executable performs no attention, routing-plan creation or full model run.

For one T1 full 48-layer forward, use the existing frozen plan and ordinary
model CLI, adding only the two observation flags. For example, in the approved
private process environment (all paths/caps/IDs below are supplied evidence):

```sh
timeout "$FORWARD_TIMEOUT_SECONDS" "$BUILD/portable_model" \
  --model "$ANY_OF_SIX_SHARDS" --manifest "$LOCAL_MANIFEST" \
  --tokens "$ONE_TOKEN_ID" --max-new 0 --context 1 \
  --threads "$THREADS" --expert-workers "$EXPERT_WORKERS" \
  --vulkan "$DENSE_DEVICE_INDEX" --device-cap-mib "$DEVICE_CAP_MIB" \
  --host-cap-mib "$HOST_CAP_MIB" \
  --expert-plan "0:$FROZEN_PLAN" --expert-tensor-mib "$EXPERT_TENSOR_MIB" \
  --expert-workspace-mib "$EXPERT_WORKSPACE_MIB" \
  --expert-free-reserve-mib "$EXPERT_FREE_RESERVE_MIB" \
  --expert-native-q8-observation "$NEW_PRIVATE_RECEIPT_PREFIX" \
  --native-moe-combine --expert-record-routes \
  --placements "$NEW_PRIVATE_PLACEMENTS" --dump-logits "$NEW_PRIVATE_LOGITS_DIR"
```

Retain the existing approved dense topology/residency flags if required by its
admission recipe; do not silently change them or add CPU fallback. Capture the
actual child exit directly, enforce the timeout, and keep all payloads private.
This command is a recipe, not a locally executed or qualified result.

Static assessment: a legal frozen plan has 1–24 total assigned assets per
native-Q8 executor. It need not cover every expert in all 48 layers: an empty
hit mask skips GPU dispatch and existing CPU misses fill the original slots.
For T1/K10, each GPU request has at most 10 entries/groups, so the existing
24-group/64-entry bounds cannot block that request. A plan containing more than
24 assets still refuses preparation; no eviction, reassignment or call splitting
is introduced. The replay tool's all-active-resident requirement is specific to
standalone replay, not the model's CPU-miss path.

After a physical FFN check, the remaining execution prerequisites are the exact
six-shard Q8 model/manifest and frozen plan identities, strict-FMA support on
every selected Vulkan device, an admitted dense/state/head/expert memory ledger
including these buffers, and a bounded T1 forward using the unchanged CPU miss
pool. No known source-level combine or resident-capacity change is needed for
that bounded invocation. Full-forward correctness remains unverified: shared,
router/HC, attention/GDN/QSA, CPU-miss boundaries and final logits still need
comparison against a matching native forward. The eight retained expert records
and synthetic combine checks do not establish those boundaries or 48-layer
qualification. Preserve the earlier unused Q8 sum-field representation failure.
