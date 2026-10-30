# Standalone native Q8_1 GPU stages

This opt-in component translates the native GPU quantizer and grouped SwiGLU
stages. It is not used by the portable model or expert executor. It does not
change placement, limits, resident caches, scheduling or third-party sources.
The GGML pin stays `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

## Native mapping

The specification is `src/kernels/cuda/iq_kernels.cu`, not generic GGML Vulkan
quantization or Intel translation:

| Native source | Standalone component |
| --- | --- |
| `q8_1_store`, `quantize_q8_1_kernel` | mode 0; one 32-invocation workgroup per native block |
| `swiglu_entries_kernel`, then `quantize_q8_1_kernel` | mode 1 over all capacity, barrier, mode 0 on hidden F32 |
| `swiglu_q8_1_entries_kernel` | mode 2 over `[grp_start[0], grp_start[n_groups])`; other packed blocks untouched |
| `q8_1_finite.hpp` | clamp scale and raw sum to signed 65504, leave NaNs as NaNs; nearest-away codes clamped to [-127,127] |
| native `ds` half2, then 32 int8 codes | nine little-endian uint32 words per block; integer IEEE half RNE conversion |

The sum is of raw F32 values. Maximum and sum use the exact XOR offsets
16, 8, 4, 2, 1 with workgroup barriers, independent of physical subgroup width.
Scale uses division by 127; codes use division by the unclipped-to-half F32
scale after finite clamping. It does not derive the sum from quantized codes,
multiply by a precomputed reciprocal, or quantize with a rounded half scale.
The hidden multiply and all reduction adds have `NoContraction` decorations.
The fused path leaves hidden F32 storage untouched, like native.

The native separate path covers **all capacity**, including unused entries;
the fused path covers only the active interval. Empty groups retain their
metadata meaning; an empty call writes nothing in the fused path. Full rows
must be multiples of 32; partial blocks are rejected, not padded. A final
partial 256-thread CUDA launch is exercised by the 832-element fixture.
Native gfx906 defaults to separate passes; this component exposes both paths
without choosing a new production default. Hadamard rotation is outside this
slice; no claim is made for that separate native branch.

The existing pin-checked Vulkan translation-unit bridge supplies device,
queue, descriptors, buffers and cleanup. Group starts/count are bounded,
validated and uploaded into owned metadata, replacing native device pointers.
Tensor views include their byte offsets. Explicit barriers and synchronization
replace stream dependencies. One caller owns the component. This synchronous
standalone interface is not a proposed model scheduler.

## Arithmetic qualification limits

The device must support FP32 denormal preservation, signed-zero/Inf/NaN
preservation and RTE rounding. All three execution modes are added to SPIR-V;
creation fails if their properties are absent. Half conversion uses integer
bits, preserving half subnormals and controlling ties independently of a
driver's `packHalf2x16` implementation.

These controls and `precise` do **not** promise correctly rounded division or
CUDA `__expf` equivalence. Vulkan `exp` has a different accuracy contract and
may differ near underflow/overflow, subnormal inputs and ordinary finite
values. Native CUDA's intrinsic also has its own exceptional-value/flush
behavior. Cross-backend division can differ at quantization ties. The generated
native oracle and exact comparator must reveal the actual affected inputs.
No GPU comparison has yet established a qualified numerical domain.

Native `q8_1_quant` casts a NaN result to int8 without a portable result
contract. This occurs for NaN inputs and can occur when a nonzero subnormal
block's scale underflows to zero. The diagnostic shader writes code zero for
that conversion only; this is an explicit unqualified domain, not a claimed
native fix. Native NaNs remain NaNs in half metadata; payload/sign conversion
may differ. Infinite values are finite-clamped as native specifies, with
opposite infinities producing a NaN raw sum. The comparator reports **all**
byte differences, including these cases, and marks these contract holes explicitly. It does not waive
NaN payloads, code differences or activation error with a tolerance.

The component is not ready for product integration. Additional grouped gate/up and down branches,
`native_moe_combine`, device/CPU overlap and native execution policy remain
separate parity gaps. The earlier native `--native --spec2` control does not
establish which optional fused kernels ran.

## Build and host checks

Use the repository's unchanged pinned source or the normal pinned fetch.
Existing Vulkan SDK, SPIRV-Headers and CMake dependencies are required; the
component adds no runtime dependency. Replace source and device placeholders
with verified local values.

```sh
cmake -S . -B build-stages -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON -DGGML_VULKAN=ON \
  -DSTRATA_VULKAN_NATIVE_STAGES=ON -DSTRATA_GGML_DIR=PINNED_LLAMA_SOURCE
cmake --build build-stages --target native_q8_vulkan_component \
  native_q8_spec_test native_q8_shader_test -j4
ctest --test-dir build-stages -R '^(native_q8_.*|portable_logits_guard_build_test)$' --output-on-failure
build-stages/native_q8_shader_test controlled.spv
spirv-val --target-env vulkan1.2 controlled.spv
```

Host tests cover signed zero, tie rounding, cancellation with a raw XOR sum,
half finite-clamp boundaries, subnormal half ties and all 63,488 finite half
bit patterns. Shader checks require three `OpFDiv` operations, contraction
controls, explicit FP32 execution modes, barriers and no physical-subgroup
operations. Oracle extraction and exact-comparator negative controls are
separate host tests. Passing them or compiling Vulkan is not GPU proof.

## Bounded native oracle and Vulkan capture

Generate one synthetic fixture, then supply **that same file** to both runners.
The fixture includes zero/signed zero, rounding ties, cancellation, subnormals,
finite clamp boundaries, infinities, NaNs, nonzero active-interval start,
empty interior group, inactive prefix/suffix, empty call and launch tail.
The Vulkan runner also uses offset tensor views and checks their sentinels.
Each runner captures input Q8_1, separate hidden F32, separate Q8_1, fused Q8_1
and empty-interval output. Native definitions are extracted verbatim from this
checkout with the native source hash, rather than reimplemented as an oracle.

```sh
build-stages/native_q8_vulkan_component --make-fixture synthetic.bin
python tools/native_q8_oracle.py --generate native_q8_oracle.cu
nvcc -std=c++17 -O3 -arch=sm_86 -Iinclude -Ithird_party/ggml \
  -Itests/kernels native_q8_oracle.cu -o native_q8_oracle
# Replace architecture and indices from verified hardware; no fast-math flags.
timeout 30s ./native_q8_oracle --device CUDA_INDEX --expect-device CUDA_DEVICE_SUBSTRING \
  --fixture synthetic.bin --output native.bin
timeout 30s build-stages/native_q8_vulkan_component --device VK_INDEX \
  --expect-device DEVICE_SUBSTRING --fixture synthetic.bin --output vulkan.bin
python tools/compare_native_q8_stages.py --fixture synthetic.bin \
  --native native.bin --vulkan vulkan.bin --report comparison.json
```

Record `git rev-parse HEAD` and the SHA-256 hashes of the fixture, generated
oracle source and both binaries with each capture.

This is a small component capture, not a model run or benchmark. The comparator
requires exact packed bytes and hidden F32 bits, reports each differing byte,
block, metadata/code field and hidden input/output bit pattern, and compares
separate/fused active regions **within each backend**. Exit zero means exact
agreement on a fixture without identified native conversion holes; any byte
difference returns one, and exact bytes with native NaN/conversion holes return
two. The report always sets product qualification to false. Preserve
nonzero results for arithmetic diagnosis. Never integrate based on a host or
software-Vulkan substitute for native GPU evidence.

## Property-gate diagnosis

The property gate admits the SPIR-V emitted by this component; it is not a
native Strata capability check. A refusal does not establish that native
Strata arithmetic is impossible on that GPU.

The borrowed physical device is queried with Vulkan-Hpp's
`getProperties2<PhysicalDeviceProperties2, PhysicalDeviceVulkan12Properties>`.
Vulkan-Hpp constructs and links the chain before the query. Pinned GGML queries
the same core 1.2 property struct. These are physical-device properties, not
logical-device features to enable. A host mock exercises that exact overload,
physical handle and returned chain for all eight combinations of the required
booleans. This checks plumbing, not a real driver's response.

Failures report each required raw FP32 property, FTZ/RTZ support, independence
enums, the FP16 controls used by the pinned helper, and device/API/driver
identity from the same query. The conjunction and shader stay unchanged.
The earlier generic error cannot reveal which required property was false.

Native grouped `iq_kernels.cu` is absent from CMake's per-file fast-math list;
`native_mmvq.cu` is on that list and is a different path. The supplied oracle
command likewise has no fast-math option. CUDA 13.1 defaults preserve ordinary
F32 subnormals and use precise division, while contraction is enabled.
The native fused hidden product explicitly uses `__fmul_rn`; HIP disables
contraction there. [NVCC 13.1 options](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-compiler-driver-nvcc/index.html#options-for-steering-cuda-compilation)
explain these defaults. Exact remote build overrides still require receipts.

Activation explicitly calls approximate `__expf`, rather than standard `expf`.
Neither the ordinary arithmetic defaults nor the new property gate establish
its subnormal behavior or bit equality with Vulkan `Exp`.
[CUDA 13.1 intrinsic documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
identifies it as approximate. Inspect the retained oracle's generated
instructions to distinguish actual FTZ handling; do not infer the intrinsic's
mode from `--ftz=false` alone.

The [Vulkan SPIR-V environment](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)
forbids using the declared controls when their corresponding properties are
false. Its `DenormPreserve` instruction list includes add/multiply but excludes
division and `Exp`; division has its own accuracy allowance. Thus the current
controls are insufficient for full native arithmetic parity even when admitted.

The next discriminator is the actual raw property tuple, preferably checked
against an independent properties-only query, plus retained native compile and
PTX/SASS evidence. If a required property is genuinely false, the current
module cannot legally be dispatched. A narrower diagnostic domain would need
operation-by-operation justification, including intermediates; finite input
values alone do not suffice. No gate relaxation, shader change or fleet retry
is authorized by this diagnosis.

## Explicit observation of unspecified FP32 denormals

`--observe-unspecified-fp32-denorm` is a separate diagnostic identity. It never
runs automatically after strict refusal. The strict shader, creation function,
property predicate and default capture encoding remain unchanged. There is no
model integration, arithmetic emulation, domain filtering or policy selection.

Observation derives from the same compiled shader by removing exactly one
four-word instruction: `OpExecutionMode ... DenormPreserve 32`. All arithmetic,
other controls, reductions, packing, intervals and barriers are retained. It
still requires supported FP32 RTE and signed-zero/Inf/NaN modes, support for
the retained capabilities, and independence for the helper's FP16 controls.
It requests neither FP32 preservation nor flushing. Missing preservation is
not reinterpreted as permission to force FTZ.

The exact pin-checked pipeline helper is copied into the build tree for this
entry point only. Its sole added hook validates and writes the final SPIR-V
immediately before shader-module creation, after all helper modifications.
The original helper remains untouched. Validation requires the input
instruction stream to survive byte-for-byte, allowing only the known helper
FP16 metadata insertions with exact counts. Any arithmetic/decoration change
is rejected before creating the shader module.

Host tests run the actual pinned FP16 transformation block with all four
helper-control combinations. For each, effective strict and observation
modules differ by exactly the omitted FP32 mode. They also reject changed
instructions and unsupported remaining controls. To emit and independently
validate those host-generated effective modules:

```sh
build-stages/native_q8_shader_test --observation-modules observation-host
spirv-val --target-env vulkan1.2 observation-host.effective-0.spv
spirv-val --target-env vulkan1.2 observation-host.effective-1.spv
spirv-val --target-env vulkan1.2 observation-host.effective-2.spv
spirv-val --target-env vulkan1.2 observation-host.effective-3.spv
```

For a separately reviewed GPU observation, reuse the retained original fixture
and native CUDA capture. Replace paths and device placeholders from verified
identity; choose fresh output paths. No new CUDA run is needed.

```sh
timeout 30s build-stages/native_q8_vulkan_component \
  --observe-unspecified-fp32-denorm \
  --device VK_INDEX --expect-device DEVICE_SUBSTRING \
  --fixture RETAINED_SYNTHETIC_BIN --output FRESH_OBSERVATION_BIN
spirv-val --target-env vulkan1.2 FRESH_OBSERVATION_BIN.effective.spv
python tools/compare_native_q8_stages.py --fixture RETAINED_SYNTHETIC_BIN \
  --native RETAINED_NATIVE_BIN --vulkan FRESH_OBSERVATION_BIN \
  --report FRESH_COMPARISON_JSON
```

The capture has a distinct binary magic and carries its sentinel status, so
an old comparator rejects it and the updated comparator detects observation
automatically. Its reports always say `qualified=false` and
`fp32_denorm_mode=unspecified`. Raw bytes and native conversion holes remain
visible. Comparator exit **1** means a mismatch or sentinel failure; **2**
means exact observed bytes that remain unqualified. Observation never returns
a comparator strict-pass status. Capture process exit zero only means the
capture completed.

The effective module is saved as `OUTPUT.effective.spv`. `OUTPUT.views.bin`
contains final raw base-buffer receipts, including prefix/suffix canaries:
magic `0x3156384e`, then five little-endian length-prefixed byte arrays in
input/gate/up/hidden/packed order. Observation retains stage and final view
receipts even when its sentinel checks fail, then returns a failure. Full
original fixture cases and inactive-output sentinels are preserved.

## Paired native Q8_0 consumer diagnostic

`native_q8_consumer` tests the numerical consequence of the specific block-5
code mismatch. It does not test a new Vulkan operation and does not waive the
original byte failure. Use the hash-checking launcher, rather than invoking
the CUDA executable directly.

The launcher checks SHA-256 values against the **retained evidence receipts**,
then snapshots the full original fixture and both captures into a fresh output
directory. Use the observation **capture's** hash, not the effective-SPIR-V
hash. It requires the original 832-element fixture shape and subnormal block,
correct native/observation encodings, intact sentinels, exactly the 32 expected
input-code differences, zero block-5 metadata, and identical other sections.
It saves the complete original comparison, still failed and unqualified.

The executable independently checks capture structure and the prescribed
mismatch before any CUDA call. It feeds the first eight Q8_1 blocks of each
capture directly through `x_q8_1`, without input requantization. The common
signal blocks stay untouched. One finite synthetic Q8_0 expert and one group
serve two entries, with `n_embd=256` and `n_ff=32`, through the unchanged
`native_expert_grouped` entry point. These are its minimum supported dimensions.
There is no new kernel, dequantization simulation or forced native variant.
The original full-fixture report remains separate from this causal check.

The weight blob is 26,112 bytes: signed int8 codes restricted to +/-1 and
half scales restricted to +2^-11 or -0. Every scale is checked finite. Positive,
negative, alternating and zero-scale rows exercise signs and retain nonzero
common signal. Native gate/up and final output arrays must be finite and
nondegenerate. Their raw F32 bits are compared entry-by-entry, including zero
signs, with no tolerance. Gate/up are materialized by this Q8/dimension path;
unwritten scratch remains NaN and fails the finite check.

Build against the existing native CUDA Release configuration and unchanged
kernel/toolchain settings. This target is CUDA-only; no HIP/SYCL support is
implied. The CPU parser/weight test needs no CUDA:

```sh
cmake --build CUDA_BUILD --target native_q8_consumer -j4
cmake --build HOST_BUILD --target native_q8_consumer_input_test -j4
ctest --test-dir HOST_BUILD -R '^native_q8_consumer_input_test$' --output-on-failure
```

For one separately reviewed bounded run, replace all paths, hashes and device
placeholders with verified retained values. The fresh output directory must
not exist; its parent must exist. The launcher records the launched child PID
and collects its actual return code directly, with a 30-second deadline.

```sh
python tools/native_q8_consumer.py --consumer CUDA_BUILD/native_q8_consumer \
  --device CUDA_INDEX --expect-device DEVICE_SUBSTRING \
  --fixture RETAINED_FIXTURE --fixture-sha256 EXPECTED_FIXTURE_SHA256 \
  --native RETAINED_NATIVE_CAPTURE --native-sha256 EXPECTED_NATIVE_SHA256 \
  --observation RETAINED_OBSERVATION_CAPTURE \
  --observation-sha256 EXPECTED_OBSERVATION_CAPTURE_SHA256 \
  --output FRESH_OUTPUT_DIRECTORY
```

`receipt.json` preserves source revision, executable/input/output/weight hashes,
argv, relevant native environment flags, child identity and exit/timeout state.
`consumer.log` records device/PCI identity, per-stage mismatch counts and exact
bits for differing rows. `native-consumer.raw` stores magic `0x3143384e`, then
three little-endian count-prefixed F32-bit arrays: gate (64), up (64), output
(512), first native-input entry then observation-input entry. The synthetic
weight bytes are saved alongside it. Raw results are saved before comparisons.

A consumer exit zero means only that these paired native row bits match, with
finite weights and common nonzero signal. The receipt still says
`qualified=false`, `fp32_denorm_mode=unspecified` (referring to the retained
observation), and `original_comparator_exit=1`. It does not establish packed-byte
parity, a Vulkan guarantee, other-format behavior or product qualification.

## Standalone Q8_0 grouped gate/up consumer

`native_q8_gu_component` ports only `native_gu_kernel<8>`: the per-entry,
full-warp branch selected by `launch_gu<8>` when `STRATA_KQ_SPLIT` is unset.
This selection is source-derived, not a dynamic launch trace. The component
consumes the retained paired native/observation Q8_1 inputs directly. It does
not run quantization, activation, down, combine, or the product grouped path.

The mapping to `src/kernels/cuda/iq_kernels.cu` is:

- `vec_dot_q8_0_q8_1` (lines 569–577): two signed four-byte integer dots;
  decode the weight half scale and the low half of Q8_1 `ds`.
- `Fmt<8>` and `row_dot` (lines 738, 773–780): lane-strided partials, starting
  at positive zero, followed by XOR offsets 16, 8, 4, 2, 1.
- `native_gu_kernel<8>` (lines 1766–1796): eight rows per 256-thread group,
  separate gate/up row ranges, active group intervals and entry token indices.
  Buffer byte offsets replace native weight pointers. Pure Q8 uses final scale 1.
- `launch_gu` (around line 3809): this component implements the full-warp
  branch only. It does not implement split-K, fused/subwarp, or tail formats.

Inspection of a complete CUDA 13.1 sm_86 symbol (Release, no per-file fast-math)
shows separately rounded scale `FMUL`, integer conversion, and `FFMA` into the
lane accumulator. The shader uses explicit `fma`; `precise` protects the scale
multiply and separate reduction additions. The first reduction addition lists
the shuffled operand first, matching that symbol. A final separate multiply
retains the native scale operation. Shared memory implements logical 32-lane
behavior independently of physical subgroup width. This is a source/compiler
mapping, not evidence that a Vulkan driver executed the same instructions.

This consumer requires FP32 RTE and signed-zero/Inf/NaN preservation, with no
FP32 denorm execution mode. For **finite half scales and signed byte codes**,
nonzero scale products and partials are integer multiples of 2^-48; the bounded
lane and reduction sums cannot enter FP32's subnormal range or overflow. That
argument concerns this packed consumer, not the preceding F32 quantizer or
SwiGLU. The finite synthetic weight recipe and retained input preflight remain
mandatory. Nonfinite scales and other native branches are not qualified.
The strict quantizer, its capability gate and original packed-byte failure
remain unchanged. All new reports retain `qualified=false` and
`quantizer_qualified=false`, even when gate/up bits match.

Build and check without GPU execution, using the existing configured opt-in
Vulkan build directory:

```sh
cmake --build BUILD --target native_q8_gu_component native_q8_gu_shader_test native_q8_gu_input_test native_q8_consumer_input_test -j4
ctest --test-dir BUILD -R '^(native_q8_.*|portable_logits_guard_build_test)$' --output-on-failure
BUILD/native_q8_gu_shader_test /tmp/native-gu-effective
for mode in 0 1 2 3; do spirv-val --target-env vulkan1.2 /tmp/native-gu-effective.$mode.spv; done
```

The shader test checks two signed integer dots, one explicit Fma, separate
scale/final multiplies and reduction additions, logical-warp barriers, and the
actual pinned helper's four FP16 metadata combinations. It validates the
module after helper modification and rejects arithmetic changes. CPU metadata
tests cover empty groups, inactive intervals, tokens, offsets and bounds. These
checks and compilation are not GPU proof.

One bounded, independent capture against an **existing** native oracle follows.
`EVIDENCE` identifies the retained consumer result directory; use its verified
receipt hashes and a fresh `OUTPUT`. `DEVICE` and `DEVICE_NAME` must already be
verified on the intended same device. This runs no CUDA oracle or model:

```sh
python3 tools/native_q8_gu.py \
  --component BUILD/native_q8_gu_component \
  --device "$DEVICE" --expect-device "$DEVICE_NAME" --output "$OUTPUT" \
  --fixture "$EVIDENCE/fixture.bin" --fixture-sha256 "$FIXTURE_SHA256" \
  --native "$EVIDENCE/native.bin" --native-sha256 "$NATIVE_SHA256" \
  --observation "$EVIDENCE/observation.bin" --observation-sha256 "$OBSERVATION_SHA256" \
  --oracle "$EVIDENCE/native-consumer.raw" --oracle-sha256 "$ORACLE_SHA256" \
  --weights "$EVIDENCE/native-consumer.raw.weights.bin" --weights-sha256 "$WEIGHTS_SHA256" \
  --native-receipt "$EVIDENCE/receipt.json" --native-receipt-sha256 "$RECEIPT_SHA256"
```

The wrapper binds the retained receipt to all original artifacts, preserves the
full original 832-element comparison failure, verifies the unchanged finite
weight recipe and only-block-5 input difference, and launches the component once
with a 30-second timeout. It retains actual exit/PID, raw gate/up bits, buffer
views including sentinels, final effective SPIR-V, hashes and a comparison
against the retained native gate/up oracle, including zero signs. Exit 0 means
this bounded gate/up comparison completed exactly; exit 1 means a mismatch,
sentinel failure, rejected input, or execution failure. Neither exit qualifies
the quantizer or authorizes product integration. No down-stage dispatch occurs.

An archive-built native consumer can have an explicitly null `source_head`
because its verified source archive contains no `.git` metadata. Keep that
original receipt unchanged. After independently verifying its source commit,
archive hash and binary identity from retained build evidence, append:

```sh
  --native-source-head "$VERIFIED_NATIVE_COMMIT" \
  --native-source-archive-sha256 "$VERIFIED_NATIVE_ARCHIVE_SHA256" \
  --native-consumer-sha256 "$VERIFIED_NATIVE_BINARY_SHA256"
```

All three are required together, only for an explicitly null source field.
The binary hash must equal the original receipt's consumer hash. The launcher
binds these explicit values to the already hash-checked original receipt and
records them as `native_archive_identity`, labelled as independently verified
later inspection rather than the original producer receipt. It also retains
the unchanged receipt snapshot. This records independently verified evidence; it does not infer a
commit from an archive hash or verify an archive build itself. Missing or
malformed values, receipt/binary conflicts, and overrides of ordinary Git
receipts are rejected. Existing Git-backed receipts require no additional flags.

## Capture the actual native down input

The paired native consumer additionally saves `native-consumer.raw.hq.bin`:
exactly 72 raw packed bytes, two Q8_1 blocks, copied after its existing native
synchronization. `native_expert_grouped` places `hq` after three aligned F32
planes; this fixture uses 256-byte planes, hence scratch offset 768. The copy
is bounded against `native_expert_scratch_bytes`. No arithmetic or kernel is
added, and the existing gate/up/output raw format is unchanged.

To obtain the missing intermediate with one bounded repetition of the same
synthetic native consumer, build `native_q8_consumer` in the existing CUDA build
and use the existing original fixture/captures plus retained oracle and weights:

```sh
cmake --build CUDA_BUILD --target native_q8_consumer -j4
python3 tools/native_q8_consumer.py \
  --consumer CUDA_BUILD/native_q8_consumer \
  --device "$DEVICE" --expect-device "$DEVICE_NAME" --output "$FRESH_OUTPUT" \
  --fixture "$EVIDENCE/fixture.bin" --fixture-sha256 "$FIXTURE_SHA256" \
  --native "$EVIDENCE/native.bin" --native-sha256 "$NATIVE_SHA256" \
  --observation "$EVIDENCE/observation.bin" --observation-sha256 "$OBSERVATION_SHA256" \
  --retained-oracle "$EVIDENCE/native-consumer.raw" --retained-oracle-sha256 "$ORACLE_SHA256" \
  --retained-weights "$EVIDENCE/native-consumer.raw.weights.bin" --retained-weights-sha256 "$WEIGHTS_SHA256"
```

The new arguments are all-or-none. Their supplied hashes are checked before
launch and fresh snapshots are saved. After the single existing native call,
the launcher requires the new gate/up/output raw bytes and weight bytes to equal
the retained artifacts exactly, and the sidecar to contain exactly 72 bytes.
Failure returns exit 1, keeps actual child exit and artifacts, and leaves
`hq_reuse_verified=false`. Only a completed guarded capture sets that flag true;
without retained artifacts the sidecar is still saved but is not verified for
reuse. The standalone down-only runner below requires that verification and its
hashes before consuming the sidecar. No hidden values are reconstructed or quantized
by Vulkan. The original packed-byte failure and unqualified status persist.

## Standalone Q8_0 down consumer

`native_q8_down_component` ports `native_down_kernel<8>` at
`src/kernels/cuda/iq_kernels.cu:1933–1955`. It consumes the verified **actual
native packed hidden sidecar**, not the earlier input Q8_1 blocks or a
reconstruction from gate/up/output. The complete finite synthetic expert blob
is reused; its down matrix starts at byte 17408, with 256 rows of 34 bytes.

The native source uses `hq + e*(n_ff/32)` for the hidden input and directly
stores to `out[ent_dst[e]*n_embd + row]`. Group starts, group-strided traversal,
entry order and this destination scatter are preserved. There is no atomic
accumulation, sorting or combine stage. Destinations shared across parallel
groups have the native store race; this component introduces no resolution
policy and the bounded fixture uses distinct destinations. Empty/inactive
intervals do not write output.

Gate/up and down include the same `native_q8_dot.glsl` machinery: two signed
packed integer dots, separately rounded scale product, explicit Fma into a
positive-zero lane accumulator, then separate additions at XOR offsets
16/8/4/2/1. Inspection of the complete CUDA 13.1 sm_86 down symbol confirms
that arithmetic and the final scale multiply/direct store. The branch selected
through `launch_down<8>` (around line 3880) is source-derived, not dynamically
traced. Q8_0 does not satisfy the alternative `expdec_down_ok` format condition.
`STRATA_KQ_SPLIT` must retain the per-entry full-warp branch recorded by the
native receipt. Split-K, subwarp, tails and other formats remain outside this
component. The existing finite-scale argument and FP32 controls apply; neither
this arithmetic mapping nor compilation establishes Vulkan GPU parity.

The standalone owner reuses the gate/up component's pipeline setup and the
pinned final-module validation helper. Product paths and the strict quantizer
remain unchanged. The original full 832-element failure remains in the report.
All down results retain `qualified=false` and `quantizer_qualified=false`.

Build and host checks:

```sh
cmake --build BUILD --target native_q8_down_component native_q8_down_shader_test native_q8_down_input_test native_q8_consumer_input_test native_q8_gu_component native_q8_gu_shader_test -j4
ctest --test-dir BUILD -R '^(native_q8_.*|portable_logits_guard_build_test)$' --output-on-failure
BUILD/native_q8_down_shader_test /tmp/native-down-effective
for mode in 0 1 2 3; do spirv-val --target-env vulkan1.2 /tmp/native-down-effective.$mode.spv; done
```

The down shader checks the shared arithmetic, logical warps, absence of atomics,
required controls and the actual pinned helper's four FP16 transformations.
CPU tests cover metadata bounds, entry/destination distinction, empty/inactive
intervals, sidecar size/finite scales, verified receipt bindings, signed-zero
mismatches and untouched destination sentinels. These are not GPU execution.

One bounded Vulkan-only capture against the retained native hq capture follows.
`EVIDENCE` identifies that new native capture directory; verify all hashes and
use a fresh `OUTPUT`. The existing receipt must bind `hq`, original fixture and
captures, raw oracle and weights, with `hq_reuse_verified=true`. Its retained
oracle/weight identities must also agree. Archive identity flags use the same
explicit later-inspection support as gate/up; omit those three flags for an
ordinary Git-backed producer receipt.

```sh
python3 tools/native_q8_down.py \
  --component BUILD/native_q8_down_component \
  --device "$DEVICE" --expect-device "$DEVICE_NAME" --output "$OUTPUT" \
  --fixture "$EVIDENCE/fixture.bin" --fixture-sha256 "$FIXTURE_SHA256" \
  --native "$EVIDENCE/native.bin" --native-sha256 "$NATIVE_SHA256" \
  --observation "$EVIDENCE/observation.bin" --observation-sha256 "$OBSERVATION_SHA256" \
  --oracle "$EVIDENCE/native-consumer.raw" --oracle-sha256 "$ORACLE_SHA256" \
  --weights "$EVIDENCE/native-consumer.raw.weights.bin" --weights-sha256 "$WEIGHTS_SHA256" \
  --hq "$EVIDENCE/native-consumer.raw.hq.bin" --hq-sha256 "$HQ_SHA256" \
  --native-receipt "$EVIDENCE/receipt.json" --native-receipt-sha256 "$RECEIPT_SHA256" \
  --native-source-head "$VERIFIED_NATIVE_COMMIT" \
  --native-source-archive-sha256 "$VERIFIED_NATIVE_ARCHIVE_SHA256" \
  --native-consumer-sha256 "$VERIFIED_NATIVE_BINARY_SHA256"
```

The wrapper launches one process with a 30-second deadline and three fixed
component dispatches: both entries to destinations 0/1, only entry 1 to
destination 2, then no groups. Four destination slots preserve unused rows.
The first case compares all 512 native output words; the latter cases compare
entry 1's retained output and untouched sentinels. All comparisons use raw bits,
including zero signs. New binary marker `0x3144384e` contains sentinel status
and three count-prefixed 1024-word arrays; `.views.bin` retains complete input
and output base buffers for each case, and `.effective.spv` records the final
module. Source/binary/input/output identities, actual PID/exit, all differences
and original failure are retained. Exit 0 means this bounded comparison matched;
exit 1 means invalid input, execution failure, sentinel failure or mismatch.
It does not qualify quantization or integrate the pipeline into the product.
No new native execution or model work occurs in this command.

### Composed packed-boundary observation (H=256, F=32 only)

The isolated stage results do not establish that the activation/quantization
stage matches on actual gate/up outputs. `native_q8_composed_component` connects
the same two-entry fixture through the existing runners: packed input to
`native_gu_kernel<8>` port, fused observation SwiGLU/Q8_1, then
`native_down_kernel<8>` port. Gate/up and the 72 packed hidden bytes stay in their
original Vulkan tensor views between calls. The existing runners synchronize
each stage. There is no host reconstruction or intermediate upload.

Both the wrapper and component require `--observe-unspecified-fp32-denorm`.
The strict shader, strict creator and unsupported DenormPreserve32 refusal are
unchanged. The component explicitly creates the existing observation module;
it does not fall back from a strict failure. Any remaining required-property
refusal stops before dispatch. Product execution is unchanged.

Build and run the focused host/SPIR-V checks with the retained Vulkan settings:

```sh
cmake --build BUILD --target native_q8_composed_component \
  native_q8_gu_shader_test native_q8_shader_test native_q8_down_shader_test \
  native_q8_spec_test native_q8_gu_input_test native_q8_down_input_test \
  native_q8_consumer_input_test -j4
ctest --test-dir BUILD --output-on-failure -R '^native_q8_(gu_shader|shader|down_shader|spec|gu_input|down_input|consumer_input|tools)_test$'
```

One bounded process follows. Reuse the same verified native hq receipt, original
captures, output oracle and weights as the standalone down check above. No
new CUDA capture or fixture generation is needed. `EVIDENCE` is the retained hq
result directory. All hash variables must come from the retained receipts and
be checked against the actual files. Use the independently verified archive
identity for the historical null Git field; do not edit that receipt. Omit the
three archive identity flags only for an ordinary Git producer receipt.

```sh
python3 tools/native_q8_composed.py --observe-unspecified-fp32-denorm \
  --component BUILD/native_q8_composed_component \
  --device "$DEVICE" --expect-device "$DEVICE_NAME" --output "$OUTPUT" \
  --fixture "$EVIDENCE/fixture.bin" --fixture-sha256 "$FIXTURE_SHA256" \
  --native "$EVIDENCE/native.bin" --native-sha256 "$NATIVE_SHA256" \
  --observation "$EVIDENCE/observation.bin" --observation-sha256 "$OBSERVATION_SHA256" \
  --oracle "$EVIDENCE/native-consumer.raw" --oracle-sha256 "$ORACLE_SHA256" \
  --weights "$EVIDENCE/native-consumer.raw.weights.bin" --weights-sha256 "$WEIGHTS_SHA256" \
  --hq "$EVIDENCE/native-consumer.raw.hq.bin" --hq-sha256 "$HQ_SHA256" \
  --native-receipt "$EVIDENCE/receipt.json" --native-receipt-sha256 "$RECEIPT_SHA256" \
  --native-source-head "$VERIFIED_NATIVE_COMMIT" \
  --native-source-archive-sha256 "$VERIFIED_NATIVE_ARCHIVE_SHA256" \
  --native-consumer-sha256 "$VERIFIED_NATIVE_BINARY_SHA256"
```

The reused wrapper validates all input hashes, native hq reuse and original
packed-input mismatch, then launches one process with a 30-second deadline.
It saves actual child PID/exit and the assembled command. The component makes
three fixed stage dispatches; it has no sparse/empty repetitions. Both actual
packed-input representations are consumed, as in the retained native oracle.
The native hq file is an offline reference only, never a component input.

`vulkan-composed.raw` has magic `0x3150384e`, a sentinel status and count-prefixed
sections of 64 gate words, 64 up words, 18 packed hidden words (72 bytes), and
512 final output words. The comparator checks every float bit, including zero
signs, and every hidden byte, reporting metadata versus code differences and
nonfinite outputs. A hidden mismatch fails even if the final output matches.
All differences are retained; there is no tolerance. Final down consumes the
actual Vulkan-produced hidden bytes even when those differ from the oracle.
Readback occurs after all three dispatches.

`.views.bin` (magic `0x3156434e`) saves the seven complete backing views in order:
weights, packed input, gate, up, unused hidden scratch, packed hidden and final
output. Prefix/suffix canaries, unchanged weights/input and untouched fused
hidden scratch are checked. The three final modules are retained as
`.gu.effective.spv`, `.activation.effective.spv`, `.down.effective.spv`; run
`spirv-val --target-env vulkan1.2` on each and retain their hashes with the
wrapper receipt. The existing shader tests check arithmetic controls and the
pinned helper transformations; no new shader is introduced.

Exit 0 means only that this bounded composed observation matched. Exit 1 means
an input, execution, sentinel or exact-comparison failure. All receipts and
comparison reports retain `qualified=false`, `quantizer_qualified=false`,
`observation_only=true`, `fp32_denorm_mode=unspecified` and the original packed
comparator failure. This does not qualify strict quantization, CUDA/Vulkan
exponential behavior for other inputs, production shapes or full inference.

### Native model geometry control (Q8_0 H=2560, F=640)

`--native --spec2` names model/speculation modes, not a guarantee that an
optional expert fusion ran. In the retained source, `kSplit<8>` exists but
`split_off<8>()` is true unless `STRATA_KQ_SPLIT` starts with `1`. The unset
control therefore takes `launch_gu<8>`'s per-entry `native_gu_kernel<8>` and
`launch_down<8>`'s `native_down_kernel<8>`. Geometry does not enable Split by
itself. `gu_split(8)` is false, so gate/up uses eight full-warp rows per block.
Q8_0 down also uses eight full-warp rows. The existing Vulkan shaders implement
these branches, including their longer accumulation loops, without a new port.

Do not force old kernels or unset a retained opt-in to obtain this branch.
The new native capture reads the actual production selectors and refuses if
Split is selected, grouped-v1 is selected, old kernels are forced, or the
execution phase differs. It records `gu_swiglu_on()` rather than assuming its
value: even when requested, `launch_gu_swiglu<8>` declines `split_off<8>()`.
The V2/V2K, EXPDEC and experimental fused layouts do not cover a Q8_0 gate/up
pair. With grouped-v1 false and no Hadamard, activation is the existing
`swiglu_q8_1_entries_kernel`. The exact control environment must be inherited
from the retained receipt; `STRATA_KQ_SPLIT` remains unset when the receipt
leaves it unset. No environment override is applied by the wrapper.

The standalone CUDA target includes the **unchanged** production
`src/kernels/cuda/iq_kernels.cu`. It inherits `strata_kernels` target options,
definitions, include paths and the configured CUDA architecture/global flags.
There are no source-specific fast-math flags on that production file. It links
the existing library for external dependencies; its included production
symbols already satisfy references to that translation unit. It invokes the
unchanged `quantize_q8_1_rows` and `native_expert_grouped` APIs. A runtime
selector receipt records the source-selected kernel names and operands. This
is runtime selector evidence in the actual production TU, **not** a profiler
launch trace. Keep the build command/compile_commands.json and log with it;
compilation alone cannot establish numerical parity.

The fixture uses one finite synthetic expert (5,222,400 bytes), three distinct
F32 input token rows, three entries, three groups with starts `[0,2,2,3]`, and
a two-group grid. This covers an empty group and group striding. Token IDs
`[2,0,1]` and destination IDs `[1,2,0]` distinguish entry, input and output
indices. Each active group references the same expert; no duplicate expert
allocation or new scheduling policy is needed. The up/down offsets are
1,740,800/3,481,600 bytes. Old H256/F32 recipes and their weight bytes remain
unchanged. Only the weight-file hash check receives the exact new 5,222,400-byte
bound; other inputs retain their original 1 MiB limit.

Native and Vulkan captures each retain 8,640 input-Q8 bytes, 1,920 gate words,
1,920 up words, 2,160 hidden-Q8 bytes and 7,680 destination-ordered output words.
Fused activation does not materialize a float hidden plane. For diagnosis,
after final output exists, each component also runs its **existing separate**
SwiGLU stage into a separate plane and captures 1,920 float words. This is
labelled `hidden_float_diagnostic`; it never supplies packed hidden or down.
No CPU intermediate reconstruction occurs. Vulkan input quantization is also
explicit observation; all actual intermediate tensor connections stay on GPU.

Build `native_q8_geometry_consumer` once with the retained successful CUDA
13.1/host compiler/sm_86 Release configuration, same GGML pin and unchanged
compiler/kernel flags. Build `native_q8_geometry_component` with the retained
Vulkan configuration. Add only `CMAKE_EXPORT_COMPILE_COMMANDS=ON` for compiler
receipts. No dependencies, clocks, native flags or model configuration change.
On the Vulkan build, focused checks are:

```sh
cmake --build VK_BUILD --target native_q8_geometry_component \
  native_q8_geometry_input_test native_q8_composed_component \
  native_q8_gu_shader_test native_q8_shader_test native_q8_down_shader_test \
  native_q8_spec_test native_q8_gu_input_test native_q8_down_input_test \
  native_q8_consumer_input_test -j4
ctest --test-dir VK_BUILD --output-on-failure -R '^native_q8_(geometry_input|gu_shader|shader|down_shader|spec|gu_input|down_input|consumer_input|tools)_test$'
cmake --build CUDA_BUILD --target native_q8_geometry_consumer -j4
```

Run only the following one native capture, then one observation on the same
verified RTX 3060. Use fresh result directories and retained resource guards.
Stop on any build/check/capture failure; do not change selectors or retry.
The fixture-generation command returns before backend initialization. Its
SHA-256 is `26c5ab4747b9957fd2fe9971d6389365ffca53b2f3b069c4c9d72754d35f39c8`.
`CUDA_DEVICE` and `VK_DEVICE` are independently verified API indices for that
physical device. Each wrapper invocation has one 30-second child deadline.

```sh
VK_BUILD/native_q8_geometry_component --make-fixture "$FIXTURE"
sha256sum "$FIXTURE"
python3 tools/native_q8_geometry.py --capture-native \
  --component CUDA_BUILD/native_q8_geometry_consumer \
  --device "$CUDA_DEVICE" --expect-device "$DEVICE_NAME" \
  --fixture "$FIXTURE" \
  --fixture-sha256 26c5ab4747b9957fd2fe9971d6389365ffca53b2f3b069c4c9d72754d35f39c8 \
  --output "$NATIVE_OUTPUT"
# Verify the new native receipt, dispatch selectors and actual artifact hashes.
# Populate these three hash variables from that retained, successful capture.
python3 tools/native_q8_geometry.py --observe-unspecified-fp32-denorm \
  --component VK_BUILD/native_q8_geometry_component \
  --device "$VK_DEVICE" --expect-device "$DEVICE_NAME" \
  --fixture "$FIXTURE" \
  --fixture-sha256 26c5ab4747b9957fd2fe9971d6389365ffca53b2f3b069c4c9d72754d35f39c8 \
  --native-raw "$NATIVE_OUTPUT/capture.raw" --native-raw-sha256 "$RAW_SHA256" \
  --native-receipt "$NATIVE_OUTPUT/receipt.json" --native-receipt-sha256 "$RECEIPT_SHA256" \
  --weights "$NATIVE_OUTPUT/capture.raw.weights.bin" --weights-sha256 "$WEIGHTS_SHA256" \
  --output "$VULKAN_OUTPUT"
```

The native receipt binds original source/fixture-helper hashes, binary, runtime
selectors, environment, input fixture, raw output and captured weights. A null
archive `source_head` is preserved, with source identity established by exact
file hashes; it is not a Git-only prerequisite. Record the staged revision and
archive hash separately. The observation checks those bindings and the same
native source mapping before launch. Both raw roles have distinct magic,
status and six count-prefixed sections in the order above, with the float
hidden diagnostic before hidden-Q8. Every float bit, zero sign, packed byte,
nonfinite metadata/output and canary failure is checked without tolerance.
Hidden-Q8 mismatch fails even when final output matches.

Preserve native `.dispatch.json` and `.weights.bin`; Vulkan `.views.bin` saves
the eight complete bases (weights, packed input, gate, up, hidden float,
hidden-Q8, output, F32 input). Its three effective modules use the same names
as the smaller composition. Run `spirv-val --target-env vulkan1.2` on each;
retain actual exits, binary/module/output hashes and cleanup. Exit 0 is a
bounded fixture match only. Both modes retain `qualified=false`,
`quantizer_qualified=false` and the original input-Q8 comparator failure.
Strict DenormPreserve32 behavior, all shaders and product execution stay
unchanged. This does not establish model inference, other weight formats,
optional Split/fused kernels or performance.

### Functional integer dot on an unaccelerated Vulkan implementation

Native HIP's signed Q8 dot is a four-signed-byte integer sum, including on
architectures where its compatibility layer lowers that sum to individual
integer operations. A Vulkan driver that supports `shaderIntegerDotProduct`
provides the same integer operation even if
`integerDotProduct4x8BitPackedSignedAccelerated` is false. The acceleration
property selects GGML performance paths; it is not an arithmetic requirement
of this component.

Only when `STRATA_VULKAN_NATIVE_STAGES=ON`, the exact pin-checked build-tree
backend copy now has five small device-initialization changes. It retains the
extension/compiler/explicit-disable admission **before** GGML applies its
acceleration filter, appends that functional feature to the normal query and
creation chain, and records enablement only after successful device creation.
The record is a derived host-side device object owned by the same shared
pointer; the base type ABI, GPU resource ownership and existing queue remain
unchanged. The component requires that creation receipt. It does not infer
enablement from a physical feature query alone.

GGML's original `device->integer_dot_product` assignment and every performance
use of it remain unchanged. `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` is still
honored, as is absence of the extension/compiler support or queried shader
feature. There is no replacement shader, arithmetic fallback, new device,
subgroup-width requirement or quantization policy. The strict module and
unsupported DenormPreserve32 refusal remain unchanged. With the native-stage
option off, generated backend source remains the exact original plus the
existing logits-guard include.

Focused host checks include `native_q8_integer_dot_test`, executing the actual
patched feature-chain block across support/acceleration/disable/creation
combinations, and `portable_logits_guard_build_test`, checking the source pin,
exact five patch hunks, unique anchors, unchanged helper and default generation.
Run these along with the prior nine native Q8 checks:

```sh
cmake --build BUILD --target native_q8_integer_dot_test \
  native_q8_geometry_component native_q8_composed_component \
  native_q8_geometry_input_test native_q8_gu_shader_test \
  native_q8_shader_test native_q8_down_shader_test native_q8_spec_test \
  native_q8_gu_input_test native_q8_down_input_test native_q8_consumer_input_test -j4
ctest --test-dir BUILD --output-on-failure -R '^(native_q8_(integer_dot|geometry_input|gu_shader|shader|down_shader|spec|gu_input|down_input|consumer_input|tools)_test|portable_logits_guard_build_test)$'
```

For independent execution, use the retained Release Vulkan build settings and
GGML pin on each host. No CUDA rebuild/reference capture is needed: none of the
native reference source fingerprints changes. First compare once on the
verified RTX3060 on02 against its retained CUDA geometry reference. Only after
that succeeds, compare once on the independently identified V340L die on05
against the same unchanged reference. The four synthetic wrapper inputs are
`fixture.bin`, `capture.raw`, `capture.raw.weights.bin`, and original native
`receipt.json`. Retain the native component log and external identity/hash
manifest as reference identity evidence. Never relabel native02/CUDA/RTX3060
as Vulkan05/V340L; the observation command's device arguments name only its
current Vulkan target.

On each host, with fresh `OUTPUT`, the one bounded command is unchanged:

```sh
python3 tools/native_q8_geometry.py --observe-unspecified-fp32-denorm \
  --component BUILD/native_q8_geometry_component \
  --device "$VERIFIED_CURRENT_VK_INDEX" --expect-device "$VERIFIED_CURRENT_VK_NAME" \
  --fixture "$REFERENCE/fixture.bin" \
  --fixture-sha256 26c5ab4747b9957fd2fe9971d6389365ffca53b2f3b069c4c9d72754d35f39c8 \
  --native-raw "$REFERENCE/capture.raw" \
  --native-raw-sha256 6a0fa01669d9d1670b40c40ed551f6efca56babd1666ce909ba9f14bc8506883 \
  --native-receipt "$REFERENCE/receipt.json" --native-receipt-sha256 "$NATIVE_RECEIPT_SHA256" \
  --weights "$REFERENCE/capture.raw.weights.bin" --weights-sha256 "$NATIVE_WEIGHTS_SHA256" \
  --output "$OUTPUT"
```

Record `native_q8_integer_dot` log tuples for both gate/up and down. On the
reported unaccelerated V340L, admission requires `requested=1`,
`queried_support=1`, `logical_enabled=1`, while `ggml_accelerated` must remain
`0`. On RTX3060 the performance flag should retain its observed value. Do not
change a disable control to force admission. Any refusal, mismatch, nonfinite,
sentinel or deadline failure stops the sequence; no tolerance or automatic
retry. Preserve all intermediate comparisons, three effective modules and
`spirv-val --target-env vulkan1.2` exits, actual child/wrapper exits, source and
binary identities and cleanup. Both observations remain unqualified and retain
the original input-Q8 mismatch.

### Optional quantization-only division trace

This probe is optional and is not a prerequisite for the corrected RN module below.

The retained AMD geometry observation differs first at input quantization. An
offline rounded-reciprocal/multiply model reproduces its full input-Q8 vector,
while a rounded-division model reproduces CUDA's. This is a supported numerical
model, not proof of the driver's ISA. Packed half scales cannot expose the
original FP32 scale bits. Do not change integer rounding, add an epsilon or
attribute downstream differences to separate matrix bugs from this evidence.

The separate `native_q8_quant_trace_component` adds only four stores per
selected element to a generated copy of the unchanged stage shader. Elements
36, 118, 1046 and 4479 cover the four differing operand patterns. Stores
capture existing `amax`, `d`, `ratio` and `rounded` values into the first 16
words of the existing hidden binding. That binding keeps its normal full input
shape; unused words and all view boundaries must remain untouched canaries.
No new arithmetic, qualifiers, folding/reassociation controls, shader compile
flags, descriptors or push constants are added. The original strict and
observation modules remain unchanged. The probe has a distinct module identity
and explicit observation creator, and only admits mode0 on the complete
retained 7,680-element fixture.

One existing quantization dispatch processes all input elements. There are no
gate/up, activation, down or model requests. Its complete 8,640 output bytes
must reproduce the retained AMD capture exactly before any trace is eligible
for interpretation. Even a difference outside the four traced blocks fails
that condition. Raw traces remain saved on failure, but `trace_interpretable`
is false. Nonfinite, unwritten or out-of-bounds trace results also fail. The
cause remains marked `cause_isa_proven=false`; a matching output is not a
qualification or corrective-algorithm test.

Build with the retained Vulkan Release configuration and GGML pin, then run
all previous checks plus the new bounds/instrumentation checks:

```sh
cmake --build BUILD --target native_q8_quant_trace_component native_q8_quant_trace_test \
  native_q8_integer_dot_test native_q8_geometry_component native_q8_composed_component \
  native_q8_geometry_input_test native_q8_gu_shader_test native_q8_shader_test \
  native_q8_down_shader_test native_q8_spec_test native_q8_gu_input_test \
  native_q8_down_input_test native_q8_consumer_input_test -j4
ctest --test-dir BUILD --output-on-failure -R '^(native_q8_(quant_trace|integer_dot|geometry_input|gu_shader|shader|down_shader|spec|gu_input|down_input|consumer_input|tools)_test|portable_logits_guard_build_test)$'
```

The source-generation test checks exact original source plus only the fixed
stores, and rejects a changed base shader. The shader test compares arithmetic
instruction profiles and NoContraction counts against the original, checks all
three divisions retain their decorations, and validates the pinned helper's
four effective-module variants. Host tests check the fixed bounds and that
full-output mismatch, canary, nonfinite, unwritten or truncated captures cannot
be interpreted. These checks do not establish hardware arithmetic.

After review, the proposed single 30-second-bounded observation is on the same
verified AMD V340L die on05 as the retained failure. Transfer only the original
synthetic fixture and AMD geometry capture for this command; no weights,
NVIDIA binary or model files are needed. Preserve their identity manifest and
all historical files. Use a fresh output directory and existing guards with
unchanged clocks and environment. No CUDA or geometry capture is repeated.

```sh
python3 tools/native_q8_quant_trace.py --observe-unspecified-fp32-denorm \
  --component BUILD/native_q8_quant_trace_component \
  --device "$VERIFIED_AMD_VK_INDEX" --expect-device "$VERIFIED_AMD_NAME" \
  --fixture "$REFERENCE/fixture.bin" \
  --fixture-sha256 26c5ab4747b9957fd2fe9971d6389365ffca53b2f3b069c4c9d72754d35f39c8 \
  --retained-amd "$REFERENCE/amd-capture.raw" \
  --retained-amd-sha256 31fbc95071ca4d901d7c811b6f59e825c3977c683ed58f660035127c27e8afe5 \
  --output "$FRESH_TRACE_OUTPUT"
```

`quant-trace.raw` has magic `0x3154384e`, a canary status and word-counted sections
of 2,160 packed input words and 16 raw FP32 trace words. Trace rows follow the
four element indices above; columns are amax/d/ratio/rounded. `.views.bin`
retains input, hidden/trace and packed backing buffers; `.effective.spv` is the
final instrumented module. Retain `spirv-val --target-env vulkan1.2` exit,
source/binary/module/output hashes, assembled command, child/wrapper exits and
cleanup. Wrapper exit0 means only reproducible output plus interpretable trace;
exit1 stops without interpretation or retry. All qualification flags remain
false, and both the original subnormal failure and AMD comparison failure
remain recorded as exit1.

### Division contracts in Strata's backends

The observed NVIDIA Vulkan match and AMD/RADV difference are implementation
results within Vulkan's division accuracy allowance, not evidence that AMD
hardware cannot compute a correctly rounded divide. Vulkan lists `OpFDiv`
with a 2.5-ULP single-precision bound in the relevant normal-divisor range.
`RoundingModeRTE` controls operations specified as correctly rounded;
`precise`/NoContraction does not upgrade that division accuracy requirement.
There is no correctly-rounded-divide switch in this existing Vulkan1.2/GLSL
pipeline. No precision-control change or replacement algorithm is proposed by
this probe. [Vulkan operation precision](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-core).

Native HIP shares `roundf(xi/d)` in
`include/strata/kernels/q8_1_finite.hpp`, and uses `amax/127.0f` in the original
Q8 store. Its compatibility headers do not replace this division with a custom
algorithm. Strata's HIP CMake paths do not explicitly set a divide-accuracy
flag for this source. The inspected ROCm compiler source defaults
`HIPCorrectlyRoundedDivSqrt` to true and exposes the corresponding
`-fhip-fp32-correctly-rounded-divide-sqrt` and negative option. This is a
compiler-source fact, not a measurement or verified fleet compiler setting.
[ROCm option default](https://github.com/ROCm/llvm-project/blob/amd-staging/clang/include/clang/Basic/CodeGenOptions.def),
[Clang HIP option](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fhip-fp32-correctly-rounded-divide-sqrt).

SYCL uses `sycl::round(xi/d)` and `amax/127.0f`. Its `sycl/CMakeLists.txt`
explicitly passes `-cl-fp32-correctly-rounded-divide-sqrt` to both JIT and AOT
backends, along with `-fp-model=precise`; it does not rely on generic precise
alone. Neither native AMD HIP nor Intel SYCL has been run against this retained
fixture in the current comparison. Their actual results must not be inferred
from the AMD Vulkan observation or from compiler flags alone.

### Explicit native nearest-even quantizer division

`--native-rn-quantization` selects a separate corrected diagnostic module. It
replaces only `amax/127` and `xi/d`, in both input and fused hidden quantization,
with binary32 round-to-nearest-even division. Integer rounding stays halfway
away from zero, as native `roundf` specifies. SwiGLU's division/exponential,
maximum/sum reduction, half packing, matrix stages, guards and product paths
are unchanged. The ordinary failed shader/module remains available unchanged;
there is no automatic fallback and no trace-capture prerequisite.

The established method is SoftFloat's direct significand quotient/remainder
with guard/sticky rounding, using a 32-bit restoring divide instead of a 64-bit
integer quotient. `native_f32_div_rn.glsl` is shared verbatim with its host test.
After normalizing two nonzero finite 24-bit significands, it generates a 31-bit
quotient and exact remainder. Every remainder is less than 2^25. Packing uses
seven guard/round bits plus the remainder as sticky; underflow extends the
right shift before rounding, including the half-minimum-subnormal boundary.
Overflow produces signed infinity, and tiny rounded results preserve signed
zero. No FP64, integer64, reciprocal approximation or floating division is
needed inside the helper.

Zero, subnormal, infinity and NaN inputs are classified by their original
bits. Subnormal inputs/outputs are preserved by the helper's integer path.
NaNs are quieted while preserving the first NaN operand's payload; invalid
0/0 and infinity/infinity return `0x7fc00000`. This explicit convention does
not claim bitwise NaN-payload agreement across CUDA/HIP drivers, nor expose
floating exception flags. The original native NaN-to-int8 domain remains
unqualified. Full capture comparisons still report all byte differences;
there is no NaN/tolerance waiver. Surrounding reduction/activation arithmetic
and observation-mode denorm limitations are unchanged.

The conceptual prior art and BSD attribution are retained in
`third_party/softfloat/NOTICE`. No library dependency is added. This bounded
correctness implementation costs a 31-step integer quotient loop per divide,
plus normalization for subnormal inputs. It is not a performance claim or a
model-wide software-arithmetic change. Further optimization requires its own
numerical proof and measurements.

Local checks include 250,000 raw-bit random operand pairs and targeted IEEE
edge pairs against host nearest-even division. Host NaN comparisons validate
classification, with the helper's payload convention checked separately; they
do not claim native GPU payload parity. Subnormal ties, signed zero, overflow,
normalization and the four retained boundary patterns are covered. The host
quantizer formula using the same helper also checks the existing 832-element
host fixture. With the actual retained native geometry capture supplied, it
compares all 8,640 input bytes. These are host checks, not GPU execution.

The shader check requires exactly one remaining `OpFDiv` (unchanged SwiGLU),
its existing NoContraction decoration, no 64-bit types, and the integer loops.
The generation test checks exact original source plus the helper and the two
quantizer call replacements. All legacy module/control checks still apply.

```sh
cmake --build BUILD --target native_f32_div_rn_test native_q8_geometry_component \
  native_q8_quant_trace_component native_q8_quant_trace_test native_q8_integer_dot_test \
  native_q8_geometry_input_test native_q8_gu_shader_test native_q8_shader_test \
  native_q8_down_shader_test native_q8_spec_test native_q8_gu_input_test \
  native_q8_down_input_test native_q8_consumer_input_test -j4
ctest --test-dir BUILD --output-on-failure -R '^(native_f32_div_rn_test|native_q8_(quant_trace|integer_dot|geometry_input|gu_shader|shader|down_shader|spec|gu_input|down_input|consumer_input|tools)_test|portable_logits_guard_build_test)$'
# Optional host check against the retained native capture, no GPU:
BUILD/native_f32_div_rn_test --fixture "$REFERENCE/fixture.bin" --native "$REFERENCE/native-capture.raw"
```

After review, one corrected geometry observation can compare directly with the
retained CUDA reference, without another diagnostic campaign or HIP hardware
run as a prerequisite. Reuse the same verified V340L die, settings, synthetic
weights and reference hashes. Do not change or overwrite the original AMD
failure. The existing geometry wrapper retains all six intermediate boundaries
and its 30-second child deadline. An input match alone does not waive any
remaining gate/up, activation, hidden or output difference.

```sh
python3 tools/native_q8_geometry.py --observe-unspecified-fp32-denorm \
  --native-rn-quantization --component BUILD/native_q8_geometry_component \
  --device "$VERIFIED_AMD_VK_INDEX" --expect-device "$VERIFIED_AMD_NAME" \
  --fixture "$REFERENCE/fixture.bin" \
  --fixture-sha256 26c5ab4747b9957fd2fe9971d6389365ffca53b2f3b069c4c9d72754d35f39c8 \
  --native-raw "$REFERENCE/native-capture.raw" \
  --native-raw-sha256 6a0fa01669d9d1670b40c40ed551f6efca56babd1666ce909ba9f14bc8506883 \
  --native-receipt "$REFERENCE/receipt.json" --native-receipt-sha256 "$NATIVE_RECEIPT_SHA256" \
  --weights "$REFERENCE/capture.raw.weights.bin" --weights-sha256 "$NATIVE_WEIGHTS_SHA256" \
  --output "$FRESH_RN_OUTPUT"
```

Verify the actual child command includes `--native-rn-quantization` and its log
reports `native_rn_quantization=true`. Preserve effective modules and validate
them with `spirv-val --target-env vulkan1.2`, all intermediate results, identity
records, source/binary/module hashes, actual exits and cleanup. Native02/CUDA
and Vulkan05/V340L identities remain distinct. Stop at any failure without
retry or tolerance change. `qualified=false` and `quantizer_qualified=false`
remain in force, and the historical failures remain intact even if a new
observation matches.

### Opt-in provider connection and one retained layer

The portable resident expert provider has an explicit
`--expert-native-q8-observation RECEIPT_PREFIX` option. It selects the existing
native Q8 components instead of the provider's host activation boundaries. The
ordinary provider factory and CLI default are unchanged. This remains an
unqualified observation with unspecified FP32 denorm handling, including when
used through the model provider. It requires `STRATA_VULKAN_NATIVE_STAGES=ON`,
explicit resident expert plans, strict accelerator execution, and a writable
shader receipt prefix. There is no automatic fallback.

The source mapping is:

- `src/worker/vulkan_native_q8_layout.hpp` follows `gpu_compute.cu::infer_impl`:
  first-occurrence expert groups, stable original-entry order inside each group,
  token indices from original `T*K` slots, and consecutive packed destinations.
- Input quantization calls native-stage mode 0 with RN quantizer division.
  `strata_vk_q8_gu_run` implements `native_gu_kernel<8>`.
- Mode 2 implements `swiglu_q8_1_entries_kernel`, including the rounded product
  followed by Q8 packing. Its output directly supplies `native_down_kernel<8>`.
  The native default `split_off<8>()` branch is the reference. Optional native
  dispatches, Hadamard layouts, and other formats are outside this connection.
- `NativeQ8Cache::run` leaves gate/up and packed hidden on the same device views.
  Down writes the packed destinations directly. Reply rows are restored to
  original request indices as in the CUDA worker. Routing weights are still
  applied by the existing caller, once, in its existing order.

The frozen packed slot arena supports 1–24 resident experts (5,222,400 bytes
per expert), at most 24 active groups and 80 active entries per call, T1–8 and
K1–10. Larger calls are refused; they are not split or reassigned. The existing
component metadata limits establish these bounds. The provider allocates one
reusable tensor buffer sized for these capacities plus three existing component
metadata buffers. It queries backend padding and block limits before tensor
allocation, reserves 64 KiB for component metadata before creating the owners,
then charges their actual buffer sizes. Counters include weights, scratch,
alignment, and component metadata. They exclude pipeline/driver allocations;
the existing provider workspace-growth and free-reserve checks still apply.
Host metadata uses a 64 KiB tensor context, 16 KiB call-view context, bounded
routing vectors and reply storage. Weight upload staging remains one expert.
No new cache or placement policy is introduced.

For a bounded validation, use `vulkan_native_q8_replay`, which calls this same
executor factory for exactly one existing captured request. It writes only
input Q8, gate, up, consumed hidden Q8, and actual down output, plus request and
group metadata. It never invokes supplementary float mode 1. All files contain
private model evidence and must remain outside public commits. This comparison
has its own explicit consumed-boundary result; it does not modify the original
omnibus comparator or erase its supplementary activation failure.

Build the Vulkan replay and model with the earlier pinned toolchain, GGML and
SPIR-V header settings, adding:

```sh
cmake -S . -B build-stages \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON \
  -DSTRATA_PORTABLE_QSA=ON -DSTRATA_PORTABLE_MODEL=ON \
  -DSTRATA_VULKAN_NATIVE_STAGES=ON
cmake --build build-stages --target vulkan_native_q8_replay portable_model
```

The paired `native_q8_layer_reference` target uses the unchanged included CUDA
production translation unit and its existing target flags, like
`native_q8_geometry_consumer`. It checks the actual native selectors before
launching; it refuses a different branch. Build it in the established CUDA
reference build. Its output records the actual device name and native branch;
the Vulkan replay records its own actual device description and device ID.
Do not relabel either device as the other.

Use the retained capture and staged expert blobs already held by the reviewer.
Select one complete recorded invocation and retain every originally active entry.
A layer-restricted plan must cover every active expert in that invocation;
`strata-gpu-manifest plan LOCAL_MANIFEST ASSIGNMENTS ONE_LAYER_PLAN` can encode
that bounded metadata using the unchanged source manifest. No ownership is
moved between devices. Keep the assignment at or below the stated limits. The
replay refuses a plan that excludes any active entry. It preserves the original
mask, raw activation, IDs, weights and request layer. Each blob is rechecked against its
existing size and digest before upload. Blob filenames use the existing staged
`sgpu1-...blob` convention and can be reused unchanged.

The following is a proposed recipe, not a completed hardware result. Substitute
verified local paths and the selected record number; use new private output
directories. Apply the existing GGML pin and exact source commit to both builds.

```sh
# On the CUDA reference host, with native dispatch environment left unchanged:
timeout 60s build-cuda/native_q8_layer_reference \
  --device 0 --expect-device 'RTX 3060' \
  "$CAPTURE" "$RECORD" "$ONE_LAYER_PLAN" "$BLOB_DIR" "$NATIVE_NEW_OUTPUT"

# Transfer only the retained inputs/reference needed by this bounded comparison.
# On the Vulkan host, explicitly select the verified device:
timeout 60s build-stages/vulkan_native_q8_replay \
  --device 0 --expect-device 'VEGA10' \
  "$CAPTURE" "$RECORD" "$ONE_LAYER_PLAN" "$BLOB_DIR" "$VULKAN_NEW_OUTPUT"

python3 tools/compare_native_q8_layer.py \
  "$NATIVE_NEW_OUTPUT/layer" "$VULKAN_NEW_OUTPUT/layer" "$NEW_REPORT"
```

The comparison refuses mismatched request/plan/group metadata and lengths,
checks finite float/half metadata, and compares all five boundaries exactly.
Its success says only `exact_consumed_boundaries=true` for that one invocation;
`qualified` and `quantizer_qualified` remain false. Capture source heads, build
flags, device identities and file SHA256 values alongside the private result.
The original input-subnormal failure and historical omnibus failure remain.

Local validation for this connection built `portable_model`, the Vulkan replay,
and both existing provider lifecycle tests. Seventeen focused CTests passed,
including stable grouping/restoration, capacity and weight-budget refusals,
exact comparison negative controls, existing default provider byte comparisons,
and the earlier shader/quantizer checks. A separate compile of the executor
without the native provider macro passed. CUDA compilation and the real-layer
GPU comparison were not run here: this environment has no CUDA compiler or
retained real layer weights. Those are the next bounded validation, not claimed
results. Timing is deferred: the provider's existing timer measures fenced host
calls; there is no reusable GPU timestamp utility here. No benchmark framework
or scheduling change is added.


The earlier 14-group provider limit was a diagnostic metadata capacity, not a
native group limit. The activation starts allocation is now 32 I32 words:
starts occupy words 0–24 and count occupies word 31. Its logical allocation
increases from 64 to 128 bytes; backend padding is queried and charged as before.
Gate/up and down keep their existing 128-I32 metadata buffers, repacked as
count[0], weight offsets[1..24], starts[25..49], entry IDs[50..113]. The spare
words remain zero. Tensor capacities, 24 resident slots, workspace admission,
and the 64 KiB component-metadata allowance are unchanged; actual metadata
allocation sizes are still included in counters. The two matrix shader modules
change metadata indices only. The activation/RN shader modules, native arithmetic,
256-thread matrix workgroups, XOR reductions, and group grid-stride loops are
unchanged. Native `native_gu_kernel` and `native_down_kernel` both stride groups
by `gridDim.y`; the provider continues to launch one block row per actual group.

To extract exactly the complete retained request, the existing offline decoder
now has a bounded staging mode. It uses `read_cpu_capture`, the existing local
manifest and `make_gpu_plan`, `GpuAssetSource` for the three native GGUF ranges,
and `GpuCache` for digest-checked atomic publication. It creates a new private
directory, reads one expert at a time and never connects to a GPU or service.
It prints every active entry/token/expert ID; no expert is silently omitted.
Capture/source identity and index must match exactly. Calls beyond 24 groups or
80 entries are refused before staging. It does not replace a resident plan or
change any running ownership.

```sh
cmake --build build-stages --target gpu_capture_replay_check strata-gpu-manifest
timeout 120s build-stages/gpu_capture_replay_check --stage-record \
  "$CAPTURE" 0 "$LOCAL_MANIFEST" "$MODEL_ROOT" "$NEW_PRIVATE_STAGE"
# Outputs: assignments.txt, one-layer.plan, blobs/sgpu1-*.blob.
# The same frozen metadata can also be encoded with the existing manifest CLI:
build-stages/strata-gpu-manifest plan "$LOCAL_MANIFEST" \
  "$NEW_PRIVATE_STAGE/assignments.txt" "$NEW_PRIVATE_STAGE/verified-copy.plan"
cmp "$NEW_PRIVATE_STAGE/one-layer.plan" "$NEW_PRIVATE_STAGE/verified-copy.plan"
```

Use `one-layer.plan` and `blobs` from this output for both paired replay commands
above. For a T2/K10 invocation with 20 distinct active experts, all 20 entries and
all 20 groups must appear in both receipts: 104,448,000 weight bytes, 5,760 input
Q8 bytes, 12,800 gate words, 12,800 up words, 14,400 hidden Q8 bytes and 51,200
output words. This is an expected shape, not a completed GPU comparison.
The selected expert IDs must come from the decoded capture, not an assumed
router list or a previous single-entry receipt.

Local checks for the 24-group extension: 19 focused CTests passed, including the
full-record staging CLI round trip and overwrite refusal. The existing replay
self-check also passed with its localhost socket fixture outside the restricted
socket sandbox. All nine dumped matrix/strict SPIR-V modules validated for
Vulkan 1.2. The strict activation module remains
`67b8b434dc1ee35fab1e12956af0f181a8e5db4f93a6802fe0047260d347cde5`.
No real record was staged or executed here. Rebuild both paired executables:
the reference shares the changed admission header, and the Vulkan host/shaders
share the changed metadata ABI. CUDA compilation of this revision remains a
fleet build check; arithmetic and native source are unchanged.

The original layer replay metadata writer streamed `uint8_t` T and K as
characters. At T2/K10 this emits a control byte for T and a newline for K,
creating nine text lines instead of eight. The writer now casts both values to
unsigned integers. The offline comparator also recognizes the exact old header
encoding, so retained captures can be compared without modifying or regenerating
them. Each metadata input is parsed strictly before comparison. Only the exact
legacy T/K header encoding is converted in memory; the complete resulting
metadata must be byte-identical, including every identity field, digest and
array. No whitespace is trimmed or lines discarded. LF records, digest fields,
array syntax, group/entry/token geometry and boundary lengths are validated.
The report retains both original metadata SHA256 values and encodings plus the
canonical SHA256. Existing `metadata_format`/`metadata_sha256` identify the native
input; `vulkan_metadata_format`/`vulkan_metadata_sha256` identify Vulkan. No tolerance
or numerical success condition changes.

An actual C++ emitter regression generates both the old byte format and the
new numeric format for T2/K10. It checks retained-format parsing, immutable input
bytes, extra-line refusal, header corruption, equivalent legacy/numeric pairing,
and genuine identity/shape/order mismatch refusal.
For existing captures, run only the updated Python comparator:

```sh
python3 tools/compare_native_q8_layer.py \
  "$NATIVE_RETAINED_PREFIX" "$VULKAN_RETAINED_PREFIX" "$NEW_OFFICIAL_REPORT"
```

A numerical mismatch still writes the official report and exits 1. No GPU
rerun is needed to repair this metadata parsing failure.

### Optional bounded provider timing

Append `--timing` to the Vulkan one-layer replay command to prepare one owner
once, run two warmups, then time exactly 16 repeated calls on the same resident
weights and tensor workspace. Without the flag, the existing single-call replay
is unchanged. The CUDA reference refuses this Vulkan-only option.

```sh
cmake --build build-stages --target vulkan_native_q8_replay native_q8_replay_timing_test
# Run under the existing independent device guard, with verified private inputs.
# NEW_TIMING_OUTPUT must not exist; do not overwrite historical reports.
timeout 60s build-stages/vulkan_native_q8_replay \
  --device 0 --expect-device 'VEGA10' \
  "$CAPTURE" 0 "$FULL_RECORD0_PLAN" "$FULL_RECORD0_BLOBS" \
  "$NEW_TIMING_OUTPUT" --timing
```

Keep the external 60-second timeout: it covers process startup as well as
execution. Timed mode also arms a process-local 60-second alarm before reading
the capture or creating the backend, and checks a steady-clock deadline between
calls. Deadline termination is exit 124; it is not a partial successful result.

The first warmup writes the same consumed-boundary files as normal replay.
Capture-once is enabled only for this timing owner, so no boundary reads/file
writes occur in the timed calls. Setup, pipeline creation, weight staging,
warmups, reply checks, allocation snapshots and report I/O are outside the 16
measurements. The measurements include the existing executor's input/metadata
uploads, dispatch fences, final output readback and row restoration. They are
fenced host-call milliseconds, not GPU timestamps or tokens per second.

The report retains all 16 samples plus min/median/max. Every repeated reply must
preserve its expected sequence/index association and match the first reply's
F32 bits exactly. Fresh snapshots query live executor-owned tensor/metadata
buffer sizes and count after each call and must match preparation. These
counters exclude backend/driver-private allocations and do not measure total
VRAM or transient driver allocation events.

Timing does not invoke or alter the exact native comparator. Stable repeated
outputs establish repeatability only. Preserve the existing numerical failures,
original reports, `qualified=false` and `quantizer_qualified=false`. This mode
makes no native parity or throughput claim. The default model/provider path and
all shader/native arithmetic are unchanged.

Local verification built the Vulkan replay, portable model and existing provider
tests. Twenty focused CTests passed, including deterministic clock checks that
exclude warmup/capture and snapshot overhead, the fixed 2+16 calls, exact-output
and sequence failure controls, owned-allocation changes and deadline refusal.
No timing run on a GPU was performed. CUDA compilation of the shared replay
entry-point change is not available locally; its single-call arithmetic remains
unchanged.

When a retained native capture has the old character header and a newer Vulkan
capture has the numeric header, the updated offline comparator accepts them
only after this strict canonical equality check. It then runs the unchanged
exact numerical comparisons. An equivalent metadata encoding cannot convert a
numerical failure into a pass: a regression exercises the real CLI and verifies
exit 1, an official JSON report and unchanged qualification flags for a changed
F32 word. No GPU recapture or original-file rewrite is required.

### Optional strict FP32 FMA diagnostic

`STRATA_VULKAN_NATIVE_Q8_STRICT_FMA=ON` requires the existing native-stage
option and selects strict gate/up and down variants for this diagnostic build.
It is off by default. All backend devices initialized by this build must
advertise `VK_KHR_shader_fma` and report `shaderFmaFloat32`; otherwise device
initialization fails. Use a separate build directory. Do not enable this flag
in an ordinary provider build merely to discover device support.

The pin-checked bridge appends `VkPhysicalDeviceShaderFmaFeaturesKHR` to its
existing feature query and logical-device creation chain, enables the extension,
and requests only FP32. Its creation receipt becomes enabled only after
`vkCreateDevice` succeeds. Strict component creation requires that receipt.
The strict build also rolls back the pinned factory cache on initialization failure. Its destructor skips
Vulkan cleanup when no logical device was created; automatic host members
still unwind. A refused lookup can therefore be retried without returning an
incomplete cached object. There is no fallback to ordinary `fma`, software FMA
or another device.

The strict shader uses `GL_EXT_spirv_intrinsics` to emit `OpFmaKHR`, capability
`FMAKHR` and extension `SPV_KHR_fma`. The separate scale multiply, signed packed
integer dot, conversion, accumulation order, XOR reductions, activation and RN
quantizer are unchanged. The default GLSL `fma` variant remains available with
the option off. Strict FMA does not resolve activation/denormal differences or
establish complete native parity: `qualified=false`, `quantizer_qualified=false`
and all historical exact comparison failures remain observations to preserve.

The [Vulkan proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_fma.html)
explains why GLSL.std.450 Fma does not guarantee fused accuracy. The
[SPIR-V extension](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_fma.html)
defines the strict instruction and required capability. Static inspection of
Mesa 26.2.6's `src/amd/vulkan/radv_physical_device.c` finds the extension and
FP32 feature advertised without a VEGA10 exclusion; its SPIR-V frontend maps
`OpFmaKHR` to `nir_op_ffma`. This is source evidence, not execution evidence.
The retained official source archive matches the SHA256 published in
[Mesa's release notes](https://docs.mesa3d.org/relnotes/26.2.6.html):
`b9436b3e1a8be76fbd4823dc884f90596ab7c6ac5e1867a305e66ffccfb31a92`.
No driver upgrade is part of this option.

Configure the same pinned diagnostic build with the additional flag:

```sh
cmake -S . -B BUILD_STRICT [EXISTING_DIAGNOSTIC_OPTIONS] \
  -DSTRATA_VULKAN_NATIVE_STAGES=ON \
  -DSTRATA_VULKAN_NATIVE_Q8_STRICT_FMA=ON
cmake --build BUILD_STRICT --target vulkan_native_q8_replay \
  native_q8_strict_fma_test strata_q8_gu_shader strata_q8_down_shader
ctest --test-dir BUILD_STRICT --output-on-failure \
  -R '^(native_q8_.*strict_fma.*_test|portable_logits_guard_build_test)$'
```

Configuration checks the Vulkan feature declarations, compiles both actual
strict consumers and validates them for Vulkan 1.2. It refuses unsupported
headers, compiler emission, validator or disassembler support with the failing
tool's diagnostic. Every strict shader rebuild also runs `spirv-val`. No
unvalidated opcode is inserted by host code. The locally checked toolchain is
Vulkan headers 1.4.363, glslc/shaderc 2026.4, glslang 16.6.0 and SPIRV-Tools
2026.4; these are observed working versions, not inferred minimum versions.

Host tests exercise the inserted feature-chain blocks across extension,
support and creation outcomes. A separate cache-lifecycle regression executes
the actual patched factory allocation/cache/rollback blocks and pinned
destructor with mocked Vulkan boundaries. It checks repeated refusal, failed
query/creation, failure after creation, successful retry/reuse and cleanup.
Shader tests validate strict and legacy modules,
check strict FMA and the separately rounded scale multiply, compare arithmetic,
barrier/reduction order and contraction controls, and forbid fast-math modes.
The unchanged effective-module validator still preserves every shader input
instruction through the pinned helper's FP16 metadata insertions. Host checks
and compilation do not verify hardware FMA execution, native numerical parity
or performance. No GPU run is authorized by this build recipe.

The opt-in [native K10 weighted combine](VULKAN-NATIVE-MOE-COMBINE.md)
adds the native arithmetic boundary with a paired synthetic fixture. The
default portable host combine and existing expert provider remain unchanged.

The explicit `--native-moe-combine` option now connects the validated strict K10
component to PortableFfn after complete original-slot expert rows return. See
[VULKAN-NATIVE-MOE-COMBINE.md](VULKAN-NATIVE-MOE-COMBINE.md#opt-in-ffn--model-connection)
for gating, budget, rollback, focused FFN and T1 full-forward recipes. Both
observation flags remain off by default; no full-forward qualification is claimed.


### 80-entry active capacity

The opt-in grouped provider now accepts all 80 active entries in a T8/K10
request when its frozen plan covers them. Gate, up, hidden, hidden Q8 and down
scratch tensors hold 80 entries; input FP32/Q8 still holds eight token rows.
GU/down metadata now occupies 130 I32 words, keeping count at 0, weights at
1–24, starts at 25–49 and entry IDs at 50–129. The same size controls host
arrays, device allocations and uploads. Other users of the shared component
owner keep their 128-word buffers. The aligned tensor and component sizes
remain charged through the existing budget and allocation snapshots.

The 24 resident slots/groups, packed weight ranges, 32-bit weight addressing,
placement, stable group/entry ordering and native arithmetic stay unchanged.
An 81-entry low-level layout is refused; the worker's T8/K10 limit already
prevents such a request. No splitting or accumulation policy is added.

`native_q8_capacity_component` and `native_q8_capacity_consumer` reuse the
composed H2560/F640 fixture for 80 entries, eight input tokens, 24 groups (one
empty), a seven-row group grid, repeated weights and a destination permutation.
The `capacity_padded` variants leave the first and last entries/destinations
inactive and check their canaries. Both capture input Q8, gate/up, separate
hidden diagnostic, consumed hidden Q8 and down. These are standalone test
fixtures, with no change to native dispatch selection or product defaults.
Host call-recorder tests cover capacity, order restoration and allocations;
they do not establish physical capacity or numerical qualification. Physical
CUDA/Vulkan comparison remains a separate paired check, retaining the existing
approximate-intrinsic qualifications.
