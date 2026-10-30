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
