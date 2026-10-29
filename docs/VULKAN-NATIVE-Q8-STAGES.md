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

The component is not ready for product integration. Additional grouped gate/up branches, down,
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
