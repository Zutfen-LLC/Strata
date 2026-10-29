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

The component is not ready for product integration. Grouped gate/up/down,
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
