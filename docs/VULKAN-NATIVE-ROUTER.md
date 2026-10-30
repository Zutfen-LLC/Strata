# Standalone native top-10 Vulkan router probe

This component follows Strata's native finite F32 512-expert/top-10 algorithm.
It is an unqualified numerical port; the retained exact comparison fails.
The standalone observation also has an [explicit FFN/model opt-in](PORTABLE-NATIVE-ROUTER.md). Default
routing still calls `portable_ffn.cpp::route_logits`. No routing policy,
expert placement, cache bound or native CUDA kernel changes.

The build requires `STRATA_VULKAN_NATIVE_STAGES=ON`. It reuses the pin-checked
component owner, same-device views, RTE and signed-zero/Inf/NaN controls,
effective-module validation and mandatory shader receipt. The owner's existing
integer-dot admission and conservative 256-invocation device bound still apply.
Its 128-word status allocation is reused to refuse nonfinite input. If the
existing strict-FMA build is selected, its feature receipt also remains required.
The router contains no FMA and adds none: every FP add/subtract/multiply/divide
is `precise` and must retain SPIR-V `NoContraction`.

## Native source contract

At base `f0230ea4b021f04bf94da3725371750bc79baa16` (source inspection, not hardware measurement):

Exact references: [native reductions and single-token route, lines 34–97](https://github.com/Zutfen-LLC/Strata/blob/f0230ea4b021f04bf94da3725371750bc79baa16/src/kernels/cuda/native_router.cu#L34-L97),
[multi-token route, lines 100–149](https://github.com/Zutfen-LLC/Strata/blob/f0230ea4b021f04bf94da3725371750bc79baa16/src/kernels/cuda/native_router.cu#L100-L149),
[CUDA compiler contract, lines 553–568](https://github.com/Zutfen-LLC/Strata/blob/f0230ea4b021f04bf94da3725371750bc79baa16/CMakeLists.txt#L553-L568),
and [portable route, lines 51–70](https://github.com/Zutfen-LLC/Strata/blob/f0230ea4b021f04bf94da3725371750bc79baa16/src/core/portable_ffn.cpp#L51-L70).

- `src/kernels/cuda/native_router.cu::route` and `route_multi`: lane `l` owns
  experts `l+i*32`, i=0..15. Each lane visits its values in ascending i order.
  XOR masks are 16,8,4,2,1 for both max and sum.
- Maximum subtraction, `expf`, lane-ordered sum, XOR sum, reciprocal then
  probability multiplication precede selection. Selection uses computed
  probabilities. Equal probabilities pick the smallest expert ID, including
  ties introduced by rounding/underflow. It does not sort original logits.
- Each rank scans lane-local candidates in ascending i order with strict `>`;
  XOR comparisons include lower-ID ties. The winning expert is removed with
  negative infinity. Each winning expert lane accumulates its own picks in
  selection order. These 32 partials are XOR-summed. Clamp to 2^-14, compute
  one reciprocal, then multiply each selected probability by it.
- `CMakeLists.txt` gives `native_router.cu` `--use_fast_math` on ordinary CUDA
  builds. NVIDIA documents `expf` → `__expf` and division → fast division,
  with a distinct reciprocal accuracy bound. The native reference
  executable calls the existing `native_router_top10_multi` launcher, so the
  reference retains the actual kernel's compiler contract.
- `src/core/portable_ffn.cpp::route_logits` uses double exponentials/sum,
  rank-ordered selected sum and final division. It is not this component's
  arithmetic oracle and remains unchanged.

The Vulkan shader explicitly emulates the logical 32 lanes through shared
memory, with a publish/read barrier for every XOR round. Physical subgroup
width does not select a different reduction. T1..8 rows use independent
32-invocation workgroups. This differs from CUDA's 32x8 block packing while
preserving each row's lane layout and ordered operations.

## Numerical limit requiring physical proof

The shader currently uses GLSL `Exp` and FP32 reciprocal expressions. CUDA's
fast intrinsics and Vulkan's transcendental/division approximations have no
cross-vendor exact-bit guarantee. This is an unresolved arithmetic boundary,
not an exact native port qualification. No host double, software exp/division,
error tolerance or alternative routing policy hides it. The retained mismatch remains a failed
paired numerical comparison. The opt-in integration follows the native method
without claiming cross-vendor bit identity.
[NVIDIA's fast-math mapping](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
explains why a host `exp`/division oracle would not establish the CUDA result.

Inputs must be contiguous F32 [T,512], IDs contiguous I32 [T,10], weights
contiguous F32 [T,10], T1..8. All buffers belong to the selected Vulkan device,
are four-byte aligned and disjoint. Other shapes refuse, with no fallback.
Nonfinite logits refuse after device status readback; output spans are invalid
if any row refuses. Finite extreme subtraction may overflow to negative
infinity, as in the native expression. FP32 denorm mode remains unspecified by
the reused observation owner; this component does not claim denorm parity.

## Fixture and verification

`native_router_input_test` generates a synthetic eight-row fixture: uniform,
signed-zero, ten repeated winning expert lanes, uneven values, near ties,
underflow, finite extremes and varied expert order. The binary format is
versioned little endian, with exact shape/length and finite-input validation.
The eight-row fixture is 16,404 bytes, SHA256
`385f43385f4be5f7e2782cbb890d6315cdbca414a2f3cf1322607070ab196101`.
The host tests exercise T1/T2/T8 spans, unsupported shapes/alignment, lower-ID
probability ties, repeated winners, clamp and an arithmetic witness that changes
bits under rank-order selected accumulation. They accept already computed
FP32 probabilities; they do not emulate native exp/division.

`native_router_shader_test` compiles and validates the optimized actual shader,
checks 32-lane geometry, all arithmetic contraction controls, two reciprocal
multiply paths, barriers/XOR operations, lower-ID tie comparisons, nonfinite
status, and absence of subgroup-dependent/F64/FMA/fast-math instructions. A
removed arithmetic contraction guard fails the same checks. This is an
instruction check, not a proof of GPU output.

Both device runners read the same fixture, use deliberately offset views,
retain all padded input/output words, validate immutable inputs and guards,
and atomically reserve a new output directory. They retain raw ID/weight
words and an observation-only report. Successful runner exit means valid
receipts, not CUDA/Vulkan equality. `compare_native_router.py` requires exact
input identity and zero differing ID and weight words; one flipped weight bit
fails. Nothing changes `qualified=false`.

## Bounded physical recipe

Use existing verified CUDA and Vulkan builds at the additive commit. Keep
private process-local ICD/SDK settings and exact device identities. Do not
replace a driver or run a full model. Record actual exit status through logging
wrappers. The output directories and fixture path must be new.

```sh
cmake --build VULKAN_BUILD --target native_router_input_test native_router_component
ctest --test-dir VULKAN_BUILD --output-on-failure -R '^native_router_.*_test$'
cmake --build CUDA_BUILD --target native_router_reference
VULKAN_BUILD/native_router_input_test --write-fixture NEW_FIXTURE
sha256sum NEW_FIXTURE
# Transfer exactly this synthetic fixture and verify the same hash on both hosts.
timeout 60s CUDA_BUILD/native_router_reference \
  --device CUDA_INDEX --expect-device CUDA_DESCRIPTION \
  --fixture VERIFIED_FIXTURE --output NEW_NATIVE_DIRECTORY
timeout 60s VULKAN_BUILD/native_router_component \
  --device VULKAN_INDEX --expect-device VULKAN_DESCRIPTION \
  --fixture VERIFIED_FIXTURE --output NEW_VULKAN_DIRECTORY
spirv-val --target-env vulkan1.2 NEW_VULKAN_DIRECTORY/effective.spv
python tools/compare_native_router.py NEW_NATIVE_DIRECTORY NEW_VULKAN_DIRECTORY
```

Retain source/build/compiler flags, kernel PTX/SASS if available, fixture,
executables, effective SPIR-V, reports, complete views, exact hashes and
comparator exit status. Preserve mismatches. Physical comparison is not run
by adding or compiling this component. The existing full-model smoke and
native-Q8/combine results do not qualify router arithmetic.

## Local validation for this slice

Passed with GCC 16.2.1, CMake 3.31.6, Vulkan headers 1.4.363,
glslc/shaderc 2026.4, retained SPIR-V headers and pinned ggml
`3cf03257f219afbe7334045ff7c6a06ac68c627d`:

- Vulkan component and host fixture compilation with native stages ON, both
  strict-FMA OFF and ON.
- Five focused CTests in the ordinary build; seven in the strict build,
  including existing strict feature/lifecycle and GU/down shader checks.
- Comparator mutation checks also pass under Python `-O`; runtime acceptance
  does not rely on removable Python assertions.
- Existing ordinary GU/down SPIR-V hashes match the retained baseline exactly.
- `git diff --check`; native CUDA router and default portable FFN source unchanged.

Corrected local attempts: CMake was outside PATH; the retained CMake executable
was used. The first backend build lacked the retained SPIR-V include directory;
the isolated build flags were corrected. A shader CTest exposed a validator
variable scoped to the strict helper; the router now discovers its own required
validator/disassembler. An initial host witness used non-distinguishing lanes;
the retained lanes 1/3 witness now distinguishes the two accumulation orders.

Not run: CUDA reference compilation (no local CUDA compiler), physical CUDA or
Vulkan dispatch, native PTX/SASS inspection, paired router comparison, model
integration, fleet tests or performance measurement. The draft remains
unqualified; the physical comparison above is pending, not denied.


## Retained row-2 finding and opt-in integration

Private evidence `306b3710058bd20a601b073dd06045f04eeb98c0` preserves both
backends' original final outputs and has zero sum/product replay discrepancies.
Five same-input exp outputs differ by one ULP; both reciprocal inputs differ,
so no independent same-input reciprocal conclusion follows. The original 21/80
weight mismatch remains. This single synthetic row supports the inspected
methodology, not global correctness or complete active-resource qualification.
The [explicit portable FFN/model connection](PORTABLE-NATIVE-ROUTER.md) keeps
default routing unchanged, adds completed participation receipts and provides
a bounded OFF/ON generation comparison recipe.
