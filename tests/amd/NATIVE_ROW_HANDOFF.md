# One real captured expert: native CPU versus Vulkan arithmetic

This is an additive diagnostic on PR #31. Preserve the original immutable
`5139cd08ae1ffa92542562ddadf3d704fa333257` build and outputs. The parent reports
that its real expert302 same-backend row comparison passed, while its Vulkan
model trace failed the unchanged CPU logit gate. The same-build CPU-only control
reportedly matches historical A0 exactly. These are supplied results, not runs
performed by the author of this diagnostic. No more model/trace runs belong here.

Use the existing `strata-vulkan-q8-expert` capture mode with the new
`--capture-native-check` flag. It retains all original capture row checks. It adds
native arithmetic for only the first active entry of the first intersecting
record: same real Q8 blob and original captured F32 token input, production
`native_quant_act`, `native_gu_rows`, host SwiGLU, `native_quant_h`, and
`native_down_rows`. Gate/up diagnostic dots use the same native ggml type trait.
The selected comparison is the already-computed direct hybrid Vulkan graph row;
no substitute weights, new runner or per-record native campaign is added.

The four metric stages are `gate`, `up`, `hidden` (host SwiGLU), and `out` (down
projection). Every stage records bit differences, max absolute error and relative
L2. Packed input and hidden boundaries are compared separately by exact bytes.
Differences are observations, not a tolerance pass. Model/CPU parity stays
unestablished even if this one row is exact.

## CPU reference admission

The reference must be the qualified AVX2/FMA/F16C, non-VNNI, non-AVX512 CPU path.
A tiny C witness is compiled inside the existing ggml-cpu target with exactly its
C flags, including `-march=native`. It reports actual AVX2/FMA/F16C, AVX512,
AVX-VNNI, AVX512-VNNI, AVX-VNNI-INT8 and fast-math macros. This does not infer ISA
from CMake cache settings, nor use the probe translation unit's baseline flags.
`--native-isa-check` is a CPU-only preflight; exit 0 requires flags=7 and
`qualified_avx2_non_vnni=1`. The requested arithmetic diagnostic also enforces
this guard before reading the capture or creating its scratch service.

If it refuses, stop and report the actual flags/compiler. Do not label that
reference CPU-equivalent, change its arithmetic/ISA or rebuild shaders to get
past it. Parent review must decide any separately scoped CPU-only library build.
Also retain the CPU library/compiler/build flags and compare them with the
qualified CPU baseline. This guard establishes an ISA condition, not identity of
all compiler/libm/environment behavior or full-model numerical parity.

## Exact incremental build on the retained offline 05 setup

After parent dispatch, use the existing successful isolated source/build and
installed dependencies, pinned ggml
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Stage the new exact commit in the same
isolated source location, retaining the old commit and executable. Do not change
Vulkan/CPU architecture settings, compiler, ICD or library paths. No downloads,
installs, credentials, listeners on private interfaces or model jobs are needed.

```bash
: "${STRATA_SRC:?retained isolated source staged at the new exact commit}"
: "${AMD_BUILD:?retained successful Unix Makefiles Vulkan build}"
: "${NATIVE_LOGS:?fresh owned log directory}"
mkdir "$NATIVE_LOGS"
cp "$AMD_BUILD/strata-vulkan-q8-expert" "$NATIVE_LOGS/probe.before-native-check"
timeout 120s cmake -S "$STRATA_SRC/tests/amd" -B "$AMD_BUILD" \
  > "$NATIVE_LOGS/configure.log" 2>&1
# Only the new ISA witness and CPU archive; no existing CPU options are changed.
timeout 120s cmake --build "$AMD_BUILD" --target ggml-cpu/fast -- -j2 \
  > "$NATIVE_LOGS/cpu-witness-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata-vulkan-q8-expert/fast -- -j2 \
  > "$NATIVE_LOGS/probe-build.log" 2>&1
timeout 10s "$AMD_BUILD/strata-vulkan-q8-expert" --native-isa-check \
  > "$NATIVE_LOGS/native-isa.log" 2>&1
```

Stop on an error, an unqualified CPU ISA, unexpected dependency compilation or
shader traversal. The only expected compiled objects are the new C witness and
the changed probe, followed by CPU rearchive and probe relink. No executor,
protocol, CUDA coordinator or production worker arithmetic is changed.

## One offline invocation, one physical die

Reuse the exact real capture, trusted manifest, one-expert layer0/expert302 plan,
5,222,400-byte content-addressed blob and physical die0 mapping already qualified
at 5139cd08. Verify the retained full asset hashes before starting. The observed
input record IDs were 0, 768, 816 with T2/K10; native arithmetic runs only for the
first selected entry in record0. Check its emitted original entry/token mapping;
do not guess a rank or modify capture IDs.

```bash
: "${VK_PHYS_DIE:?same confirmed physical die0 Vulkan index from retained inventory}"
: "${CAPTURE:?same sealed real capture}"
: "${LOCAL_MANIFEST:?same trusted source manifest}"
: "${SMALL_PLAN:?same layer0 expert302 plan}"
: "${REAL_BLOBS:?same staged real blob directory}"
: "${NATIVE_OUT:?new output path that must not exist}"
timeout --signal=TERM --kill-after=5s 120s \
  env GGML_VK_VISIBLE_DEVICES="$VK_PHYS_DIE" \
  "$AMD_BUILD/strata-vulkan-q8-expert" --backend vulkan --device 0 \
  --native-q8-boundaries --capture "$CAPTURE" --plan "$SMALL_PLAN" \
  --local-manifest "$LOCAL_MANIFEST" --blob-dir "$REAL_BLOBS" \
  --capture-native-check --out-dir "$NATIVE_OUT" \
  > "$NATIVE_LOGS/native-row.log" 2>&1
```

Require the original `CAPTURE_ROWS_ACCEPTED` receipt and exact saved
`capture.recordN.rows.f32` / `capture.recordN.direct.f32` comparisons. Also require
exactly one `CAPTURE_NATIVE_OBSERVED` line for record0/layer0/expert302, with the
original entry/token, and a qualified ISA receipt. This observation line is not a
numerical acceptance marker. Read `capture.native.metrics.tsv`,
`capture.native.txt`, all `capture.native.*.f32` / `capture.selected.*.f32`, the
original F32 input and packed native/selected input/hidden files. Native output
rows are not weighted or reduced. Input-Q8 equality should hold; hidden-Q8
inequality would show that projection/SwiGLU differences cross a quantizer
boundary. Report either outcome without fitting a tolerance.

Report the earliest differing stage, bit counts, max absolute errors, relative
L2 and both packed equalities. If gate/up already differ with identical packed
inputs, CPU-equivalent projection accumulation is the next compatibility target;
if only down differs with equal hidden Q8, focus on that projection first.
Exact input/hidden quantization does not reproduce CPU integer-dot/FMA lane and
horizontal-reduction order. No existing Vulkan precision switch establishes that
order. Do not implement a compatibility shader before these stage results.

The harness still closes its local client before stopping its service; a single
silent worker failure increment can reflect EOF/stop cancellation during teardown,
as at 5139cd08. It is not itself an arithmetic failure or a reason to change
counters. The same resource-budget limitations apply as in the original handoff.
No full GEN, trace rerun, model-gate relaxation, fleet access or credential work.
