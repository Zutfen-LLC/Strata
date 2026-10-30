# One-row native/Vulkan router diagnostic

`STRATA_ROUTER_DIAGNOSTIC=ON` builds separate diagnostic executables. It is OFF
by default and requires tests plus ordinary CUDA or Vulkan native stages. It
adds no model option and changes neither baseline router source nor comparator.
Only the original eight-row synthetic fixture is accepted, bit-for-bit:
SHA256 `385f43385f4be5f7e2782cbb890d6315cdbca414a2f3cf1322607070ab196101`.

The diagnostic retains the full original launch: native one block of 32x8,
Vulkan eight workgroups of 32. Only row 2 writes intermediate data. All eight
rows' final IDs/weights and padded input/output views remain in the normal
receipt. Instrumentation may change compiler decisions; unchanged final output
against the retained baseline is mandatory before any attribution.

## Exact source correspondence

`tools/native_router_trace_source.py` generates separate build-tree variants
from the hash-pinned `fbdec8` baseline sources:

- CUDA `native_router.cu`: SHA256
  `89131c17bb9809e7ed02d08d2e3b03d2a2b0e077885d2f82df9471a7b8bf2256`.
- Vulkan `native_router.comp`: SHA256
  `373c88f4ce7c530ea84f42607c18a620ae965142a52c1c7482dd63103ef2b08d`.

CUDA copies only the existing warp helpers and `route_multi` body into a
separate diagnostic namespace/kernel. The native source and existing kernel
library are untouched. The generated kernel retains `--use_fast_math`, Release
optimization, C++20 and the selected native architecture. Use the same verified
CUDA 13.1.115/sm86 build settings as the retained reference.

Both variants add stores of existing locals. The only expression split names
the existing FP32 full XOR sum before reciprocal division; no operation, type,
rounding, order, normalization or mathematical formula is replaced. Removing
marked stores and reversing this naming/signature change must restore the
baseline kernel body byte-for-byte. A source hash/anchor change refuses
variant generation. The generated CUDA helper retains its original MIT notice.

The Vulkan diagnostic uses the same pipeline owner, feature admission, float
controls, final effective-module receipt and original run/destroy functions.
The existing 128-word status allocation supplies 26 additional float slots after
the first status word; the remaining words are checked as guards. Native trace
storage has four guard words on either side. Baseline shaders and sources stay
unchanged when the diagnostic is selected, and the default build does not
compile the trace variant.

## The 26 words

`trace.f32` is exactly 104 bytes, little-endian FP32 words. `trace.complete.json`
is written only after trace validation and guard checks. The operator wrapper
also records the actual process exit in `capture.rc`; replay requires zero.
Vulkan additionally requires `module-validation.rc` from validation of the
physical effective SPIR-V module.

| Word | Existing local/result captured on row 2 |
|---:|---|
| 0 | Maximum, exactly 8 |
| 1–10 | Exponentials at experts `7+32*j`, j=0..9; arguments `-j/8` |
| 11 | Representative tail exponential at expert 0; argument -24 |
| 12 | Full XOR sum, before first reciprocal |
| 13 | First reciprocal |
| 14–23 | Selected probabilities in rank order |
| 24 | Selected XOR sum after existing clamp |
| 25 | Final reciprocal |

These are the only eleven distinct exponential arguments in this exact row;
all 502 tails have the same input and scalar intrinsic path. Row 2's maximum
and subtractions are exactly representable. No extra intrinsic calls are added.
The ten selected experts belong to logical lane 7, preserving its accumulation
witness. Other fixtures or shapes are refused by the runners.

## Offline admission and replay

`tools/replay_native_router_trace.py` pins the original retained input-view,
IDs and both native/Vulkan weight hashes. It verifies both diagnostic runs'
complete final views and payloads against the retained `fbdec8` receipts before
reading or interpreting either trace. Valid output guards, completion records,
zero process statuses, exact trace length and finite written fields are required.
A changed final bit, stale input, malformed trace or failed run refuses attribution.
There is no tolerance and the original paired comparator remains unchanged.

The replay uses exact rational arithmetic and explicit binary32 nearest-even
packing for additions and products. It never computes exp or substitutes a
reciprocal oracle. From each backend's captured eleven exponentials it restores
the original 32x16 lanes, reproduces lane-local and XOR sums, uses its captured
first reciprocal to check all ten probabilities, reproduces winning-lane
selected accumulation/clamp, then uses its captured final reciprocal to check
the final ten products. Reciprocal comparisons explicitly distinguish identical
inputs from different inputs caused by prior propagation.

- Exit 0: changed same-input intrinsic outputs with matching per-backend
  arithmetic replay. This supports methodology parity for this case; it is not
  exact numerical parity or general qualification.
- Exit 1: an arithmetic/order replay boundary differs. Investigate that boundary
  or capture correspondence before claiming an intrinsic-only explanation.
- Exit 2: invalid/perturbed evidence; attribution is refused. Preserve it.

Reports always retain `qualified=false` and `numerical_parity=false`.

## Bounded paired capture recipe — not executed here

Use clean sources at the reviewed additive commit and the retained build
settings, devices and process-local runtime. Add only the diagnostic option:

```sh
cmake -S SOURCE -B CUDA_BUILD -DSTRATA_ROUTER_DIAGNOSTIC=ON
cmake --build CUDA_BUILD --target native_router_trace_reference
cmake -S SOURCE -B VULKAN_BUILD -DSTRATA_ROUTER_DIAGNOSTIC=ON
cmake --build VULKAN_BUILD --target native_router_trace_component native_router_trace_fixture_test
ctest --test-dir VULKAN_BUILD --output-on-failure -R '^native_router_.*_test$'
```

The private handoff supplies a source/fixture-pinned wrapper. On each designated
host, run it once with its existing verified device/runtime and a NEW result
directory. Each command contains one component dispatch bounded by 60 seconds;
it writes `capture.rc` even for timeout/failure. Do not run a baseline GPU probe
or alter an earlier result directory. Transfer both small diagnostic directories
without rewriting them, then on one offline host:

```sh
python SOURCE/tools/replay_native_router_trace.py \
  --baseline RETAINED_6FE7FBB_PAIRED_RESULTS \
  --native NEW_NATIVE_DIAGNOSTIC --vulkan NEW_VULKAN_DIAGNOSTIC \
  --report NEW_REPLAY_JSON
```

The baseline directory contains `native-result` and `vulkan-result` from private
agenthandoffs commit `6fe7fbb1533a9f0282512629c0269b3ae6c52ffc`.
Native binary disassembly remains at private commit
`e18eb9c3c1673a8bec413e3487a7dfa9146ea8b9`. Original numerical failures and
all earlier evidence remain intact. Retain generated sources, compiler flags,
executable/fixture/module/trace hashes, raw receipts and exact child statuses.

## Local checks

Vulkan diagnostic and baseline components compile in the existing strict-FMA
native-stage build. Eight focused CTests pass, including exact fixture/trace
I/O, generator correspondence, actual optimized SPIR-V arithmetic/control
comparison, exact RNE edge cases, intrinsic-versus-arithmetic mutation cases,
final-output/guard refusal and completion/process-status gates. Host tests also
pass under Python `-O`. With the option OFF, the baseline component rebuild
and five focused baseline checks also pass. The original baseline router SPIR-V hash remains
`db3a9aebad1817064d6041d6d15ac87fe08c0b738593f1e8ec4915eb77c5b47a`.

CUDA diagnostic compilation is not run locally because this workspace has no
CUDA compiler. No GPU capture, fleet test, model integration or numerical
qualification has been performed by this change.
