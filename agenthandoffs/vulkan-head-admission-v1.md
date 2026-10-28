# Guarded Vulkan output-head admission correction

This narrow correction starts at PR49
`f8deab2d97baf8534b4737d23b401d9de85c7f4d`. It does not change a model graph,
precision, routing, merge order, expert plans, device caps, free reserve or
runtime defaults. No fleet run or enlarged plan was dispatched. Neither pinned
running test was touched. Independent pinned-source review returned GO, but
unchanged 1152-experts/die plans must not be dispatched: the projected die0
ledger exceeds its existing cap, as recorded below.

## Plan qualification after source review

The parent supplied these projections from the retained actual 480 ledgers;
these are admission arithmetic, not a new runtime result:

| Device | Projected 1152 admission (bytes) | Existing cap (bytes) | Result |
| --- | ---: | ---: | --- |
| Die0 | 8180726368 | 8053063680 | Exceeds cap by 127662688 bytes |
| Die1 | 7973270624 | 8053063680 | Fits arithmetically, 79793056 bytes remain |

Therefore source-review GO does not permit dispatch of the unchanged 1152/die
plans. Keep all caps and reserves. First revalidate runtime head credit using
the old 480 plans; then choose a meaningful count that fits both dies using
exact padded-tensor arithmetic. No code change or cap relaxation is needed.
The parent is waiting for the current PR49 timing run to finish before any
inferswarm05 build or inference. This documentation update starts neither.

## Pinned-source justification

The local `ggml-vulkan.cpp` was compared byte for byte with the upstream file at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Both SHA-256 values were
`cbacf747bee9f72cab18706acd1f96e5f4a87abaa178f297388e04c7217e8343`.

At this pin, [the huge-matrix split precedes the T=1 matvec dispatch](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6946).
The split condition is `ggml_nbytes(src0) > maxStorageBufferRange`. With a
sufficient range, the contiguous Q8_0 output weight and T=1 F32 input reach
`ggml_vk_mul_mat_vec_q_f16`. In [that path](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6384),
`qx_needs_dequant = x_non_contig`; prealloc_x growth is conditional on it.
MMVQ/integer-dot selection can convert Y, not the full contiguous X matrix.
An unavailable matvec pipeline asserts instead of selecting a full-X-dequant
fallback. These are pinned-source conclusions, not new measured GPU results.

The parent's [hardware receipt](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793186078592419)
reports `maxStorageBufferRange=4294967295` bytes for both RADV VEGA10 V340 dies,
PCI `0000:07:00.0` and `0000:0b:00.0`, from existing vulkaninfo output. This
coding task did not query those hosts; admission uses a fresh runtime property,
not those reported values or an AMD/vendor assumption.

## Exact runtime proof and component accounting

`head_storage_limit` queries physical properties using the same linked Vulkan
loader and process ICD environment. It matches `VK_EXT_pci_bus_info` identities
to `ggml_backend_dev_get_props(...).device_id`; raw enumeration positions are
never equated with GGML indices. Exactly one matching discrete/integrated GPU
is required. Missing PCI/extension, a failed/incomplete query, duplicate matches,
software/virtual devices, or a non-Vulkan backend yields no credit. The temporary
instance is destroyed on every return path; no logical device or model buffer
is created by this query. CPU-only builds retain a stub and no Vulkan dependency.
`ggml_backend_buft_get_max_size` is not used: its suballocation block size does
not establish this physical limit.

Credit additionally requires this exact graph contract:

- MUL_MAT with default precision/no hint, output F32 `[248320,1,1,1]`.
- Contiguous, non-view Q8_0 weight `[2560,248320,1,1]`, 675430400 bytes.
- Contiguous F32 RHS `[2560,1,1,1]` (T=1).
- Matching physical `maxStorageBufferRange >= 675430400` bytes.

Other shapes/dtypes/strides/backends retain the old conversion allowance.
When proven, only the full output-matrix conversion component is omitted:

| Head workspace component | Conservative bytes | Guard-proven bytes |
| --- | ---: | ---: |
| Conversion bound: max(output F32 image, HC matrix F32 image) | 2542796800 | 13107200 |
| Other workspace (input/output conversion and split-K allowance) | 16056320 | 16056320 |
| Combined head private/shared workspace before scheduler accounting | 2558853120 | 29163520 |

The HC conversion bound remains. Scheduler tensors/copies are still charged
from their actual reservations; other backend workspace remains additive. The
existing per-backend conversion ledger uses `max(previous, head conversion)`;
this change does not subtract the full table difference blindly from total
admission. QSA token-temporary peaks, expert tensors/workspace/free reserve and
each physical device cap retain their existing terms and checks. No full
workspace-max shortcut or credit from observed peaks is used.

## Fields to retain during parent revalidation

The new stderr record begins `head_admission` and reports:
`pci`, `max_storage_buffer_range`, `weight_type`, `weight_ne0`, `weight_ne1`,
`weight_bytes`, `weight_contiguous`, `rhs_type`, `tokens`, `direct_q8_matvec`,
`conservative_conversion_bytes`, `retained_conversion_bytes`, and
`other_workspace_bytes`.

For the stated full-model head on a successfully matched V340, expect
`weight_type=q8_0 weight_ne0=2560 weight_ne1=248320 weight_bytes=675430400
weight_contiguous=1 rhs_type=f32 tokens=1 direct_q8_matvec=1
conservative_conversion_bytes=2542796800 retained_conversion_bytes=13107200
other_workspace_bytes=16056320`, with the actual queried PCI/range fields.
These are conditional expected fields, not a hardware-run receipt. With an
unproved range/shape, `direct_q8_matvec=0` and the retained conversion equals
the conservative bound.

Retain the existing per-die fields alongside this record:
`actual_scheduler_bytes`, `workspace_and_transient_bound`,
`backend_private_bound`, `token_temporary_bound`, `shared_conversion_bound`,
`admitted_bytes`, `cap_bytes`, `headroom_bytes`; and the existing expert tensor,
workspace/free reserve, physical memory, host/cgroup/swap and cleanup receipts.

## Local validation

Validation is CPU execution only, including for Vulkan-linked test binaries.
No Vulkan hardware enumeration, full model run or numerical qualification was
performed by this coding task. Existing mechanism/numerical/quality statuses
are unchanged.

Builds use `/tmp/cmake-3.31.6-linux-x86_64/bin/cmake`, existing pinned GGML at
`/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d`,
`FETCHCONTENT_FULLY_DISCONNECTED=ON`, Release, CUDA off, tests/native experts and
portable FFN/GDN/QSA/model on, worker TLS off, GGML OpenMP off. CPU build:
`/tmp/strata-head-admission-cpu`. Vulkan build:
`/tmp/strata-head-admission-vulkan`, additionally Vulkan on, existing
`/usr/bin/glslc`, prefix `/tmp/strata-spirv-headers-prefix`, and CXX include flag
`-I/tmp/strata-spirv-headers-prefix/include`. No dependency install or pinned
source modification. The first Vulkan build failed to find
`spirv/unified1/spirv.hpp`; adding that existing retained include path corrected
the isolated build configuration.

Targets: `portable_head_admission_test portable_memory_admission_test
portable_head_test portable_qsa_memory_test portable_model`, parallel 2.
Focused test command:
`ctest --test-dir BUILD --output-on-failure -R '^portable_(head_admission|memory_admission|head|qsa_memory)_test$'`.
The new test uses metadata-only tensors and fake physical observations, covering
exact contiguous Q8/F32/T1/range-positive, equality/insufficient range, missing or
ambiguous PCI/query proof, unsupported shape/dtype/contiguity/backend, retained
HC and other workspace, shared-ledger arithmetic, QSA peak overlap and additive
expert terms/cap refusal. Portable supervisor tests:
`python -m unittest discover -s tools/validation-harness -p 'test_portable*.py'`.

Observed results on Valinor: CPU and Vulkan target builds PASS; focused C++
tests 4/4 PASS in each build; portable supervisor tests 52/52 PASS; diff whitespace
check PASS. Build logs are retained locally as
`/tmp/head-admission-cpu-build2.log` and
`/tmp/head-admission-vulkan-build2.log`; configure logs use the matching
`/tmp/head-admission-*-configure.log`/`*-reconfigure.log` names. Initial failed
Vulkan build: `/tmp/head-admission-vulkan-build.log` (missing retained SPIRV
include path, corrected as described above). No hardware execution or numerical
comparison was attempted. No unresolved local build/test blocker remains;
actual PCI query/guard activation with the old 480 plans remains unvalidated.
Unchanged 1152/die plans are excluded by the cap arithmetic above.
