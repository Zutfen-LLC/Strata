# CUDA and Vulkan parity roadmap

Tracking: [umbrella #65](https://github.com/Zutfen-LLC/Strata/issues/65).

Source baseline: `3dc736044b9b276a44729dfe708b740238d9fc3a`.
This is an **unconfirmed static-analysis roadmap**, not a new hardware result.
Follow Strata's model-specific CUDA/HIP/SYCL design unless a backend constraint
or measured benefit justifies a difference. Product parity includes supported
behavior and execution methodology, not only generated tokens or speed.

## Status update: first GDN slice

As of 2026-10-29, [PR #72](https://github.com/Zutfen-LLC/Strata/pull/72)
implements the first opt-in slice of [issue #66](https://github.com/Zutfen-LLC/Strata/issues/66).
Source and hardware qualification have been reported; the PR is not merged.
Its device-owned GDN production policy removes full recurrent/conv/PLE-history
mirror readbacks, retains explicit eager diagnostics, and uses an integer
logits-status guard through an opt-in integration with the pinned Vulkan backend.
Residual/FFN transfers and separate device commit copies remain.
This is a bounded implementation checkpoint, not full CUDA/Vulkan parity.
The baseline findings and first-slice plan below remain historical; the other
roadmap items remain open.

## Current boundary

The portable Q8 qwen4exp path implements GDN, QSA, PLE, HC, routed/shared MoE
and the output head. It remains an opt-in serial token-ID executable, with
context at most 256 and greedy sampling. Persistent dense weights and GDN
state are default-off trials. Broader quant-role support, long-context
behavior, native sampling, conversation reuse and serving integration are
separate gaps.

The active native reference is **Verifier windows**: a native pack bypasses
the older SessionGraphs loop
([selection](../src/program/generate.cpp#L7219)).
Verifier retains device scratch and captured execution
([arena](../src/core/verify.cpp#L807),
[window reuse](../src/core/verify.cpp#L2773)).
SYCL likewise retains model-specific layer state
([ownership](../sycl/src/core/session.cpp#L63)).
Similar flags alone do not establish equivalent paths.

| Work | Current gap | Next result |
| --- | --- | --- |
| [GDN #66](https://github.com/Zutfen-LLC/Strata/issues/66) | Resident state still downloaded and mirrored each token | Device-authoritative normal inference; explicit eager diagnostics |
| [QSA #67](https://github.com/Zutfen-LLC/Strata/issues/67) | Fresh attention scheduler and active-prefix KV uploads | Bounded execution reuse, then persistent device KV/pool/tail |
| [Scheduling #68](https://github.com/Zutfen-LLC/Strata/issues/68) | Shared work precedes blocking dispatch; serial expert entries/devices | Route readiness, useful shared overlap and grouped expert work |
| [Prefill #69](https://github.com/Zutfen-LLC/Strata/issues/69) | Full T1/head execution for every prompt position | Small native-style chunks; omit unused vocabulary projections |
| [Product breadth #70](https://github.com/Zutfen-LLC/Strata/issues/70) | Experimental token-ID feature subset | Explicit supported/experimental/missing capabilities |

## First slice: GDN production ownership

Normal PortableModelSession does not inspect the layer's host state vectors,
yet GDN downloads candidate recurrence and histories every token
([readbacks](../src/core/portable_gdn.cpp#L461)).
Use the existing resident allocation and cached graph. Remove full state
downloads from normal resident execution; retain the smaller residual exchange
needed by the current FFN seam. Device copies may remain initially where
required by the backend. Do not build a generic state framework to remove them.

Add an explicit eager diagnostic mode retaining today's full mirrors,
intermediate finite checks, trace/state observers and rejected-candidate
tests. Legacy const-vector observers must not return stale mirrors in
production. Keep them diagnostic-only, or provide a checked snapshot with
documented synchronization, failure and lifetime semantics
([existing getters](../src/core/portable_gdn.cpp#L534)).

This is a deliberate error-policy distinction. Production may advance device
state before a later failure; poison the layer/session and require reset/replay.
Publish position and results only after success. It need not retain diagnostic
standalone-layer rollback. Native CUDA T1 windows self-commit
([commit](../src/core/verify.cpp#L3264)) and guard logits on device
([guard](../src/core/verify.cpp#L2300)), rather than maintaining full GDN host
mirrors. Keep a bounded device logits-status guard in the production slice;
do not replace mirrors with no numerical failure detection.

Acceptance for #66:

- No full recurrence/conv/PLE-history downloads during normal resident steps.
- Eager diagnostics preserve state/trace values, finite failure behavior and
  rollback tests; trace requests select or require this mode.
- Matched same-backend successful execution preserves arithmetic and routes.
- Backend/provider/nonfinite failure poisons production; another step refuses.
  Reset/replay recovers, and reset failure remains poisoned.
- Memory admission, inspection behavior and readback/timing counters describe
  each policy. Diagnostic read counts are not product correctness requirements.

## Execution work after GDN

**QSA:** reuse bounded attention execution first, then append KV/indexer state
on device. Project/pool/combine graphs already persist; attention does not
([construction](../src/core/portable_qsa.cpp#L464)).
Test partial blocks, pooled spare/tail behavior, reset A/B/A, ties, capacity
and sparse selection transitions. The assembled 256-position path does not
qualify selection beyond the 2051-cell cutoff. Bound retained graph/workspace
memory instead of retaining every possible shape without admission.

**Shared and routed experts:** follow native route publication and shared
fork/join
([native scheduling](../src/core/verify.cpp#L1982)).
Preserve original masks, slot order and exactly one weighting/merge.
Group entries without silently changing the current hybrid Q8 arithmetic.
Measure actual overlap and exposed wall time. CUDA/HIP already use different
shared-stream defaults where measurements justify it
([backend policy](../src/core/verify.cpp#L254)); Vulkan needs its own evidence.

**Prefill:** follow the existing model-specific
[Prefill decomposition](../include/strata/prefill/prefill.hpp).
Start with small single-sequence chunks. Compare serial/chunked states at
tails and arbitrary segmentation, including GDN, QSA and PLE history. Preserve
final-prompt logits for first-token selection, and compute requested diagnostic
rows. Skip other unused vocabulary projections. Multi-request batching and
MTP are later work, not prerequisites.

## Preserve correctness accommodations

The [per-head QSA attention](../src/core/portable_qsa_graph.hpp#L6)
works around a documented pinned-backend regression. Keep it until the
affected regression and all-head/cache-length tests pass. Exact BF16 indexer
widening and stable routing/selection tie handling also remain deliberate
policies. Scheduler placement does not remove coordinator host transfers,
and intentional CPU experts are distinct from implicit CPU graph fallback.

## Product breadth and completion

#70 tracks quant-role coverage; context/RoPE/KV options; sampler/stops;
conversation checkpoints; serving/cancellation/Windows lifecycle;
multi-GPU stages/capacity; independent request batching and MTP; and model
families/vision. Implement those as focused changes, with truthful unsupported
configuration errors. Dense, head and expert quant support must be listed
separately.

Reuse [#23](https://github.com/Zutfen-LLC/Strata/issues/23) for worker
qualification, [#26](https://github.com/Zutfen-LLC/Strata/issues/26) for GPU
transport security, and [#11](https://github.com/Zutfen-LLC/Strata/issues/11)
for distributed latency. This roadmap does not duplicate their protocols.

Use focused fixtures and existing harnesses. Keep operator references,
same-backend replay, cross-backend numerical screens and task accuracy
separate: matching tokens alone proves none of the other gates. Performance
comparisons must match arithmetic, residency, placement, process lifecycle
and timing boundaries. Record specific backend limitations or measured
deviations in each child issue. This document changes no runtime defaults.
