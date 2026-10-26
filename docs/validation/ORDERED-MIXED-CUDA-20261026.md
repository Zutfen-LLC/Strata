# Ordered CPU and network-CUDA bridge

Issue #18, bounded follow-on to `01539e99b0b5a4b83f4270605b4bda0ab0cf6da0`, on
draft PR #10. This implements fixed execution delegation while retaining full CPU
resident ownership. It does not reduce the arena or change GPU placement policy.

## Contract

The CPU group's setup partitions, trusted manifest and partition identity remain
the full complement of the unchanged local GPU placement. A separate immutable
bitmap contains the union of trusted network-GPU plans. It must be a subset of
that backing. GPU plans must be disjoint and match the trusted model/layout/assets.

`holds` describes CPU backing; `executes` excludes the delegated GPU subset. Only
one resident CPU endpoint is supported in mixed mode. CPU setup requires an idle,
complete forever-resident cache and a cache hit. Empty/mismatched caches fail;
there is no replacement load, fallback upload or arena shrink. The existing worker
service already refuses a different setup identity while retaining its cache.

The CPU group exposes completed original-entry rows without reducing them in this
mode. The GPU group assembles the disjoint/exhaustive CPU and GPU row union, then
calls `ExpertWorkerService::accumulate_partial` once in original entry order. The
dispatcher adds that one result to `RemoteExpertOpt` once. All network results are
finished and checked before commit; the dispatch guard drains remaining exchanges
on early failure. Unsupported row peers fail rather than return weighted partials.
The group's existing keeper protects GPU acquisition; a final serialized probe
checks CPU readiness before setup succeeds. No additional CPU socket owner exists.

This orders the union that previously formed the CPU contribution. Existing local
GPU/helper reduction and outer merge stay in place. It does not establish one
new global reduction over every local and network GPU entry. Keep local placement
fixed when comparing configurations.

## Attach arguments

Keep the existing coordinator's exact model/native source, trusted manifest,
profile, local device order, local cache counts, CPU budgets, packed/raw selection,
request settings, endpoint bindings and deadlines. Use this argument fragment in
the retained invocation, with already verified values:

```text
--remote-cpu-worker <existing-CPU-IPv4:port>
--remote-cpu-local-manifest <unchanged-trusted-manifest>
--remote-cpu-resident
--remote-cpu-entry-rows
--remote-gpu-worker <existing-GPU-IPv4:port> <trusted-disjoint-plan>
[repeat --remote-gpu-worker for the other fixed plans]
```

Use exactly one CPU group endpoint. Replace the legacy CPU endpoint option with
the group option; do not combine `--remote-cpu-experts` with it. Mixed mode excludes
local upload fallback, capture and regroup diagnostics. Network CUDA automatically
negotiates entry rows in this combination. The retained other worker-mode
restrictions still apply. No new runner, listener, tunnel or worker-lifecycle
command is introduced. This is an argument fragment, not a verified live launcher.
The coordinator and CUDA workers need compatible new builds; an already capable
CPU resident service is reused, without replacing/reloading it.

Actual resident reuse is **pending operator verification** of complete bytes,
cache hit and unchanged load/allocation/unload counters. This task performed no
live attach, resident operation, model inference or deployment.

## Validation

Local GNU C++ 16.2.1 CPU-only build, using existing local ggml sources. Five focused
synthetic loopback tests passed in 23.92 s:

- `gpu_frame_test`: 0.12 s.
- `gpu_worker_test`: 1.57 s.
- `gpu_worker_group_test`: 8.27 s.
- `cpu_worker_group_test`: 11.27 s.
- `cpu_worker_group_packed_test`: 2.66 s.

The new CPU fixture retains a full resident cache while delegating execution,
checks unweighted row bytes, separate backing/execution ownership, unchanged
load/allocation/unload counters, rejected foreign delegation and rejected resident
partition mismatch, plus fresh zero-CPU-work windows and keeper survival beyond
its service idle deadline. Mixed assembly checks compare the production reducer
against an independent ordered reference and refuse gaps, overlaps, foreign rows
and nonfinite CPU values before commit. Existing wire, weighted, packed and failure
fixtures remain in the focused set. One intermediate GPU-fixture assertion was
placed after its mask had been deliberately replaced by a subsequent CPU-only
window; it failed, was moved to the correct window, and the final set above passed.

No CUDA compiler is available locally. The changed dispatcher and CLI, and the
extended CUDA-only dispatcher fixture, remain **uncompiled/unexecuted locally**.
The existing `cpu_group_dispatch_test` now covers a real full CPU resident, actual
CUDA rows, delayed acquisition with the existing keeper, one ordered mapped merge,
GPU loss with CPU drain/no commit, and unchanged backing counters. These new
hardware checks are pending. The parent-supplied first-slice CUDA result is recorded
separately in [CUDA-ENTRY-ROWS-20261026.md](CUDA-ENTRY-ROWS-20261026.md); it does not
qualify this bridge.

Use an existing configured CUDA/native test build with its existing dependencies:

```sh
cmake --build "$CUDA_BUILD" --target strata strata-cuda-expert-worker gpu_frame_test gpu_worker_test gpu_worker_group_test gpu_worker_cuda_test gpu_worker_dispatch_test cpu_worker_group_test cpu_group_dispatch_test -j 4
ctest --test-dir "$CUDA_BUILD" --output-on-failure -R '^(gpu_frame_test|gpu_worker_test|gpu_worker_group_test|gpu_worker_cuda_test|gpu_worker_dispatch_test|cpu_worker_group_test|cpu_worker_group_packed_test|cpu_group_dispatch_test)$'
```

These build and synthetic fixture commands require no model load. Compile the CLI
as well as worker/test targets before attempting a live attach. Do not infer live
exactness or throughput from these fixtures. CPU and CUDA activation quantizers
and kernels differ. Ordered F32 transport/reduction avoids additional regrouping
loss but does not prove bitwise equality to the prior CPU placement; the actual
numerical acceptance gate remains separate.
