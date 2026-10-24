# CPU-worker RPC latency: Hermes handoff

Historical, revision-specific handoff. Its pinned source gate and build recipe apply only to the
candidate described below. For the current branch use [the cohesive worker handoff](../cpu-worker-lan/HERMES.md);
do not retarget these older CMake caches to a new source directory. Earlier evidence is retained here.

This is a source-validation and later controlled-measurement handoff. The candidate is based on
`9bf66c042c06e1b3a1840d131b7444333ce9e502`. It does not change expert arithmetic, ownership,
concurrency, polling, deadlines, frame shapes or GPU accepted sockets. Do not combine it with the
unpublished multi-CPU work at `b770d280a65893d1df92b76b679380d76afade3b`.

Hermes found one compile-only defect in published candidate
`66235df177368a6e9e3b79afc9f6838cc4af1f71`: `expert_tier.hpp` did not directly include
`rpc_timing.hpp`. On CPUworker05 the worker, `rpc_timing_test`, `worker_socket_test`,
`expert_worker_test` and `expert_worker_e2e_test` built and passed, while `expert_tier_test` failed to
compile. The CUDA `strata` target failed for the same reason on host03. The additive correction includes
the missing direct dependency; those two failed targets still require remote validation.

The two controls are independent and default off:

- Worker `--tcp-nodelay` requests checked `TCP_NODELAY` on each accepted CPU-worker descriptor. Omit it
  for baseline behavior. Client-created sockets retain their existing `TCP_NODELAY` behavior.
- Worker `--rpc-timing` and engine `--remote-cpu-rpc-timing` enable bounded aggregate histograms. Omit
  both for the no-instrumentation path. No RPC or STATUS wire shape changes.

All durations use a monotonic clock on the host that reports them. Never subtract a worker duration from
a client duration: the hosts' clocks are not synchronized. Client `prepare` precedes publication;
`queue_wake`, `encode_send` and `receive_decode` are inside the communication thread; `finish_wait` is
only the caller's final wait; and `result_merge` is after successful joins. These overlap the existing
client `request wall`, engine/GPU work and one another as documented; they are not additive network time.
Worker `request` runs from receipt of a complete INFER frame through reply send and contains
`decode_validate`, `quant_prepare` (scratch allocation, activation quantization and job preparation),
`pool`, `weighted_reduce`, and `encode_send`. Existing `cpu_service_ns` is the same pool boundary and is
nested inside worker `request`. Histograms retain only log-scale counts plus count/total/max; reported
percentiles are bucket upper bounds. Calls, failures, bytes, tokens, routed entries, distinct experts and
T=0..8 shape counts make the two host reports matchable without cross-host timestamps.

## 1. Source validation only

Do this first. It does not touch a running service, model asset or retained binary.

```sh
SRC=/path/to/clean/candidate/Strata
test "$(git -C "$SRC" rev-parse HEAD^)" = 66235df177368a6e9e3b79afc9f6838cc4af1f71
git -C "$SRC" merge-base --is-ancestor 9bf66c042c06e1b3a1840d131b7444333ce9e502 HEAD
git -C "$SRC" status --short
git -C "$SRC" diff --check HEAD^
git -C "$SRC" diff --name-only HEAD^
```

Confirm the correction changes only the affected public headers and this documentation.
Do not build or run from the live worker checkout.

## 2. Approved split build and focused fixtures

Keep the validated split topology: build the CPU worker and all CPU fixtures on CPUworker05, and build the
CUDA engine on host03. Reuse each host's retained build directory, Makefiles generator and pinned
dependencies; do not switch generators, refresh dependencies or invent a CPU-only `strata` target on05.
Direct runtime outputs to new correction-specific directories so retained binaries are not overwritten.
Do not touch model shards, local manifests, partition files, tunnels or unrelated processes. Record old
binary paths and SHA-256 hashes first.

```sh
SRC05=/path/to/clean/correction/Strata
BUILD05=/path/to/retained/cpu-worker-build
BIN05=/path/to/new/correction-binaries
test "$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD05/CMakeCache.txt")" = "Unix Makefiles"
test ! -e "$BIN05" && mkdir -m 700 "$BIN05"
cmake -S "$SRC05" -B "$BUILD05" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN05"
cmake --build "$BUILD05" --parallel 1 --target \
  strata-expert-worker rpc_timing_test worker_socket_test \
  expert_worker_test expert_worker_e2e_test expert_tier_test
ctest --test-dir "$BUILD05" -V -j1 -R \
  '^(rpc_timing_test|worker_socket_test|expert_worker_test|expert_worker_e2e_test|expert_tier_test)$'

SRC03=/path/to/clean/correction/Strata
BUILD03=/path/to/retained/cuda-build
BIN03=/path/to/new/correction-binaries
test "$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD03/CMakeCache.txt")" = "Unix Makefiles"
test ! -e "$BIN03" && mkdir -m 700 "$BIN03"
cmake -S "$SRC03" -B "$BUILD03" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN03"
cmake --build "$BUILD03" --parallel 1 --target strata
```

Preserve configure/build logs and each fixture's output. Report passed, failed and not run separately.
`rpc_timing_test` is the fixed-memory enable/disable, boundary, reset and failed/empty attribution check.
`worker_socket_test` proves the accepted descriptor option in off/on arms and preserves timeout behavior.
The worker and tier fixtures cover rejected, disconnected and deadline paths plus real loopback inference.

## 3. Direct route first, using existing binaries

Hermes' direct authenticated route is independent. Establish and measure it first with the currently
running, retained binaries. Do not restart either worker and do not make the direct-route result depend on
this candidate. Hold engine/GPU placement on CPUworker02, model, prompt, seed, output cap, exact CPU
partition, pool workers, affinity and three-request arm constant. Preserve raw output and routing evidence.

## 4. Controlled candidate comparison

A worker replacement or restart releases its arena. Stop/restart only after separate approval, announce
the exact service and PID, and capture its final counters before replacement. Recreate and verify the same
exact-partition CPU residency before inference. Never silently unload. Keep unrelated services and the
direct-route tunnel untouched.

Run serial arms with the same route and workload, preserving a warmup separately from the three measured
requests:

1. Candidate binary, both new controls omitted: compatibility/no-instrumentation baseline.
2. Same binaries and configuration, worker adds `--tcp-nodelay`: socket candidate.
3. Attribution arm, worker adds `--rpc-timing` and engine adds `--remote-cpu-rpc-timing`. Choose the
   desired socket setting explicitly and label it; do not use this arm as the uninstrumented speed result.

For a strict 2x2 check, add timing-on arms for both socket settings. Otherwise the first two arms isolate
the accepted-socket change and the third explains the selected transport. Capture local and direct-remote
arms on CPUworker02 and CPUworker05 separately. Compare end-to-end tokens/s and client exchange latency
only between matched arms. Treat pool time, request wall, finish wait and phase counters as overlapping.
The worker aggregate appears only on graceful stop, so capture it at the next explicitly approved stop;
do not restart merely to obtain the report.

Success means identical output/routing and no refusal or timeout regression. This diagnostic alone does
not authorize buffer changes, quantization/reduction changes, batching, socket-poll changes, multi-CPU
scheduling, deployment, publication or a default change.
