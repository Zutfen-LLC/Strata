# Integration interface: replacement monitor into the original supervisor

Audience: the separately approved Codex integration pass, dispatched against
this commit. This document is static review material; it authorizes no
execution. Entry points and signatures below are quoted from the published
monitor artifact at `a3b706deae4101e90592bed58701df20f9bd4da2`
(docs/network-gpu-worker/planning/resource_guard.py, resource_telemetry.py).

## Original lifecycle (as published in the sanitized sources)

1. Controller `orchestrate-ROLE_CONTROLLER.sh` opens the relay bridge, launches
   the CPU-worker driver, waits for STATUS=LAUNCHED then a
   "strata-worker: listening" line, then launches the engine driver with
   GLOBAL_T0.
2. `driver05-ROLE_CPU_C.sh`: admission (MemAvailable >= 126345216 KiB, cgroup
   headroom), SWAP_BASE snapshot, setsid+timeout-wrapped worker launch,
   worker-identity receipts, then a 2 s monitor loop with three memory tests
   (RSS ceiling, headroom floor, NEW-SWAP vs SWAP_BASE − 65536 KiB) plus
   cgroup oom/high event tests; any violation TERMs the wrapper once and sets
   STATUS=RESOURCE-STOP.
3. `driver03-ROLE_ENGINE_A.sh`: per-phase admission re-checks (including the
   same SwapFree-vs-baseline−65536 KiB strict test against mem-baseline.txt),
   timeout-wrapped engine sessions B0/B1/G0/G1 with gate files, per-phase
   sampler subshells, rc/epoch receipts, verdict propagation to the
   controller.
4. `driver04-ROLE_GPU_B.sh`: launch/stop subcommands for the GPU worker with
   TERM -> 20 s poll -> KILL escalation (the only KILL in the set) and final
   counter collection.

## Codex integration entry points

E1. driver05 monitor loop replacement (primary). Replace the hand-rolled
    loop (sanitized driver05 lines 75-107) with the monitor seam:
    `resource_guard.monitor(tree, worker_identity, namespace_receipt,
    hard_deadline, journal, admission_swapfree_bytes=...,
    admission_monotonic=...)` after `resource_guard.capture(...)` at
    admission time. Keep: STATUS/RESOURCE-STOP writing, violation log naming,
    the launch CLI and wrapper relationship (setsid nohup timeout
    TERM/15s), and the final-state block (lines 108-113).
E2. driver03 check_admission/sampler. The replacement's CpuGuard.check covers
    the same strict swap test with baseline preservation; the per-phase
    sampler subshells map onto host_sample/Observed polling. Engine-side
    process identity (EPID) uses the same OwnedProcess pin by PID+parent.
E3. Orchestrator abort() ownership set. Unchanged targets: STOP-03, driver04
    stop, forward PIDs, CPU wrapper PID — all positive-PID TERMs of recorded
    owned processes; the monitor's stop_owned adds grace + reap that the
    original lacks.

## Required interface: supplying the original admission baseline/deadline

The replacement monitor REFUSES to run without original-run receipts:
`monitor(...)` raises `CoverageError("original-admission-baseline-receipt-required")`
unless BOTH are supplied:

- `admission_swapfree_bytes`: the original run's admission SwapFree in BYTES
  (KiB × 1024). Historical source: driver05 admission.txt SwapFree line,
  captured at admission (wall-clock receipt).
- `admission_monotonic`: the admission instant on the integrating host's
    time.monotonic() clock, satisfying `admission_monotonic <= started`.
  Historical source: worker-launch.epoch / T0.txt (wall clock only).

Important: the original run recorded WALL-CLOCK receipts only. No monotonic
receipt exists for the B0 run, and none can be reconstructed post hoc — a new
integration run must capture `time.monotonic()` at its own admission instant
(drivers already stamp epochs; add the monotonic stamp alongside). The B0 run
also never captured VmSwap, pswpin/pswpout, PSI or the CPU worker's cgroup
path; these are observable only in a new run. No historical namespace or
pidfd ownership proofs exist; the replacement's capture() receipts are new
instrumentation, not historical facts.

`hard_deadline`: earliest original worker/trial/phase cap on the same
monotonic clock INCLUDING 15 s cleanup, validated as
`started + 15 < hard_deadline <= admission_monotonic + 7200`. For a CPU worker
under the original budget: launch_epoch + 7185 (absolute cap) is the natural
anchor; the 7245 s loop bound in driver05 already encodes cap+60.

## Preserved-guard contract (must survive integration)

- Strict SwapFree decline test preserved by the replacement
  (`free < swap_baseline - SWAP_DECLINE_BYTES`, SWAP_DECLINE_BYTES = 65536 KiB
  in bytes; exactly 65536 KiB decline does NOT stop).
- Original cgroup-guard behavior preserved (oom/high event increase -> stop;
  absent file reported unavailable, never fabricated zero).
- Supervisor MUST gate SETUP on the ADMITTED marker (per monitor docstring:
  "Supervisor must gate SETUP on ADMITTED marker") and retain an independent
  watchdog/reaper; the monitor is a blocking library seam, not a process.
- Existing supervisor behavior that is real and stays: admission checks,
  STATUS protocol, timeout-wrapped launches, single-positive-PID TERM policy,
  abort() owned-target cleanup, driver04 TERM->KILL escalation.
- Still needing implementation (never existed in the original): launch-release
  gating (no admission-gated SETUP release exists — engine connect triggers
  arena load), independent watchdog/reaper ownership checks, monitor-side
  grace/escalation, per-run monotonic receipts.

## What integration must NOT do

No threshold relaxation, no retry of the model trial, no live validation in
the integration commit itself, no changes outside the approved monitor +
driver seam, no alteration of PR #2 or the bf51f09e implementation pin.
