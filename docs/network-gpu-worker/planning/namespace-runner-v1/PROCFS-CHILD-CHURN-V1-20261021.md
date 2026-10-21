# PROCFS-CHILD-CHURN-V1-20261021

Active continuation from `984b6fb6f8b608211feb55834a2353a349a55954`.
This supersedes older procfs convergence rules, focused gates, request/DONE/logit
expectations and active-time/usage ledgers. Apply only this continuation's steps.
Keep [GPU-RESERVE-WARNING-V1](GPU-RESERVE-WARNING-V1.md), frozen
[primary sizing](PRIMARY-STARTUP-SIZING-CORRECTION.md) and the other requirements
of the [original budget](../EXECUTION-BUDGET-bf51f09e.md). Earlier evidence and
documents remain historical; they do not supply additional inference attempts.

## Supplied failed evidence

Hermes reports the nine procfs race fixtures passed. B0 reached READY after
427.4 seconds of CPU setup. Its single GEN64 emitted 63 valid tokens, without
DONE. After 253 successful two-second samples, the supervisor refused with
`unresolved-task-children-coverage:scans=4`: 62 tasks before and after, identical
task sets, no vanished tasks. B1/G0/G1 did not run. There is no completed baseline.
The earlier task ENOENT did not recur in this run; that is not universal proof
of its absence. Exact transient-child creator provenance was not supplied.

No CUDA allocation failure was reported. CPU loaded 21369 experts / 111597465600
bytes in one arena, network weights zero, RSS 106442.4 MiB, VmSwap zero. Engine
RSS 8740.1 MiB; local free-VRAM minima 1004/432 MiB, warnings only. First logits:
993280 bytes / 248320 finite float32 values, top token 11 matching the first
emitted token. Routing contained 3025 complete records plus a 40-byte partial
tail; do not treat that as complete serving evidence. All owned work was gone,
CPU released, no remote cache and GPUs back to baseline. Preserve the private
report/archive; supplied archive SHA-256:
`7087cb646615ed0f7db0b2dd85a3c060b27e664878d05293698a7e347709914c`.
These are supplied observations, not newly inspected private logs or author PASS.

## Source findings and narrow correction

The previous `OwnedProcess.children()` required two identical child maps. That
mistakes ordinary forking/exiting children for missing task coverage even when
every required task read succeeds. The replacement retains the same four-scan
bound, fresh directory descriptors, before/after enumeration and task ENOENT
reconciliation, but converges on **complete stable task sets**, not identical
child maps. Two complete scans of the same stable task set suffice. All child
PIDs observed during the bounded scans are retained for ownership reconciliation.
Changing child maps alone do not refuse coverage. Continued task-set churn,
still-present tasks with missing children, EACCES and malformed data still fail.

The adjacent assumption was also real: a listed child could exit before pinning,
between liveness and status/cgroup/namespace reads, or during watchdog adoption.
The shared `ProcessExited` distinction requires a held pidfd confirming exit.
For newly discovered descendants only, ESRCH from pidfd lookup means no instance
exists at that lookup; return no identity and send no signal. ENOENT/ESRCH during
process-entry pinning is terminal only with the held pidfd confirming exit.
Live missing entries, permissions, malformed stat and parent/start/executable
identity mismatches remain failures. A failed monitor's own namespace read is
not excused by a descendant's exit. Root/required worker identity loss remains
fatal. New children are still namespace/parent/start/executable/pidfd checked.

`OwnedTree` retires only pidfd-confirmed exited descendant handles, so the
16-process tracking bound does not become a lifetime count of dead transient
children. A reused numeric PID receives a fresh ownership/identity proof; no
old handle signals its replacement. A parent mismatch fails rather than adopting
a foreign PID. Live observed children are pinned and recursively discovered;
gone candidates contribute no fabricated ownership or zero-valued telemetry.
Sampling omits only confirmed exited nonrequired descendants. Other read/schema/
identity failures propagate. The original memory/event/swap guards are intact.

Watchdog adoption still proves direct-child relationship with waitid. ECHILD
means it is no longer that guardian's child; no arbitrary PID is signalled.
Exited zombies remain for actual waitpid reaping. The exit-between-wait-and-pin
race returns no identity. If a historical watchdog PID slot is reused by a new
proven direct child, the new handle stays cleanup-tracked and the old PID-only
reap receipt is explicitly incomplete rather than silently overwritten as
success. Cleanup retains partially discovered pinned descendants even if their
parent exits. TERM/grace/KILL/reaping and original deadlines are unchanged.

Source attribution matters: `outer.gpu_free()` launches and reaps a short-lived
management-query subprocess. During monitoring it belongs to the monitor;
workload and monitor are sibling children of the watchdog. Therefore that
query is not in the workload-rooted normal collection tree. Watchdog cleanup
does discover monitor descendants, so its lifecycle must work there too.
Admission/baseline-return queries run synchronously under the guardian and are
reaped by `gpu_free()`. Controller transports are owned separately. A static
search of pinned product paths found no identified OS-process fork in the
generation/worker path. `verify.cpp`'s local `fork` lambda records CUDA stream
events; its name is not evidence of a process fork. A conversation-file test
does call OS fork, but is not this product run. This does not exclude external
library/runtime behavior, or establish which process created the reported child.
Do not claim that the model itself forks.

The guarantee is **safe periodic discovery and identity-pinned tracking/cleanup**,
not an atomic global procfs snapshot or observation of every process born and
exited wholly between samples. No moment of total process-tree quiescence is
required. No product change, new monitoring/timer framework, longer scan limit,
guard relaxation or broad exception swallowing is introduced. Existing task,
read, journal, sample and process caps remain. Failures retain path/identity
context; unresolved ownership remains a failure, not invented coverage.

Twelve focused mocked methods cover stable-task changing maps, surviving child/
grandchild discovery, gone-before-pidfd and exit-during-pin, genuine read/schema/
identity failure, safe PID-slot reuse, terminal descendant sampling, required
monitor namespace reads, adoption races, bounded termination and request bytes.
The old request fixture and cleanup mock shape are aligned with these changes;
their campaigns are not requested. Tests are authored **NOT RUN** here. Static
review followed these concrete lifecycle paths and descriptor/partial-cleanup
ownership. No test, build, product/model, live proc/GPU query or fleet operation
was executed in this coding task. This is source review, not runtime validation.

## Frozen inputs and amended schedule

Product `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`; main
`34a59d83a51bdb3f0299d65306740bd357818715`; PR #2 draft and unmerged.
Primary 960, helper 2247, CPU full 21369, CPU uncovered 18080, remote 3289.
CPU memory-budget 106428 MiB; one 111597465600-byte full arena including remote
duplicates. Remote assigned payload 17176473600 bytes. Existing plan 92156 bytes,
SHA-256 `a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578`.
All retained binary/source/plan/input bindings were already validated by Hermes.
No resizing, replanning, manifest regeneration, weight hashing, preparation
transfer, product rebuild, merge or deployment is authorized by this correction.

B0 (cold CPU startup) and G0 (cold remote cache startup) are single-A smoke:

```text
GEN 64 9419
QUIT
```

B1 (resident CPU) and G1 (warm remote GPU cache) keep identical full A/B/A:

```text
GEN 64 9419
GEN 64 14773,6353
GEN 64 9419
QUIT
```

This adds **8 requests / at most 512 generated tokens**. Two interrupted requests
already started; charge each full 64-token allowance: **2 requests / 128 charged
tokens** before resume. The observed 24+63 tokens are only a lower bound, not a
final total. The complete amended schedule reaches at most **10 requests / 640
charged tokens**, below the current **11-request / 704-token ceiling**. Do not
reset counters, extend EOS-shortened requests or spend the unused allowance on
another attempt. The older 12/768 ceiling does not replace the current ceiling.

Require READY and one normal finite product DONE plus `PHASE.first-logits.0`
for each B0/G0. Require three ordered normal finite DONEs plus
`PHASE.first-logits.0/.1/.2` for each B1/G1. Each engine must exit 0 with complete
supervisor cleanup. Product DONE counts are distinct from supervisor DONE
receipts. Reconcile token records with DONE actual counts <=64, prompt lengths
1 for smoke and 1/2/1 for A/B/A, zero reused tokens and full prompt reads. ERR,
cancelled/nonfinite/partial output, unexpected requests, missing DONE/files or
nonzero exit fail acceptance. Routing partial tails are incomplete evidence.

Every logit file must be exactly 4*vocab finite float32 bytes, using trusted
vocab; short/missing/write-error files fail even with rc 0. Compare all three
matching B1/G1 pairs: centered relative L1 <=.03 and
KL(softmax(B1)||softmax(G1)) <=.03 nats at temperature 1. Center each by its mean;
denominator max(sum(abs(centered B1)),1e-12), stable double log-softmax. Preserve
within-mode drift maxabs <=1e-5*max(1,maxabs(reference)): repeated A at B1 .0/.2
and G1 .0/.2, cold/warm matching A at B0 .0/B1 .0 and G0 .0/G1 .0. G0 has no B
or repeated-A files; do not require or invent them. Report raw/centered maxabs,
RMSE, top1/margin and token agreement. No bitwise/final-token/full-parity claim.

B1/G1 are the complete like-for-like timing streams, with actual token counts
and EOS/trajectory caveats. B0/G0 are not equal-length timing windows. Separate
startup, CPU setup, selective transfer/cache/READY, engine wall, prompt/decode,
request/wait, worker service and helper staging/wait times. No speedup gate.
CPU resident reuse and G0 assigned-transfer/G1 zero-transfer checks stay exact.

## Complete Hermes continuation

Reuse Hermes's existing approved ROLE_CONTROLLER and transport bindings without
relocating orchestration. Authorized controller SSH orchestration is permitted.
Keep existing ROLE_ENGINE_A (inferswarm03, two local GPUs), ROLE_GPU_B
(inferswarm04), ROLE_CPU_C (inferswarm05), approved interpreter and trusted
namespace bindings. No local tests/builds/models on either user desktop.
Focused fixtures run on 05; products only on 03/04/05. Do not install dependencies
or change security/network settings, permissions, services, cgroups or swap.

1. Fetch this amendment and refreshed integration/runner manifests at its exact
   publication commit through existing authorized GitHub access. Verify author
   receipt SHA-256/byte pins. Retrieve changed `resource_telemetry.py`,
   `resource_guard.py`, integration `runtime.py`, `commands.py`,
   `test_procfs_child_churn.py`, `test_procfs_task_race.py`, `test_integration.py`
   and README, preserving planning-relative FILES_ROOT paths. Verify retained
   small dependencies against the manifests or fetch only missing listed files.
   No mixed revisions, repository clone, new supervisor, dependency download or
   renewed provenance campaign. The runner manifest replaces its superseded
   task-race amendment row with this amendment and has 32 entries; the old file
   remains in Git history. Do not invoke the old runner/campaign.

2. On inferswarm05 invoke exactly once with the approved interpreter:

   ```sh
   "$PYTHON" -B -I "$FILES_ROOT/supervisor-integration-v1/test_procfs_child_churn.py"
   ```

   The entry caps itself at 128 MiB address space, 5 CPU seconds, 10 wall seconds
   and 1 MiB regular output. Bound captured stdout/stderr to 1 MiB each with
   existing tooling. Record hashes, exact argv, actual start/end, rc and bounded
   output. Require exit 0, zero failures/errors/skips and exactly:
   `PROCFS_CHILD_CHURN_FIXTURES_PASS_V1 tests=12 live_signals=0 workloads=0`.
   No old 69+3, warning, task-race-nine, sizing or listener campaign reruns.
   On focused failure preserve evidence and stop. On PASS **automatically
   continue the actual B0/B1/G0/G1 trial**, without another approval or prompt.

3. Preserve both prior interrupted-run reports/archives and unrelated services.
   Confirm earlier owned work remains gone using existing scoped receipts.
   Reuse all accepted frozen bindings/assets/binaries. New private output roots,
   one new cold remote cache for G0 reused by G1. Engine SETUP loads the fresh
   full CPU arena; no standalone preload. One CPU worker/arena persists across
   sessions, new engine each phase and new remote GPU process per G phase.
   Keep serial order, stop/reap transitions and 11-second gaps after B0/G0.

4. Charge **2390.0 active seconds** total, leaving **3010.0 of 5400 seconds**.
   All-stopped retrieval, coding, focused fixtures and handoff are excluded.
   At readiness, before new owned transport/guardian startup, record controller
   monotonic T and wall correlation. Use existing
   `Controller(..., start=T-2390.0)` with all-owner finally cleanup. Deadline is
   T+3010.0; existing 45-second cleanup reserve gives launch/TERM cutoff
   T+2965.0. No new timer framework, old wall cutoff, clock pause, counter reset
   or cross-host raw-clock comparison. Include new owned startup and active
   teardown after T. Engine caps remain 1200/600/1800/900 seconds, GPU G0/G1
   1900/1000, CPU admission+7200, CPU setup 600000 ms, GPU setup 1200000 ms and
   idle 1800000 ms. Refuse a phase whose full required envelope cannot fit;
   report NOT RUN rather than extending caps or spending the run on preparation.

5. Execute exactly the phase streams and complete-output/numerical checks above.
   Preserve strict local prelaunch/pre-exec admission 1152/512 MiB, warning-only
   valid local runtime reserve crossings, all required telemetry, parent/start/
   executable identity, host/cgroup memory/events, fixed-admission SwapFree
   decline >64 MiB, RSS/anonymous ceilings, pidfds, leases/watchdogs and deadlines.
   Remote GPU reserve 4096 MiB, fixed-plan fit, cache 18432 MiB, cold disk 22.25
   GiB and running disk 4 GiB stay strict. Genuine product/CUDA failure stops
   normally. No hidden CPU retry, guard bypass, auto-sizing or failure injection.

6. Reconcile disjoint/exhaustive ownership and positive actual execution on both
   local GPUs, remote GPU and CPU. G phases serve only CPU-uncovered 18080 while
   full 21369 remains in the same arena/bitmap/partition. Require one allocation,
   warm reuse 111597465600 bytes and zero new CPU load/hash/allocation/expiry/
   unload/network weights between sessions. G0 assigned/cached/logical expert
   reads/network/newly persisted bytes each 17176473600; G1 reads/network/newly
   persisted zero, reuse 17176473600. Distinguish physical I/O/transport/staging/
   RSS/GPU allocation. Reconcile route/client/worker/helper entries and remote
   final counters, classifying normal EOF separately. No whole-model mirror or
   unassigned transfer. Record local minima/crossings/warnings. Missing or partial
   evidence stays incomplete, never inferred PASS.

7. Correct ordinary in-scope launcher/config/evidence-expectation mismatches
   within these exact settings, bytes and caps without unnecessary approval.
   Record corrections. Do not patch product behavior to satisfy the harness,
   restart a failed focused test, weaken guards or add an inference attempt.
   Any required failure stops dependent phases. Preserve diagnostic creator/
   identity context privately when actually available; do not invent provenance.

8. Release owned engines/GPU tasks, require baseline return <=30 seconds and
   bounded TERM/grace/KILL/actual parent/adopted reaping. Close evidence within
   the remaining active allowance, <=256 MiB per role/controller, no weights.
   Failure stops CPU; otherwise default-stop unless prior explicit finite
   retention applies (earliest success+30 min, actual idle expiry, admission+2 h).
   No broad cache deletion or unrelated-service interruption. Return phase
   outcomes, actual/cumulative active time and request/token charges, finite
   outputs/numerical comparisons, CPU reuse, selective cold/warm transfer,
   ownership/guard diagnostics, cleanup and evidence hashes privately. Stop at
   report; no build, merge, deployment, permanent service or cosmetic retry.
