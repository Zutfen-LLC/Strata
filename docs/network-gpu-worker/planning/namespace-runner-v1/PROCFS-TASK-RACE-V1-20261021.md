# PROCFS-TASK-RACE-V1-20261021

This is the active authorized continuation from planning commit
`d5b8f7b59a601c63bf4ca6d4323046fdd20d1ed3`. It supersedes older focused-test gates,
active-time/usage ledgers and the requirement that **all four phases** use A/B/A.
It preserves [GPU-RESERVE-WARNING-V1](GPU-RESERVE-WARNING-V1.md), the fixed
[primary sizing](PRIMARY-STARTUP-SIZING-CORRECTION.md) and all other requirements
of the [original budget](../EXECUTION-BUDGET-bf51f09e.md). Apply only this
document's continuation, not earlier documents' executable handoff steps.

## Supplied failed evidence

Hermes reportedly passed all nine focused warning fixtures. B0 then reached
READY and emitted 24 tokens from its first generation request; no product DONE
was observed. The supervisor refused required coverage with
`CoverageError:children:unavailable:ENOENT`, then TERMed the engine (exit -15).
B1/G0/G1 were cancelled. This was actual partial inference, not a completed
baseline or comparison. The exact disappearing task/cause was not recorded.
Thread disappearance is consistent with the source; it is not a newly verified
diagnosis of those private logs. No CUDA allocation failure was reported.

Supplied local VRAM minima: primary 1004 MiB, helper 432 MiB, one warning crossing
each. CPU: 21369 experts / 111597465600 bytes, one allocation, setup 426.898 s,
zero network weights and owned VmSwap, RSS 106442.168 MiB. First logits reportedly
contained 248320 finite float32 values; top token matched the first emitted
token. All owned products, guardians and transports were reaped; GPU baseline
returned within a conservative 29.012 s bound; CPU released, no remote cache.
The private report was not accessed here. Preserve it and its archive, supplied
SHA-256 `340303cb0b8d27c652e354aa3225f825e90585f4342da19de3952a093b270762`.
These are attributed observations, not new author runtime PASS claims.

## Narrow source correction and static review

The failing source is shared `resource_telemetry.py:OwnedProcess.children()`,
used by integration monitoring, descendant discovery and watchdog adoption.
Previously it enumerated task names once and required every task's children
file. A task may exit between those operations while the process remains live.
Linux can reparent its children to a surviving thread in the same process;
that recipient may already have been read. Skipping ENOENT alone would therefore
lose descendant ownership evidence. A task can also disappear after a successful
children read, without producing ENOENT.

The correction rescans the entire held task directory, before and after child
reads, with at most four scans. Each enumeration opens a fresh `.` descriptor
relative to that held directory, so repeated `scandir(fd)` does not inherit an
exhausted shared directory offset. Return requires two consecutive complete,
identical task-to-child maps with stable task sets. An ENOENT children read is
recoverable only after verifying the pinned process and observing that task
absent in the after enumeration. A still-present name, including a possible
reused TID, fails conservatively. Any task-set change invalidates convergence
and forces a full rescan, including prior recipient tasks. All successfully
observed child PIDs are retained across scans, under the original 16-process
bound, then pass through unchanged `OwnedTree.refresh()` parent/identity/pidfd
pinning and recursive discovery. A stale or reparented-to-another-process child
does not gain an ownership exemption. Short-lived process identity loss still
fails; this correction concerns disappearing tasks inside a valid owned process.

EACCES and other non-ENOENT errors, malformed children, missing task directory
or leader, stable required-entry absence, process start/executable identity
failure and unresolved coverage remain fatal. No unreadable file becomes empty.
Errors include the full proc path, PID/start/executable identity, scan number,
before/after task sets and vanished names. Existing two-second sample timing,
task/read/process caps, guard policies, watchdogs and safe cleanup are unchanged.
No new monitoring framework, sleep or deadline extension is introduced.
The monitor journal retains this reason. Watchdog adoption/descendant
discovery failures now also retain up to 2048 characters of the reason in their
cleanup receipt instead of just the exception type; identity/path/cause precede
the task lists so truncation retains the decisive context. Cleanup still fails.

Static review followed the live call sites and checked reparenting onto an
already-read task, exit after a successful read, task creation, child union
retention, bounded churn, descriptor closure and error/identity propagation.
Review corrected the fresh-directory-offset requirement and the focused
fixture's error-context assertion; the fixture requires a new descriptor per
enumeration and closure of every scan descriptor plus the held directory.
This is author source review, not independent review or runtime evidence.

The controller has no hardcoded three-request/logit expectation. The only
production request-byte assumption is `agent.py`'s engine pipe write; it now
calls `commands.requests(phase)`. No controller event protocol changes.
Pinned product `src/program/generate.cpp` reads serial GEN commands through
QUIT, emits a product DONE per request, and writes numbered first-logit files
per request (`.0`, `.1`, `.2`), independent of supervisor DONE receipts.
`Controller.run()` still requires engine rc 0 and complete cleanup. That receipt
alone never proves complete finite request output or numerical acceptance.

Nine mocked methods in `test_procfs_task_race.py` exercise the real shared
children/OwnedTree path, continuing child/grandchild discovery after ENOENT,
successful observations before task exit, new tasks and changed child mapping,
stable absence, permissions/malformed data, process loss/reuse propagation,
bounded churn and original caps, plus exact phase request bytes/logit prefixes.
All tests are authored **NOT RUN** here. No Python source, tests, builds, product,
model load, management query or fleet work was executed in this coding task.

## Frozen experiment and explicit protocol amendment

Product remains `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`; main remains
`34a59d83a51bdb3f0299d65306740bd357818715`; PR #2 is draft and unmerged.
Primary 960, helper 2247, CPU full 21369, CPU uncovered 18080, remote 3289.
CPU memory-budget 106428 MiB; one fresh 111597465600-byte arena retaining the
full CPU complement, including remote duplicates. Existing remote plan is
92156 bytes, SHA-256
`a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578`,
remote payload 17176473600 bytes. All retained source/binary/plan/input bindings
were already validated by Hermes. No resizing, replanning, manifest regeneration,
weight hashing/transfer preparation, product rebuild, merge or deployment.

New B0 is application-cold CPU startup/smoke only:

```text
GEN 64 9419
QUIT
```

B1 (resident CPU), G0 (warm CPU/cold remote GPU cache) and G1 (warm CPU/warm
remote GPU cache) each keep the complete identical stream:

```text
GEN 64 9419
GEN 64 14773,6353
GEN 64 9419
QUIT
```

This is 10 new requests / at most 640 generated tokens. The failed request
started once and produced at least 24 observed tokens; its final count is
unknown. Charge its full 64-token allowance. Cumulative charged usage after
the amended schedule is at most **11 requests / 704 tokens**, below original
12/768 caps. Before resuming, usage is 1 request / 64 charged tokens, leaving
11 requests / 704 tokens of headroom. The new schedule uses only 10/640 of that
headroom; the unused request is not permission for a retry. Never reset counters
or call 24 the final generated total. EOS may shorten completed requests; no
automatic extension, replacement request or inference retry is authorized.

Require product READY, rc 0, one normal finite product DONE and exactly
`B0.first-logits.0` for B0; three ordered normal finite DONEs and
`PHASE.first-logits.0/.1/.2` for each B1/G0/G1. Verify observed token records
against each DONE's actual count (at most 64); prompt lengths are 1/2/1,
zero reused tokens and full prompt reads. Unexpected extra requests, ERR,
nonfinite/cancelled/partial output, missing DONE/logit evidence or nonzero exit
fails acceptance. Keep supervisor DONE/cleanup distinct from product DONE.

Each logit file must be exactly 4*vocab bytes, all finite float32, using trusted
vocab. Diagnostic write errors, short/missing files fail the gate even with rc 0.
Compare all three matching B1/G1 first-logit pairs: centered relative L1 <= .03
and KL(softmax(B1)||softmax(G1)) <= .03 nats at temperature 1. Center by each mean,
denominator max(sum(abs(centered B1)),1e-12), stable double log-softmax. Preserve
within-mode drift maxabs <= 1e-5*max(1,maxabs(reference)): repeated A at B1 .0/.2,
G0 .0/.2 and G1 .0/.2; matching G0/G1 cold/warm pairs and matching B0 .0/B1 .0.
B0's absent B/second A are not missing required files. Report raw/centered
maxabs, RMSE, top1/margin and token agreement. No bitwise/final-token/full-parity
claim. Compare complete B1/G1 streams like for like, with actual token counts
and EOS/trajectory caveats. B0 is not an equal-length timing window. Keep
startup, CPU setup, transfer/cache/READY, engine wall, prompt/decode/request/wait,
worker service and helper staging/wait times separate. No speedup gate.

## Complete Hermes continuation

Continue the authorized experiment from the immutable commit containing this
amendment. Reuse Hermes's existing approved ROLE_CONTROLLER and transport
bindings without relocating orchestration. Keep the existing private ROLE_ENGINE_A
(inferswarm03, two local GPUs), ROLE_GPU_B (inferswarm04) and ROLE_CPU_C
(inferswarm05) bindings, approved interpreter and trusted namespace receipts.
Existing authorized controller SSH orchestration is permitted. Run no local
tests, builds or model workloads on either user desktop. The focused fixtures
execute on inferswarm05; product workloads execute only on inferswarm03/04/05.
Do not install dependencies or change permissions,
security/network settings, services, cgroups or swap. Preserve other work.

1. Fetch this amendment and both refreshed SHA256 manifests through existing
   authorized GitHub access, at the exact publication commit. Verify the author
   receipt hashes/sizes. Retrieve changed `resource_telemetry.py`,
   `supervisor-integration-v1/agent.py`, `commands.py`, `test_procfs_task_race.py`
   `runtime.py` and README, preserving planning-relative FILES_ROOT paths. Verify retained
   small dependencies against the manifests or fetch only missing listed files;
   do not mix revisions, clone the repository, generate a new supervisor or
   reopen accepted source/binary/plan provenance. Reuse existing validated
   binaries, profile, manifests, remote plan and private bindings.

2. Once on ROLE_CPU_C, invoke exactly once with the approved interpreter:

   ```sh
   "$PYTHON" -B -I "$FILES_ROOT/supervisor-integration-v1/test_procfs_task_race.py"
   ```

   The direct entry caps itself at 128 MiB address space, 5 CPU seconds, 10 wall
   seconds and 1 MiB regular output. Bound captured stdout/stderr to 1 MiB each
   with existing tooling. It mocks proc reads, processes and ownership; no live
   proc/namespace/cgroup/GPU admission or workload. Record hashes, exact argv,
   actual start/end, rc and bounded output. Require exit 0, zero failures/errors/
   skips and exactly `PROCFS_TASK_RACE_FIXTURES_PASS_V1 tests=9 live_signals=0 workloads=0`.
   Do not rerun 69+3, warning nine, sizing or listener campaigns. On focused
   failure preserve evidence and stop. On PASS **automatically continue the
   actual B0/B1/G0/G1 experiment below**, without another approval or prompt.

3. Confirm earlier owned work remains gone using existing scoped receipts;
   preserve unrelated services and failed evidence. Use new private run/output
   roots, new cold remote cache for G0 and that same cache for warm G1. Engine
   SETUP loads the new CPU arena; no standalone preload. One CPU process/arena
   persists through the four sessions. New engine per phase, new remote GPU
   process per G phase; stop/reap each before the next. Keep 11-second gaps after
   B0/G0, serial order and full original phase launch envelopes. Engine caps
   1200/600/1800/900 s, GPU G0/G1 1900/1000 s, CPU admission+7200 s, CPU setup
   600000 ms, GPU setup 1200000 ms and idle 1800000 ms remain unchanged.

4. Cumulative active execution/cleanup charge is **1667.705 seconds**, leaving
   **3732.295 of 5400 seconds**. All-stopped retrieval, coding, focused fixtures
   and handoff are excluded. At resume readiness, before any new owned transport
   or guardian startup, capture controller-local monotonic T and wall correlation.
   Use existing `Controller(..., start=T-1667.705)` with the approved all-owner
   finally cleanup wrapper. Deadline is T+3732.295; existing 45-second cleanup
   reserve makes launch/TERM cutoff T+3687.295. Do not call unadjusted
   `launch(config)`/`coordinated(...)`, reset clocks, pause active accounting or
   compare raw cross-host clocks. Include all new owned startup and active
   teardown after T. Preserve running caps and refuse a phase whose full required
   envelope cannot fit; report it NOT RUN rather than extending the allowance.

5. Execute the phase-specific byte streams above through the corrected agent.
   Preserve every resource/ownership/cleanup guard: strict local prelaunch and
   initial pre-exec admission 1152/512 MiB; warning-only valid local runtime
   reserve crossings, required telemetry/identity, host/cgroup headroom/events,
   fixed-admission SwapFree decline >64 MiB, RSS/anonymous ceilings, pidfds,
   leases/watchdogs and deadlines. Remote GPU reserve 4096 MiB, fixed-plan fit,
   cache 18432 MiB, cold disk 22.25 GiB/running disk 4 GiB remain strict. No whole
   guard bypass, hidden CPU retry, auto-sizing or failure injection. A genuine
   product/CUDA allocation failure stops normally; never disguise it as success.

6. Apply the complete-output/numerical gates above. Reconcile disjoint/exhaustive
   ownership and positive actual execution on both local GPUs, remote GPU and
   CPU. G phases execute only CPU-uncovered 18080 while full 21369 stays resident
   in one unchanged arena/bitmap/partition. Require warm reuse 111597465600 bytes,
   one CPU allocation, zero new CPU load/hash/allocation/expiry/unload or network
   weights between sessions. G0 assigned/cached/logical expert reads/network/
   newly persisted bytes each 17176473600; G1 reads/network/newly persisted zero,
   reuse 17176473600. Keep physical I/O/transport/staging/RSS/GPU allocation
   distinct; reconcile route/client/worker/helper entries and remote final
   counters, classifying normal EOF separately. No whole-model mirror or
   unassigned transfers. Log local per-phase VRAM minima/crossings/warnings.
   Missing evidence is incomplete, never inferred PASS.

7. Correct in-scope launcher/config/evidence expectation mismatches without
   unnecessary approval requests, within these exact settings, bytes and caps.
   Record any correction. This is not permission to patch product code, weaken
   guards, restart a failed focused test or add a model/inference attempt. Once
   inference starts, use only the amended schedule; any required failure stops
   dependent phases. Preserve new detailed procfs refusal context privately.

8. Release owned engines/GPU work, enforce original baseline return <=30 s and
   bounded TERM/grace/KILL/actual parent/adopted reaping. Close evidence within
   the remaining active allowance, <=256 MiB per role/controller, no weights.
   Failure stops CPU; otherwise default-stop unless prior explicit finite
   retention applies (earliest success+30 min, actual idle expiry, original
   admission+2 h). No broad cache deletion or unrelated-service interruption.
   Return phase results, actual/cumulative time and request/token charge, finite
   outputs/numerical comparisons, CPU reuse, cold/warm transfer, ownership/
   guards, cleanup and evidence hashes privately. Preserve the failed B0 report.
   Stop at report: no product rebuild, deployment, merge or permanent service.
