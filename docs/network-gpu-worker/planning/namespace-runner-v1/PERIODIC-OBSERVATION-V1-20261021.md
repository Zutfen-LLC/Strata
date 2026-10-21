# PERIODIC-OBSERVATION-V1-20261021

Active continuation from `33b196addfbb200773a3bae1221ebf1b86b96de3`.
This supersedes task/child convergence rules, focused gates, request/DONE/logit
counts, repeated-A acceptance, full-maximum-envelope admission and time/usage
ledgers. Follow only this document's execution steps. Keep
[GPU-RESERVE-WARNING-V1](GPU-RESERVE-WARNING-V1.md), frozen
[primary sizing](PRIMARY-STARTUP-SIZING-CORRECTION.md) and other protections and
acceptance in the [original budget](../EXECUTION-BUDGET-bf51f09e.md).

## Supplied outcome

Hermes reports twelve churn fixtures PASS. B0 completed A with DONE 64, engine
rc 0, request wall 22.5 seconds and 3072 complete routing records, no partial
tail. Cold CPU setup took 427.8 seconds. B1 reconnected in 0.1 seconds, reached
READY and emitted 32 of 64 tokens without DONE, then stopped on four-scan
task-set nonconvergence. Every scan had complete identical before/after sets,
but sets changed across scans. G0/G1 did not run. Creator provenance is unknown.

Reported evidence: one 111597465600-byte / 21369-expert CPU arena, one allocation,
one warm hit reusing all bytes with no new load/hash/network. B0.0/B1.0 first
logits were bitwise identical, reported numerical differences zero; B1's first
32 tokens matched B0. This is one finished baseline request and partial B1,
not a complete comparison. No CUDA allocation failure was reported. Engine peak
RSS 7554 MiB; GPU minima 1004/432 MiB, warning-only crossings; CPU RSS 106442.5
MiB, VmSwap zero. All owned products/transports were reaped, CPU released, GPUs
at baseline, no remote cache. Preserve private evidence; supplied archive SHA-256
`14782d8f2e4443089ff16c2d72dded0ef56d923319b86bcb3e9bb4d641ce8b58`.
These are supplied observations, not independently inspected private evidence.

## Periodic observation and clipped upper bounds

Shared `OwnedProcess.children()` now always makes two finite observation passes.
Each enumerates tasks, reads their children and enumerates again, under existing
task/read/process caps and pinned parent checks. The second pass rereads tasks
that may receive reparented children. All observed children are retained for
namespace/parent/start/executable/pidfd reconciliation. Neither thread sets nor
child maps must converge within or between passes. Churn alone never refuses.

A missing task's children ENOENT is recoverable only after that task is absent
in the following enumeration and the owned parent still verifies. A still-present
task missing its required entry, EACCES, malformed data, required leader/task
directory absence, identity mismatch or unresolved ownership still fails.
Confirmed dead descendants and reused numeric IDs retain pidfd-safe handling;
gone candidates grant no identity or signal. The shared method covers normal
monitor discovery, watchdog adoption and cleanup, with no remaining set-equality
gate. There is no product change, larger scan limit or new monitoring framework.

The guarantee is bounded **periodic best-effort discovery with identity-safe
tracking and cleanup**, not an atomic global procfs snapshot. A process born and
exited wholly between observations can be unseen. A task created after a final
read can be discovered by later sampling/adoption. This finite observation gap
does not excuse a genuine unreadable required entry or fabricated ownership.
Every observed live child still needs ownership pinning. No global quiescence
is required. Management queries are monitor children; monitor and workload are
watchdog siblings. Those queries enter cleanup discovery, not workload-rooted
sampling. No supplied evidence identifies the creator of the reported churn;
do not claim the model forks. The product's CUDA-stream `fork` helper is not an
OS fork.

The controller no longer requires the full GPU maximum envelope before launch.
`bounded_job_deadline()` takes the lesser of an existing process deadline and
the remaining local global limit, requiring a positive interval after the
existing 15-second owned cleanup. Controller phase waits use the clipped cap;
agents apply the same bound to CPU (also admission+7200), engine and GPU jobs.
Fresh leases can only shorten job/gate/readiness deadlines. Late phase refusal
also closes the prepared input descriptor. No existing cap is extended.

Upper bounds remain engine 1200/600/1800/900 seconds, GPU 1900/1000, CPU 7200;
CPU/GPU setup 600000/1200000 ms, idle 1800000 ms. The overall 5400-second limit
and 45-second global cleanup reserve remain real. Remaining durations already
exclude that reserve. The existing five-second nonce/challenge transit deduction
and 15-second owned cleanup remain: a phase needs more than 20 seconds before
the global reserve, yielding some positive execution interval. This minimum
comes from the current conservative lease/cleanup protocol, not a full maximum
timeout. No timer framework or raw cross-host clock comparison is added.

Twelve focused mocked methods cover continuous thread/child churn, the exact
cross-scan pattern, reparenting amid churn, required read/identity failures, safe
PID reuse, terminal descendants, adoption, bounded termination, clipped waits/
lifetimes/leases and exact request bytes. Selected protection methods are reused,
not historical campaign entrypoints. Tests are authored **NOT RUN**. Static
review checked these concrete paths, bounds and descriptor ownership. No tests,
builds, models or fleet operations ran in this coding task.

## Reduced protocol and acceptance

B0 and G0 each receive A only:

```text
GEN 64 9419
QUIT
```

B1 and G1 each receive matching A then B:

```text
GEN 64 9419
GEN 64 14773,6353
QUIT
```

Require READY, one normal finite product DONE and `.first-logits.0` per B0/G0;
two ordered normal finite DONEs and `.first-logits.0/.1` per B1/G1. Engine rc 0
and complete supervisor cleanup are separate requirements. Each request generates
at most 64 tokens; reconcile emitted records with actual DONE counts, prompt
lengths 1 or 1/2, zero reused tokens and full prompt reads. ERR, cancelled/
nonfinite/partial output or routing, missing/extra requests, missing DONE/files
or nonzero exit fail acceptance. EOS never authorizes an extension.

Each logit file is exactly 4*vocab finite float32 bytes using trusted vocab;
missing/short/write-error files fail even with rc 0. Compare **both** matching
B1/G1 pairs: centered relative L1 <=.03 and KL(softmax(B1)||softmax(G1)) <=.03
nats at temperature 1. Center each by its mean; denominator
max(sum(abs(centered B1)),1e-12); stable double log-softmax. Preserve cold/warm
matching-A drift B0.0/B1.0 and G0.0/G1.0:
maxabs <=1e-5*max(1,maxabs(reference)). Report raw/centered maxabs, RMSE,
top1/margin and token agreement; no bitwise/final-token/full-parity claim.
Repeated-A acceptance is explicitly **dropped**, reducing repeatability coverage.
There is no `.2` request/file; absent checks are not PASS.

B1/G1 remain like-for-like two-prompt timings, with actual tokens and EOS/
trajectory caveats. B0/G0 are startup/cache smoke, not equal-length timing windows.
Separate startup/setup, transfer/cache/READY, engine wall, prompt/decode,
request/wait, worker service and helper staging/wait. No speedup gate.

## One complete Hermes continuation

Reuse the existing approved ROLE_CONTROLLER and transport bindings without
relocating orchestration. Authorized controller SSH orchestration is permitted.
Focused tests run on inferswarm05; products only on inferswarm03/04/05. No local
test/build/model workload on either user desktop. Keep existing interpreter,
namespace receipts and accepted private source/binary/plan/input bindings.

1. Fetch this amendment and refreshed integration/runner manifests at the exact
   publication commit; verify author receipt SHA-256/byte pins. Retrieve changed
   shared `resource_telemetry.py`, integration `agent.py`, `commands.py`,
   `controller.py`, `policy.py`, `test_periodic_observation.py`, retained test
   helper revisions and README, preserving planning-relative FILES_ROOT paths.
   Verify retained small dependencies against manifests or fetch missing listed
   files. The runner manifest retains 32 entries by replacing the superseded
   amendment and omitting its obsolete HERMES handoff row; all files remain in
   Git. No old runner/campaign, repository clone, dependencies, renewed provenance,
   planner, weight hashing/preparation transfer, product rebuild or deployment.

2. Invoke exactly once on inferswarm05:

   ```sh
   "$PYTHON" -B -I "$FILES_ROOT/supervisor-integration-v1/test_periodic_observation.py"
   ```

   Entry bounds: 128 MiB address space, 5 CPU seconds, 10 wall seconds, 1 MiB
   regular output; bound stdout/stderr to 1 MiB each using existing tooling.
   Record hashes, argv, actual start/end, rc and output. Require exit 0, zero
   failures/errors/skips and exactly:
   `PERIODIC_OBSERVATION_FIXTURES_PASS_V1 tests=12 live_signals=0 workloads=0`.
   No old 69+3, warning, procfs-nine/twelve, sizing or listener campaigns. Failure
   stops and preserves evidence; PASS **automatically continues the actual trial**
   below without another prompt or approval.

3. Preserve prior evidence and unrelated services; confirm previous owned work
   remains gone using scoped receipts. Product remains
   `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`, main
   `34a59d83a51bdb3f0299d65306740bd357818715`, PR #2 draft/unmerged.
   Primary 960, helper 2247, CPU full 21369, CPU uncovered 18080, remote 3289;
   CPU budget 106428 MiB, one 111597465600-byte full arena. Existing remote plan
   is 92156 bytes, SHA-256
   `a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578`.
   No placement/binary/plan change. New output roots/cold G0 cache, same cache
   for G1; one CPU arena across sessions, engine SETUP loads it, no standalone
   preload. New engine each phase, new GPU process per G phase, serial transitions
   and 11-second B0/G0 gaps remain.

4. Reported active execution/teardown is **2995.1 seconds**, leaving **2404.9 of
   5400**. Reconcile the retained receipt before resume: add any additional
   documented owned cleanup not included, without double counting. Preserve
   unresolved measurement uncertainty; do not silently call an unmeasured delay
   zero. All-stopped coding/retrieval/fixtures/handoff are excluded. At readiness
   T before owned startup, use `Controller(..., start=T-charged_seconds)` with
   all-owner finally cleanup. With no additional documented charge, deadline
   T+2404.9, global ordinary-work cutoff T+2359.9. Preserve the real 45-second
   reserve and five-second transit/15-second owned cleanup. Use clipped effective
   phase bounds; **do not apply old full-maximum-envelope admission**. No old
   wall cutoff, cap extension, timer framework, paused clock or counter reset.

5. Usage is **4 started requests / 256 charged tokens**. Execute only six new
   requests above, at most 384 tokens: cumulative **10 requests / 640 charged
   maximum**, under the current **11 requests / 704 tokens ceiling**. Interrupted
   requests are charged full 64; observed tokens are not final totals. Unused
   allowance grants no retry. Apply the exact DONE/logit/numerical gates above
   and state the reduced repeatability coverage honestly.

6. Keep strict local admission/pre-exec floors 1152/512 MiB, warning-only valid
   local runtime crossings; required telemetry, host/cgroup memory/events,
   fixed-admission SwapFree decline >64 MiB, RSS/anonymous ceilings, identity/
   ownership/pidfds, watchdog/lease/sample and cleanup protections remain.
   Remote GPU reserve 4096 MiB, fixed-plan fit, cache 18432 MiB, cold disk
   22.25 GiB/runtime 4 GiB stay strict. Real CUDA/product failure stops normally;
   no hidden CPU retry, auto-sizing, guard bypass or failure injection. Do not
   install dependencies or change security/network/settings, services or swap.

7. Reconcile disjoint/exhaustive ownership and positive work on both local GPUs,
   network GPU and CPU. G phases execute only CPU-uncovered 18080; full 21369
   stays in the same arena/bitmap/partition. One allocation, warm reuse
   111597465600 bytes, zero new CPU load/hash/allocation/expiry/unload/network
   weights between sessions. G0 assigned/cached/logical expert reads/network/new
   persisted each 17176473600 bytes; G1 reads/network/new persisted zero, reuse
   17176473600. Separate physical I/O/transport/staging/RSS/GPU allocation;
   reconcile complete routes, client/worker/helper and final remote counters,
   classifying normal EOF separately. Missing evidence remains incomplete.

8. Fix ordinary in-scope launcher/expectation errors without needless approval,
   within exact bytes/settings/effective bounds. No product change to satisfy
   a harness, rerun of a failed focused test or extra inference attempt. Required
   failure stops dependent phases. Release owned work, GPU baseline return <=30
   seconds, bounded TERM/grace/KILL and actual parent/adopted reaping. Close
   evidence within the remaining allowance, <=256 MiB per role/controller, no
   weights. Failure stops CPU; default-stop otherwise unless prior explicit
   finite retention applies (earliest success+30 min, idle expiry, admission+2 h).
   No broad cache deletion or unrelated-service interruption. Return phase/
   output/numerical/transfer/reuse/ownership results, time/usage reconciliation
   and uncertainty, diagnostics, cleanup and evidence hashes privately. Stop at
   report; no build, merge, deployment or permanent service.
