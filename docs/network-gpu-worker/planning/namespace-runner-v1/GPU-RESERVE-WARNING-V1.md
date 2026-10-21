# GPU-RESERVE-WARNING-V1

Current execution continuation: [GPU-COMPLETION-V1-20261021](GPU-COMPLETION-V1-20261021.md).
It supersedes the focused check, all-phases A/B/A requirement and time/usage
ledger below after the reported interrupted inference runs. The local
runtime warning policy in this document is unchanged.

This authorized experiment policy amendment supersedes earlier instructions to
terminate on the engine role's local runtime reserve crossings. Placement and
product binaries do not change. Historical artifacts, receipts and budget bytes
remain unchanged. This file is the active continuation policy.

## Reported evidence and change

The user reports all five e7803083 sizing fixtures passed. B0 loaded exactly
21,369 CPU experts / 111,597,465,600 bytes in one allocation, zero network weights,
setup 428.496 seconds. CPU peak RSS was 106,441.254 MiB, owned VmSwap zero, host
SwapFree decline 5.75 MiB. Later startup was stopped solely by gpu-reserve-floor:
primary 1,114 MiB versus 1,152; helper 432 MiB versus 512. No reported CUDA OOM,
engine READY, inference or remote GPU work. All owned work was reaped, the CPU
arena released, no remote cache created and GPU baseline restored within 0.846
seconds. The reported evidence archive SHA-256 is
`2d6913451cc2fd0ab0dd3f75def935f152d5250afd57d63a3aef50f82c368017`.
These are supplied observations, not independently inspected logs or new PASS
claims by the author. No Python, tests, scripts, builds or runtime were executed
by the author or static reviewers.

`OuterGuards.check(admission=True)` still requires at least 1,152 MiB primary and
512 MiB helper before each engine launch and at the first monitor coverage sample
before exec-gate release. The product's own startup admission and allocation
checks are also unchanged. After this gate, the host-local monitor's
two-second samples treat valid local free-memory observations below those values
as diagnostics only, including later CUDA/serve/prompt initialization. Even zero
reported free bytes adds no replacement supervisor stop threshold. Required missing,
malformed, negative or misidentified telemetry still refuses coverage and stops
owned work; unavailable is never zero.

`outer.py` returns per-phase local diagnostics. `runtime.py` journals them as
`LOCAL_GPU_RESERVE_DIAGNOSTIC_V1`, with policy identifier, selected primary/helper
order, free bytes, minimum sampled free bytes, reserve bytes, below-reserve flags
and crossing counts. A crossing is entry from at/above reserve to strictly below;
continued low samples do not increment it, recovery then re-entry does. The first
low runtime sample counts as entry from successful pre-exec coverage. Minima cover evaluated
samples, not continuous device peaks. Counters reset only at a new strict phase
admission. `GPU_RESERVE_WARNING_V1` is emitted once per GPU's first crossing per
phase: at most two warning records per phase, eight over four engine phases.
Later diagnostic samples retain every entry count and minimum without console
spam. Journals keep their existing 64 MiB/record bounds; log failure still refuses
coverage. Existing sample deadlines and cadence are unchanged.

All local reserve enforcement paths were inspected: the agent's admission uses
`admission=True`; initial monitor coverage repeats the strict local check before
exec release, then the phase monitor uses runtime checks; cleanup `returned()`
still independently requires restoration to the original admitted baseline within
its existing bound. A warning does not waive cleanup or next-phase admission.
The remote GPU's 4,096 MiB reserve, fixed-plan fit, disk/cache protection and
required telemetry remain strict at admission and runtime. Host RAM/RSS/swap/
cgroup guards, identities, pidfds, listener coverage, watchdogs and deadlines are
unchanged. No whole guard or exception path is bypassed.

Actual CUDA allocation failure remains possible and must fail/clean up through
the existing product exit and supervisor propagation. No new CUDA error detector
has been added. A nonzero product exit or existing failure must not be rewritten
as success; this policy changes only local free-memory reserve crossing handling.

The nine new mocked cases cover warning continuation/minima/crossings including
zero, strict admission and pre-exec boundaries, bounded warning records with continued monitor
coverage, malformed telemetry, identity/host-guard refusal, strict remote guards,
cleanup baseline and existing product/phase failure propagation. Management and
ownership failures are injected, not live probes or hardware error detection.
The prior sizing fixture now explicitly exercises admission floors. All new or
modified fixtures are authored NOT RUN here; no unrelated campaign is requested.

## Immutable inputs and unchanged experiment

Implementation: `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`.
Main: `34a59d83a51bdb3f0299d65306740bd357818715`; PR #2 draft and unmerged.
Primary 960 (20/layer), helper 2,247, CPU full 21,369, CPU uncovered 18,080.
CPU payload 111,597,465,600 bytes; memory-budget 106,428 MiB; one fresh arena.
Remote assignment remains exactly 3,289 experts / 17,176,473,600 payload bytes.
The existing 92,156-byte plan SHA-256 remains
`a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578`.
No planner, assignment change, manifest regeneration, payload preparation, model
hash sweep or product rebuild is required or authorized by this coding change.

All unchanged workload, numerical, transfer/ownership, resource and cleanup rules
are in [the primary sizing continuation](PRIMARY-STARTUP-SIZING-CORRECTION.md),
including its references to the original budget. Apply that procedure with ONLY
the runtime local reserve policy, focused check and active-time ledger overridden
here. Do not apply its obsolete strict local runtime floors or old time balance.

## Complete Hermes continuation

Continue the authorized experiment with the immutable commit containing this file.
Use existing private ROLE_ENGINE_A (two local GPUs), ROLE_GPU_B (network GPU),
ROLE_CPU_C (CPU worker) and ROLE_CONTROLLER bindings, authenticated transport,
trusted namespace inputs and live verified bf51f09e binary receipts. Preserve
all failed evidence and other services. No additional preparation approval.

1. Through existing authorized GitHub access, fetch this amendment and refreshed
   integration/runner SHA256 manifests at the exact published commit. Verify
   author-receipt hashes/sizes. Fetch changed `outer.py`, `runtime.py`,
   `test_placement_sizing.py`, new `test_gpu_reserve_warning.py` and integration
   README, preserving planning-relative paths. Verify retained dependencies against
   the refreshed manifests, or retrieve only missing listed small files. Do not
   mix revisions, clone the repo, generate another supervisor or download dependencies.

2. On ROLE_CPU_C invoke exactly once with the approved interpreter and artifact root:

   ```sh
   "$PYTHON" -B -I "$FILES_ROOT/supervisor-integration-v1/test_gpu_reserve_warning.py"
   ```

   The reviewed direct entry sets 128 MiB address space, 5 CPU seconds, 10 wall
   seconds and 1 MiB regular-output cap. Bound captured stdout/stderr to 1 MiB
   each with existing tooling. It uses mocked snapshots/processes only, no live
   GPU/proc/namespace/cgroup admission. Record hashes, exact argv, actual start/end,
   RC and bounded output. Require exit 0, zero failures/errors/skips and exactly:

   `GPU_RESERVE_WARNING_FIXTURES_PASS_V1 tests=9 live_signals=0 workloads=0`

   Do not rerun the sizing, listener or 69+3 campaigns. On focused failure stop
   and preserve evidence; on PASS continue automatically through B0/B1/G0/G1.

3. Confirm earlier owned work remains gone, without touching other services. Reuse
   the exact frozen placement, plan, profile/manifests, binaries and approved private
   supervisor invocation. No resize/replan/build. Engine SETUP loads the new CPU
   arena; no standalone preload. Apply the unchanged four-run procedure linked
   above, exact published requests/settings, resource guards, phase envelopes,
   numerical acceptance, transfer/cache and ownership reconciliation. Local runtime
   crossings alone must warn and continue; admission, telemetry, remote reserve,
   product failures and cleanup remain enforced. Actual CUDA OOM may occur: stop
   and clean up normally, without hidden CPU retry or another placement change.

4. Charge cumulative active execution/cleanup of 1,109.118 seconds. Of 5,400 seconds,
   4,290.882 remain. All-stopped retrieval, focused fixtures, coding and handoff are
   excluded. At resume readiness capture controller-local monotonic T and wall
   correlation; use existing `Controller(..., start=T-1109.118)` with the approved
   all-owner cleanup wrapper. Deadline is T+4290.882; existing controller launch/
   TERM cutoff is T+4245.882. Do not use unadjusted launch(config)/coordinated(...)
   to reset the allowance or revive the withdrawn wall cutoff. Include new owned
   startup and active teardown after T. Preserve actual per-process caps; no clock
   pauses, cross-host raw clock comparisons or process lifetime extension. Refuse
   phases lacking their full remaining envelope and cleanup reserve.

5. Usage remains zero of 12 requests / 768 generated tokens, at most 64 per request.
   Record local diagnostic minima and crossings by phase and GPU, with warning
   markers; distinguish this policy from actual allocation errors or product exits.
   Preserve the original finite-output, first-logit numerical, routing/owner counters,
   selective cold/warm cache and like-for-like timing requirements. Missing evidence
   remains incomplete, not PASS. Do not extend/retry inference or run a cosmetic test.

6. On failure stop dependent phases. Release owned engines/GPU tasks, enforce
   baseline restoration and bounded TERM/grace/KILL/reaping, and close evidence
   inside the remaining active allowance. CPU retention only if already explicitly
   approved under the existing earliest idle/success/absolute caps; failure stops it,
   otherwise default-stop. No broad cache deletion, unrelated-service interruption,
   permanent service, merge or rebuild. Return exact phase/consumption/guard/product
   outcomes, diagnostics, cleanup and evidence hashes privately. Stop at report.
