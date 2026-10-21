# G1-EXIT-RECONCILIATION-V1-20261021

This amendment supersedes GPU-COMPLETION-V1's schedule, focused entry, baseline
contract and new-window accounting. Run **G1 only**, after the focused check
below passes. Never rerun B0, B1 or G0. Preserve their actual evidence and the
complete cache. Product, placement, numerical thresholds and resource guards
remain unchanged. Sources and fixtures were statically reviewed, **NOT RUN**
by this author; no Python, scripts, syntax checks, tests, builds or fleet work
were executed on either user desktop.

## Supplied result and narrow correction

The supplied previous eight completion fixtures passed. The initial agent launch
then failed with UnboundLocalError: RETAIN's local `completion = Stamp(...)`
shadowed the imported module throughout `serve`. This revision publishes the
equivalent three-line `completion_stamp` rename that Hermes applied privately.
Original published agent SHA-256:
`72b80388169f588a75f8a4b30f737b212f82f7e97d1c68606c8c45218b2c59ee`.
The equivalent edited file here hashes to
`3629053aa80a8973ed6c3d095a89e5465c795db5557966202e9bd0540aadaabb`,
consistent with the supplied runtime pin. The private patch was not inspected.
Keep the failed-launch evidence and distinguish original from executed source.

G0 then reportedly reached READY, REUSED 0, DONE 64 for A, actual engine rc 0,
finite 993280-byte logits and 3072 complete routing records. Its monitor failed
on missing `ns/cgroup` while its pidfd still appeared alive; the controller
received required-peer-failed before normal completion. **G0 product completed;
G0 supervisor acceptance FAILED.** Its files are valid comparison inputs under
normal identity/config checks; this amendment cannot manufacture a past PASS.
G1 did not run. Supplied G0 archive (35 members):
`strata-gpucompletion-490314aa-20261021T214323Z-evidence.tar.gz`, SHA-256
`511471cd5cbdbee193426be21dd8ffdb6c89755d822294b6594633ef9040a552`.
Retained B0/B1 baseline archive SHA-256:
`35a57b3a4c84aec93ece3c1b255b4704f58d140abf9f07dc779e8a83a723f41d`.
These are supplied observations, not newly verified private evidence.

Static source inspection confirms namespace reads previously refused ENOENT
unless a simultaneous pidfd check already reported exit. That ordering can
misclassify proc teardown. Shared namespace/stat/exe/required-process reads now
reconcile only ENOENT/ESRCH against held proc/pidfd handles: check readable
PID/start identity, wait at most 100 ms on that pidfd, then check again. Pidfd
death yields ProcessExited; unresolved absence yields an identity/path-bearing
ExitPending, never an empty observation or coverage grant. EACCES, malformed
stat, namespace/executable mismatch and changed identity still fail. Missing
monitor-own namespace or external cgroup entries are not excused.

For the root only, the monitor forwards EXIT_PENDING without claiming exit or
normal completion. The true-parent guardian uses its actual waitpid status,
with at most another 100 ms clipped to its deadline, to return the real rc.
Still-unconfirmed exit fails. Earlier guard/refusal packets remain fatal.
Watchdog adoption and cleanup use actual parent wait status for known children
and pidfd-confirmed death for other held descendants. They never signal a
replacement numeric PID, infer death from a missing namespace or zombie text,
or fabricate rc 0. Cleanup retains partial discovery and existing finite caps.
Periodic discovery still provides identity-safe tracking and bounded cleanup,
not an atomic procfs snapshot or a guarantee to observe every fleeting child.
The existing two-pass task/child churn policy is unchanged.

## G1 window, schedule and retained references

Historical time is **5744.2 seconds = 4889.5 + 854.7**. Preserve that ledger and
report actual new-window time and their sum separately, including any additional
documented owned cleanup in the retained receipt. Current usage is **9 started
requests / 576 charged tokens**. Authorize one new **1200-second / 20-minute**
G1 window, measured from T before any owned peer/forward startup through all
owned teardown. All-stopped retrieval, coding, fixture and baseline preparation
are excluded. No running cap/lease extension, counter reset or extra attempt.

Reported CPU cold load 428.9 seconds + the entire prior resumed GPU setup 208.4
seconds as a conservative planning allowance + two 25.09-second requests +
45-second cleanup = **732.48 seconds**, leaving about 467.5 seconds for startup
and variance. Warm-cache validation is not promised to take that exact time.
The estimate supports this bounded window; it does not promise success.
Fresh engine/GPU maxima are 1200 seconds, clipped by the remaining window;
existing setup maxima remain 1800000 ms, CPU local setup 600000 ms and absolute
CPU maximum admission+7200 seconds. Fresh CPU idle maximum is 1200000 ms.
The tighter overall deadline, **45-second cleanup reserve**, five-second
challenge transit and 15-second owned cleanup remain real. Fresh leases only
shorten caps. No maximum-envelope admission or overtime.

Start a fresh CPU service and one fresh engine/GPU worker against the same
complete cache. CPU was released: require a new cold load, not prior-G0 CPU
warm reuse or end-to-end CPU-warm startup. Run exactly:

```text
G1: GEN 64 9419
    GEN 64 14773,6353
    QUIT
```

This consumes the remaining **2 requests / 128 maximum tokens**, ending at
**11 requests / 704 charged tokens**. Charge every started interrupted request
its full 64-token allowance. No use of unused tokens for another attempt.
Require READY, two ordered DONEs, engine rc 0, `.first-logits.0/.1`, complete
output/routing and supervisor cleanup. Actual DONE counts must be <=64, with
prompt lengths 1 then 2, zero reused prompt tokens and no ERR/nonfinite/partial
output, missing/extra request, logit-write error or nonzero exit. Compare actual
tokens/EOS trajectories, not a fabricated 64-token outcome on failure.

Bind actual retained B0/B1 and G0 stdout/logits and timings from the canonical
reports, checking their accepted model/profile/placement, UUID order, request
bytes, flags and settings. Reuse accepted provenance; no new binary/model/plan
campaign. Controller private `baseline` receipt extends the previous contract:

```text
archive_sha256: 35a57b3a4c84aec93ece3c1b255b4704f58d140abf9f07dc779e8a83a723f41d
g0_archive_sha256: 511471cd5cbdbee193426be21dd8ffdb6c89755d822294b6594633ef9040a552
implementation: bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5
plan_sha256: a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578
B0: {rc: 0, done_tokens: [64], output: ARTIFACT, logits: [ARTIFACT]}
B1: {rc: 0, done_tokens: [64,64], output: ARTIFACT, logits: [ARTIFACT,ARTIFACT]}
G0: {rc: 0, done_tokens: [64], product_complete: true, supervisor_status: FAILED,
     output: ARTIFACT, logits: [ARTIFACT]}
ARTIFACT: {path: actual existing absolute private path, sha256: its retained SHA-256}
```

Bind all seven distinct small files from trusted retained evidence. Existing
bounded hash/stability checks read <=8 MiB each. Metadata comes from the trusted
reports; the helper does not itself authenticate metadata, parse DONEs or replace
the existing numerical/output evidence workflow. Missing/mismatched inputs stop
comparison; do not generate replacements.

## Acceptance and limits

The supplied B0/G0 A centered relative L1 **.0352 exceeds .03**; KL .00194 nats,
raw maxabs .372, RMSE .0710, top1 id 11. Preserve this real informational warning;
do not call it a PASS or relax thresholds. Formally compare retained B1 A/B with
new G1 A/B, both centered relative L1 <=.03 and KL(softmax(B1)||softmax(G1))
<=.03 nats at temperature 1. Center each vector by its mean; denominator
max(sum(abs(centered B1)),1e-12); use stable double log-softmax. Retained G0/new
G1 matching-A drift remains maxabs <=1e-5*max(1,maxabs(reference)). Logits must
be exactly 4*vocab finite float32 bytes using trusted vocab. Report raw/centered
maxabs, RMSE, top1/margin and token agreement. If G1 violates these numerical
checks, report **NUMERICAL_FAIL**, finish cleanup and stop; no infrastructure
iteration, retry or threshold tuning. Repeated-A coverage remains absent.

Reported G0 cache accounting was R=16011878400 reused + M=1164595200 newly
transferred = W=17176473600 assigned bytes (223 missing experts), reads/network/
new persistence each M. Cache is now reported complete, 3290 files including
metadata. Reuse that directory through existing product validation: it checks
assigned keys, sizes and digests and reuses complete valid entries; it does not
resume inside an incomplete blob. Never assume an aggregate proves validity,
delete cache files, mirror the model or run a separate cache hash sweep.
G1 requires **R=W, M=source weight reads=network weight payload=new persistence=0**,
assigned/cached/GPU weight bytes W and no unassigned payload. After parsing actual
counters call `gpu_completion.cache_accounting("G1", reused=R, transferred=M,
persisted=..., source_read=..., cached=..., assigned=...)`. Product validation
and frozen plan membership remain mandatory; the helper checks equations only.
Hashing existing cache entries is distinct from new source-weight reads.
Unexpected misses/transfer fail this warm-cache acceptance; no hidden G0 retry.

Require fresh CPU full 21369-expert / 111597465600-byte arena, one allocation,
zero CPU network weights, same arena retained for both G1 requests. Do not demand
zero fresh CPU load/hash or claim reuse of G0's released arena. G phases execute
CPU-uncovered 18080; reconcile disjoint/exhaustive ownership and positive work
on local GPUs, remote GPU and CPU. Separate CPU cold setup, cache validation,
GPU loading/READY and request/worker/helper time. Compare B1/G1 A/B request
execution like-for-like, with token/EOS/trajectory and host-load caveats. G0 is
resumed-cache setup; G1 is GPU-cache warm with CPU cold. Neither is an equal
full-phase timing window to the retained B streams.

## Complete Hermes continuation

1. Reuse the existing approved ROLE_CONTROLLER and transport bindings; do not
   relocate orchestration or change routes/security/network settings. Authorized
   controller SSH orchestration is permitted. No Python/scripts/tests/builds/
   models/fleet operations on either user desktop. Fetch this amendment and all
   listed source dependencies at the exact immutable publication commit, verify
   author hash/byte pins and refreshed integration/runner manifests, preserving
   planning-relative FILES_ROOT paths. No repository clone, installs, rebuild,
   sizing/listener/warning/periodic-observation or old completion campaign.

2. Run exactly once on inferswarm05:

   ```sh
   "$PYTHON" -B -I "$FILES_ROOT/supervisor-integration-v1/test_g1_exit.py"
   ```

   Entry caps are 128 MiB address space, 5 CPU seconds, 10 wall seconds and 1 MiB
   regular output; bound stdout/stderr to 1 MiB each. Record hashes, argv, actual
   start/end, rc and output. Require exit 0, zero failures/errors/skips and exactly
   `G1_EXIT_FIXTURES_PASS_V1 tests=12 live_signals=0 workloads=0`.
   Cases cover shared namespace teardown, persistent/identity/read faults, actual
   guardian rc including nonzero, bounded unresolved exit, preceding refusal,
   cleanup/adoption, the shadowing regression, G1-only schedule/reference/cache
   acceptance and window/accounting. NOT RUN by author. Reuse previous passes
   for unchanged paths; PASS **automatically continues actual G1**, no prompt.
   Fixture failure stops and preserves evidence.

3. While stopped, bind the private baseline above to actual retained files.
   Preserve previous reports, failed-launch/monitor evidence and complete CACHE_B.
   Set `gpu_completion: true` in controller and all three agent configs,
   controller `retain: false`, agents `retention_approved: false`; attach the
   `baseline` receipt to controller config. Use new private output/run roots,
   existing approved interpreter/binary/config bindings and the same exclusively
   owned real cache directory. No copied private agent patch is needed.

4. Use updated `launch(config)`, capturing T before owned forwards/peers. It
   selects G1 only and the 1200-second window. Alternatively use the existing
   all-owner wrapper with `Controller(peers, gpu_completion=True,
   baseline=receipt, retain=False, start=T)`. Do not call default `coordinated`,
   subtract historical time from T, extend running caps or reset history. Count
   cold CPU load, validation/setup, both requests and all owned teardown. Report
   historical 5744.2, actual new elapsed and their sum, plus 9/576 before and
   actual final charge <=11/704. Preserve any documented additional cleanup.

5. Keep product `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`, main
   `34a59d83a51bdb3f0299d65306740bd357818715`, PR #2 draft/unmerged. Placement
   primary 960, helper 2247, CPU full 21369, uncovered 18080, remote 3289;
   CPU budget 106428 MiB, payload 111597465600 bytes. Plan remains 92156 bytes
   with SHA above. No replanning, weight hashing/transfer outside product setup
   or product change. Preserve strict local admission/pre-exec 1152/512 MiB,
   warning-only valid local runtime crossings, real host/cgroup memory/events,
   fixed-admission SwapFree decline >64 MiB, RSS/anonymous ceilings, required
   telemetry/identity/pidfd checks, leases/watchdogs, remote 4096 MiB reserve,
   18432 MiB cache budget, fixed-plan fit and G1 warm disk/runtime 4 GiB guards.

6. Run only G1's two GEN64 requests and all acceptance above. Correct ordinary
   in-scope launcher/config/expectation errors without new approval, preserving
   this schedule and bounds. Do not hide product/guard/numerical failures, repeat
   inference or alter thresholds. Stop CPU and all owned engines/GPU services/
   forwards inside the window, baseline return <=30 seconds, bounded TERM/grace/
   KILL and actual parent/adopted reaping. Preserve cache and unrelated services.
   Evidence <=256 MiB per role/controller, no model blobs. Return exact focused
   and G1 outcomes, G0's failed supervisor status, references/comparisons including
   the warning, cache/CPU accounting, separate setup/request times, total usage,
   terminal diagnostics, cleanup and evidence hashes privately. Stop at report;
   no merge, deployment, permanent service or additional attempt.
