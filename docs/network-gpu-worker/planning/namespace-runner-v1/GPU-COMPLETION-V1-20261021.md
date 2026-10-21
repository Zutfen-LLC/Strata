# GPU-COMPLETION-V1-20261021

Active GPU-only continuation from `133156849f6c19145c28ebfd9601beeb5d4389fb`.
Successful B0/B1 are retained baseline inputs, not repeated. This replaces the
old shared-window execution steps, fresh-empty G0 cache assumption, fixed
all-new-transfer requirement and old GPU setup maxima. Preserve prior evidence,
[GPU reserve warnings](GPU-RESERVE-WARNING-V1.md), periodic observation,
[placement](PRIMARY-STARTUP-SIZING-CORRECTION.md) and substantive protections.
Follow only this continuation's execution steps. No product merge or deployment.

## Supplied result and source findings

Hermes reports all twelve periodic-observation fixtures PASS. B0 completed DONE
64, rc 0, request wall 20.97 seconds, CPU setup 428.5 seconds. B1 completed both
64-token A/B requests; warm CPU reconnect 0.1 seconds. Matching B0/B1 A tokens
and logits were bitwise identical. G0 stopped before inference when its
1200000 ms GPU SETUP budget expired, with a reported 16011879040 of 17176473600
assigned bytes persisted. Reported relay throughput was approximately 106 Mbps.
G1 did not run. This establishes a setup timeout under that relay path, not CUDA
compute incorrectness or a measured bottleneck on any particular link.

All owned services/forwards stopped, CPU arena released and partial cache-B
preserved. Historical active charge is **4889.5 seconds**, usage **8 requests /
512 charged tokens**. Preserve the canonical prior evidence archive and report;
supplied archive SHA-256:
`35a57b3a4c84aec93ece3c1b255b4704f58d140abf9f07dc779e8a83a723f41d`.
These are supplied observations, not newly inspected private evidence.

Pinned product `bf51f09e` supports resuming **complete validated cache entries**:

- `gpu_assets.cpp:GpuCache::load/prepare` checks exact assigned keys, sizes and
  digests, then creates a missing bitmap. Missing/wrong-size/digest-invalid
  assigned entries are misses. Unsafe entries, unreadable files and disk-budget
  failures remain errors, not misses that bypass a guard.
- `GpuWorkerTier::open` uploads only missing assigned assets. `publish` verifies
  each complete blob, writes/fsyncs a temporary file, renames and fsyncs the
  directory. In-flight partial uploads held in memory are lost at disconnect;
  there is no resume within a blob. Final entries are revalidated on restart.
- Cache inventory counts all retained files, including temporary/unassigned
  regular files, against the unchanged 18432 MiB budget. Conservative reservation
  may refuse invalid replacements if old files consume too much space. Preserve
  that refusal; do not delete files or raise budgets to force progress.

Thus reuse the same retained cache directory and product validation. The reported
persisted aggregate is not proof of the reusable entry set. Do not subtract it
and assume the remainder is the missing bitmap. Do not mirror the full model,
delete the partial cache, run a standalone cache hash sweep or alter transport.

Worker `--setup-ms` accepts up to 86400000 ms; engine
`--remote-gpu-setup-ms` accepts positive values. Both are 1800000 ms in this
completion mode. SETUP/READY response waits use the remaining whole setup budget;
ordinary per-frame/request I/O remains 10000 ms. CPU setup precedes GPU setup.
The existing CPU keepalive preserves its connection but does not renew resident
idle TTL, so a 30-minute CPU TTL could undercut a 30-minute GPU setup. Fresh
completion CPU jobs therefore use a **2700000 ms idle maximum**, with no running
TTL extension. The same guardian window stops them earlier when necessary.

## Window, scope and estimate

Retire the old 5400-second window as historical accounting. Do not zero its
4889.5-second charge or silently carry its unused balance into this new window.
Authorize exactly one new **2700-second / 45-minute GPU-completion window**.
Report historical time, actual new-window time and their sum separately.

At approximately 106 million bits/s, all 17176473600 assigned bytes would take
about 1296.3 seconds of payload transfer. Adding reported CPU load 428.5 seconds,
three times the observed 20.97-second A request as a rough inference allowance,
the 11-second G0 gap and 45-second cleanup reserve totals about **1843.7 seconds**,
leaving about **856.3 seconds** for cache verification, GPU loading, engine/peer
startup and variance. This is an estimate, not measured distributed inference
or a promise of bandwidth. Resume should transfer less if entries validate.
The 45-minute window is reasonable even for the reported-rate full-payload
fallback; it does not promise that every maximum timeout can occur in sequence.

New engine and GPU worker phase maxima are each 2700 seconds, clipped to the
remaining window. These no longer contradict 600 seconds of CPU setup followed
by 1800 seconds of GPU setup. CPU absolute maximum remains admission+7200, also
clipped by the new window. Retain real 45-second global cleanup, five-second
challenge transit and 15-second owned cleanup. Fresh leases only shorten bounds;
no running lease or process cap is extended. No full-maximum-envelope admission.
If the remaining window expires, fail/clean up; do not retry or grant overtime.
Completion mode disables retention and default-stops CPU inside this window.

Only G0 and G1 launch. New CPU worker cold-loads one full arena during G0 and
reuses that same arena for G1. G0 is **resumed-cache setup**, not a fresh empty-cache
cold-fill benchmark. New engine per phase, new GPU worker per G phase, same
retained cache, original serial transitions and 11-second G0 gap.

```text
G0: GEN 64 9419
    QUIT

G1: GEN 64 9419
    GEN 64 14773,6353
    QUIT
```

These are the remaining **3 requests / 192 maximum tokens**, bringing cumulative
maximum to **11 requests / 704 charged tokens**. Charge any started interrupted
request at its full 64-token allowance. No B reruns, extra prompts, retries or
EOS extensions. B0/B1 current phase events must never be fabricated.

## Retained baseline and acceptance

Use the actual successful B0/B1 outputs, logit files and timings from the canonical
prior report. Check their existing identities/hashes and normal frozen settings:
same model/profile/placement, UUID order, manifests, request bytes and inference
flags. Timing-only allowances may differ as documented here. Reuse accepted
receipts; no new source/binary/model provenance campaign. Missing or mismatched
required baseline artifacts block comparison; never manufacture replacement data.

Controller private `baseline` receipt schema:

```text
archive_sha256: the canonical archive SHA above
implementation: bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5
plan_sha256: a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578
B0: {rc: 0, done_tokens: [64], output: ARTIFACT, logits: [ARTIFACT]}
B1: {rc: 0, done_tokens: [64,64], output: ARTIFACT, logits: [ARTIFACT,ARTIFACT]}
ARTIFACT: {path: existing absolute private file path, sha256: its existing SHA-256}
```

Output means the corresponding actual engine stdout; logits are numbered .0,
then .1. Bind all five distinct files from retained evidence and its trusted
manifest, not guessed hashes. `validate_baseline` rechecks only these small files
(<=8 MiB each) with existing hash/stability logic. The archive identifier locates
the trusted prior report; the helper does not authenticate metadata by itself,
parse DONEs or replace the existing numerical/output evidence workflow.

Require G0 READY, one normal finite DONE and `.first-logits.0`; G1 READY, two
ordered normal finite DONEs and `.first-logits.0/.1`. Engine rc 0 and complete
supervisor cleanup are separate. Reconcile actual token counts <=64, prompt
lengths 1 or 1/2, zero reused prompt tokens, full reads and complete routing.
Missing/extra requests, ERR, nonfinite/partial/cancelled output, missing DONEs,
short/logit-write errors or nonzero exits fail. Logits remain exactly 4*vocab
finite float32 bytes using trusted vocab.

Compare retained B1 A/B against new G1 A/B: centered relative L1 <=.03 and
KL(softmax(B1)||softmax(G1)) <=.03 nats at temperature 1 for both pairs. Center
each by its mean, denominator max(sum(abs(centered B1)),1e-12), stable double
log-softmax. Compare G0/G1 matching-A drift:
maxabs <=1e-5*max(1,maxabs(reference)). Preserve the actual completed B0/B1 A
evidence. Report raw/centered maxabs, RMSE, top1/margin and token agreement;
no all-token/bitwise/full-parity claim. Repeated-A coverage remains absent.
Compare like-for-like B1/G1 A/B request execution with actual tokens, EOS/
trajectory and host-load caveats; label different setup/cache states. Separate
CPU setup, validated cache reuse, new transfer, GPU loading/READY and request/
worker/helper times. Do not claim resumed G0 is a cold-fill measurement.

Let W=17176473600, R=product-validated reused assigned bytes and M=new missing/
invalid assigned payload. For successful G0 require **R+M=W**, assigned/cached/GPU
weight bytes W, source logical reads/network weight payload/newly persisted
payload each M, no unassigned transfer. Derive R/M from actual client/worker
counters and frozen plan, not the old aggregate. Whole fixed entries are
5222400 bytes. Physical I/O, wire overhead, temp files and retained-disk bytes
are distinct; verified bytes may include multiple passes. For G1 require R=W
and M=source reads/network/newly persisted=0. After parsing actual evidence call
`gpu_completion.cache_accounting(phase, reused=R, transferred=M, persisted=...,
source_read=..., cached=..., assigned=...)`; it checks these equations, not the
cache files or plan membership. Product validation and ownership reconciliation
remain mandatory. CPU full arena 111597465600 bytes / 21369 experts, one allocation
in this window, same bitmap/partition for G0/G1, warm reuse all bytes and zero
new CPU load/hash/allocation/expiry/unload/network weights in G1. G phases execute
only CPU-uncovered 18080; require positive actual work on local GPUs, remote GPU
and CPU and disjoint/exhaustive ownership.

## Complete Hermes continuation

Reuse existing approved ROLE_CONTROLLER and transport bindings; do not relocate
or change network/security settings or routes. Authorized controller SSH
orchestration is permitted. No local Python/tests/builds/models/fleet operations
on either user desktop. Focused test runs on 05; products only on 03/04/05.

1. Fetch this amendment, changed small sources and refreshed integration/runner
   manifests at the exact immutable publication commit. Verify author receipt
   hashes/bytes and all listed source dependencies, preserving planning-relative
   FILES_ROOT paths. New `gpu_completion.py` is required; changed `agent.py`,
   `commands.py`, `controller.py`, `policy.py`, `outer.py`, README and new
   `test_gpu_completion.py` belong to the same revision. Periodic-observation
   sources/tests remain unchanged and their reported PASS is reused. No old
   campaign, repository clone, dependencies, model hash scan, planner or rebuild.

2. Run exactly once on inferswarm05:

   ```sh
   "$PYTHON" -B -I "$FILES_ROOT/supervisor-integration-v1/test_gpu_completion.py"
   ```

   Entry limits 128 MiB address space, 5 CPU seconds, 10 wall seconds, 1 MiB
   regular output. Bound stdout/stderr to 1 MiB each; record hashes, argv,
   actual start/end, rc and output. Require exit 0, zero failures/errors/skips,
   exactly `GPU_COMPLETION_FIXTURES_PASS_V1 tests=8 live_signals=0 workloads=0`.
   Eight focused methods cover selective scheduling/order, retained baseline,
   setup/TTL/cap alignment, resumed directory and counter acceptance, window
   clipping and final usage. Authored NOT RUN here. Failure stops and preserves
   evidence; PASS **automatically continues G0/G1 only**, without another prompt.

3. While all old work is stopped, bind the baseline receipt from retained actual
   files and accepted settings. Preserve old report/archive and partial cache-B.
   Set `gpu_completion: true` in controller and all three agent configs,
   controller `retain: false`, agents `retention_approved: false`. Add the private
   `baseline` object to controller config. Keep GPU CACHE_B pointing to the
   existing exclusively owned real directory, including G1. Use new private run/
   output roots and the existing approved interpreter/binaries/config bindings.
   Mode mismatch is rejected before launch; no manual fake B phase receipts.

4. Use updated **`launch(config)`**, which captures start before creating any
   owned forwards/peers and selects the 2700-second GPU-only window. Alternatively
   the existing all-owner wrapper may call
   `Controller(peers, gpu_completion=True, baseline=receipt, retain=False, start=T)`
   with T before owned startup. Do not use default `coordinated(...)`, a historical
   `T-used_seconds` offset, or reset the old ledger. Count CPU load, cache
   validation/transfers, all engine/GPU work and teardown inside this new window.
   Capture actual end after all owned cleanup and report historical 4889.5,
   new elapsed, and their sum. All-stopped retrieval/coding/fixtures/baseline
   preparation/handoff are excluded. No running cap/lease extension or overtime.

5. Keep product `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`, main
   `34a59d83a51bdb3f0299d65306740bd357818715`, PR #2 draft/unmerged. Primary 960,
   helper 2247, CPU full 21369, uncovered 18080, remote 3289; CPU budget 106428
   MiB, payload 111597465600 bytes. Plan remains 92156 bytes with SHA above.
   No placement/plan/binary change. Keep strict local admission/pre-exec
   1152/512 MiB and warning-only valid local runtime crossings, host/cgroup
   memory/events, fixed-admission SwapFree decline >64 MiB, RSS/anonymous ceilings,
   required telemetry/identities/pidfds, periodic observation, leases/watchdogs,
   remote 4096 MiB reserve, fixed-plan fit and disk/cache guards. In particular
   retain existing G0 disk admission 22.25 GiB and runtime 4 GiB; no resource
   guard is weakened to make resumption pass.

6. Execute only the three GEN64 requests above and all acceptance here. Fix
   ordinary in-scope launcher/config/expectation mistakes without unnecessary
   approval, within this mode and its bounds. No repeated baseline, extra
   inference attempt, hidden CPU retry, cache deletion, product alteration or
   failure masking. Required failure stops dependent work. Preserve partial cache
   and old/new evidence; missing acceptance remains incomplete.

7. Stop CPU and all owned engines/GPU services/forwards inside the window,
   enforce baseline return <=30 seconds, bounded TERM/grace/KILL and actual
   parent/adopted reaping. Evidence <=256 MiB per role/controller, no model blobs.
   Preserve unrelated services. Return exact test/phase outcomes, baseline source
   identities and comparisons, validated R/M accounting, G1 zero-transfer and CPU
   reuse, separate historical/new/combined time, cumulative request/token charge,
   diagnostics, cleanup and evidence hashes privately. Stop at report; no merge,
   deployment, permanent service or additional attempt.
