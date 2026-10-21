# Helper capacity correction; guards and remote assignment unchanged

This amendment supersedes older active sizing/continuation instructions only
where stated below. Historical execution budgets, metadata plans, source handoffs
and failed reports remain unchanged. It changes supervisor configuration and
expected arena bytes, not implementation bf51f09e or any resource guard.

The user reports the eight focused listener cases passed. B0 then stopped during
local helper setup: helper free memory 510 MiB versus the unchanged 512 MiB floor;
primary 1212 MiB versus 1152 MiB. No CPU arena was loaded; remote GPU and inference
never started. All owned processes/forwards were reaped, no residency remains,
and active execution plus teardown consumed 56.3 seconds. These are user-reported
receipts, not locally reproduced results or independently inspected private logs.

## Fixed sizing and source compatibility

| Quantity | Corrected value |
| --- | ---: |
| Primary experts | 1056 (22 per layer) |
| Helper experts / CLI device1 capacity | 2247 |
| Full CPU complement | 21273 |
| Full CPU payload bytes | 111096115200 |
| Full CPU payload MiB | 105949.51171875 |
| CPU CLI memory-budget MiB | 105950 |
| Budget slack bytes | 512000 (less than one expert) |
| Unchanged remote GPU experts | 3289 |
| CPU execution owners in G phases | 17984 |

Each expert is 5222400 bytes. Removing 16 helper experts frees 83558400 payload
bytes, or 79.6875 MiB, provisionally. This is not a measured increase in available
VRAM and does not guarantee successful admission. Primary clamp or any unexpected
local count still fails the tight CPU budget before allocation; even one extra
CPU expert does not fit. Do not respond by relaxing guards or resizing again.

Static inspection of the unchanged pinned implementation establishes compatibility
for the same frozen profile/primary selection:

- `src/program/generate.cpp:6559-6630`: with this single fixed helper and no auto
  sizing, its ranking is unchanged when only the requested count decreases.
- `src/core/remote_experts.cpp:159-169`: select the first unclaimed ranked experts
  until the fixed slot count. The 2247 selection is a prefix subset of the old
  2263. Primary selection is unaffected; no newly selected helper expert can
  collide with the existing remote assignment.
- `src/program/generate.cpp:6692-6768`: derive a fresh CPU bitmap from current
  local ownership; validate remote overlap against those local holdings and
  compare the plan to one rebuilt from the full source manifest and its exact
  assigned IDs/model/layout/digests. That comparison does not encode CPU bitmap.
- `src/program/generate.cpp:6791-6796`: require every remote assigned ID inside
  the freshly frozen CPU partition. Enlarging that complement retains all old
  assigned IDs. CPU partition identity is freshly computed at setup.

Keep the exact 92156-byte binary remote plan and all its 3289 IDs unchanged:
`a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578`.
Remote payload remains 17176473600 bytes. Do not run the metadata/default planner,
regenerate a manifest, hash all model weights or transfer unassigned weights.
The full source manifest and profile retain their existing verified identities.
The new CPU bitmap intentionally adds the 16 dropped helper IDs. It differs from
the previous trial bitmap; freeze it identically across B0/B1/G0/G1. No old arena
is reused or shrunk. No exact new CPU bitmap hash is claimed before setup.

The 107000 MiB arena admission envelope plus 16384 MiB reserve, CPU RSS108024 MiB,
all role host/cgroup/anonymous limits, 1152/512 MiB local GPU floors and 4096 MiB
remote reserve remain unchanged. Cache18432 MiB, disk guards, fixed-admission
SwapFree decline strictly greater than64 MiB, events, ownership/watchdogs and
deadlines remain unchanged. CPU status now requires111096115200 arena bytes;
the focused test and existing status fixture expect the same value.

`test_placement_sizing.py` has four deterministic cases: rendered engine/plan
flags, tight CPU budget/complement, new-versus-old idle arena bytes, and unchanged
role/GPU resource limits. These fixtures are authored NOT RUN here. Existing
69+3 and eight-listener results remain attributed to prior user reports, covering
their executed revisions; changing this configuration does not invent a new PASS.

## Complete Hermes continuation

Continue the already authorized experiment, without another preparation gate.
Use the immutable commit containing this document as ARTIFACT_COMMIT. Preserve
all earlier evidence, including the helper-floor refusal; no merge or PR change.
ROLE_ENGINE_A is the existing two-GPU engine role, ROLE_GPU_B the network GPU,
ROLE_CPU_C the resident CPU worker and ROLE_CONTROLLER the existing controller.
Use their already approved private bindings/transport/authentication. No real
bindings or private raw logs belong in GitHub.

1. All owned work is reported stopped. Confirm that remains true using existing
   scoped ownership receipts; do not stop another service. Fetch only small
   planning artifacts from the immutable revision through existing GitHub access:

   `https://raw.githubusercontent.com/Zutfen-LLC/Strata/ARTIFACT_COMMIT/docs/network-gpu-worker/planning/`

   Fetch refreshed `supervisor-integration-v1/SHA256SUMS.integration-v1` and
   `namespace-runner-v1/SHA256SUMS.runner-v1`, this amendment, integration README,
   commands.py, cache_status.py, test_integration.py and test_placement_sizing.py.
   Match their full hashes/sizes to the publication receipt. Verify retained
   dependencies against the manifests, or fetch only those listed small files
   into a new planning-relative FILES_ROOT. Do not mix revisions. Preserve other
   files/assets. No repo clone, build, dependency/model download or new wrapper.

2. Run exactly one focused check on ROLE_CPU_C, before restarting owned work:

   ```sh
   PYTHON -B -I FILES_ROOT/supervisor-integration-v1/test_placement_sizing.py
   ```

   Substitute the approved interpreter and artifact root. The reviewed direct
   entry enforces128 MiB address space,5 CPU seconds,10 wall seconds and1 MiB
   regular-output size. Bound captured stdout/stderr to1 MiB each with existing
   tooling. No proc/model/plan reads, subprocesses, signals to other processes,
   sockets, live namespace/cgroup/PID1 prerequisites or GPU work in these tests.
   Record exact file identities, argv, actual start/end, RC and bounded output.
   Require exit0, zero errors/failures/skips and exactly this marker:

   `PLACEMENT_SIZING_FIXTURES_PASS_V1 tests=4 live_signals=0 workloads=0`

   Do not rerun the eight listener or69+3 campaign. On focused failure, stop and
   preserve evidence. On PASS, continue directly through the four model phases.

3. Before launch, retain and verify existing bf51f09e product binary receipts on
   each role (including the actual GPU binary), original profile/full manifests,
   private bindings, namespace provenance and exact remote plan SHA/size above.
   Do not recreate the assignment or source manifest. Only active flags change:
   `--expert-cache-device1 2247` and `--memory-budget-mib 105950`.
   Require primary1056/helper2247/fullCPU21273 and exact CPU payload111096115200.
   In G phases require remote3289/uncoveredCPU17984. Record the fresh CPU bitmap
   identity; all four sessions must retain that same bitmap and one arena.

4. Time accounting: the former absolute wall cutoff is withdrawn. Stopped coding,
   retrieval, focused fixtures and handoff do not consume active workload time.
   Retain56.3 seconds consumed; total active allowance remains5400 seconds, so
   only5343.7 seconds remain. At resume readiness take controller-local monotonic
   anchor T and correlated wall time. The existing `Controller(..., start=...)`
   API accepts `start=T-56.3`, giving deadline T+5343.7 and launch/TERM cutoff
   T+5298.7. Use that existing constructor argument in the approved private
   invocation, preserving its all-owner finally/cleanup. `launch(config)` and
   `coordinated(...)` default to a fresh5400 seconds and do NOT carry usage; do
   not call them unadjusted. No new supervisor/deadline manager is needed.
   Include newly owned transport/guardian startup and all active teardown after T
   in the ledger. Do not reset this deadline between phases or compare raw host
   monotonic timestamps. No existing process lifetime is extended or paused.

5. Run the unchanged supervised serial B0/B1/G0/G1 sequence automatically on
   focused PASS and successful admission. Engine setup initiates CPU loading;
   there is no standalone preload. One new CPU worker/arena, reused between all
   runs. B0 application-cold CPU baseline; B1 resident CPU baseline; G0 fresh
   network GPU process/new empty owned cache; G1 fresh GPU process/same warm cache.
   CPU keeps all21273 blobs, including remote duplicates. Existing engines/GPU
   processes stop and release after each phase;11-second gaps after B0/G0.
   Engine caps1200/600/1800/900 seconds, remote GPU1900/1000, CPU absolute7200,
   CPU setup600000 ms, GPU setup1200000 ms and idle1800000 ms remain unchanged.
   Refuse a phase without its full remaining required envelope and cleanup reserve.
   No lifetime reset, hidden CPU retry, model failure injection or auto-resizing.

6. Identical settings and request bytes in each phase, prompt-cache0/full reads:

   ```text
   GEN 64 9419
   GEN 64 14773,6353
   GEN 64 9419
   QUIT
   ```

   Cumulative usage starts0 requests/0 tokens; max12 requests/768 generated tokens,
   max64/request. No retries/extensions after inference. Preserve all flags and
   numerical/evidence rules from the original execution budget except the explicit
   sizing/time amendments here. All admission/runtime protections remain active:
   primary1152 MiB/helper512 MiB, CPU123384 MiB admission/RSS108024 MiB/8 GiB
   runtime headroom; engine12 GiB admission/4 GiB runtime/RSS24 GiB/anon8 GiB;
   GPU host4 GiB admission/2 GiB runtime/RSS2 GiB, VRAM4096 MiB/cache18432 MiB,
   cold disk22.25 GiB/runtime4 GiB. Actual available VRAM decides;79.6875 MiB
   freed payload is an estimate. Required missing telemetry, any cgroup event,
   fixed-admission SwapFree decline>64 MiB or other existing guard stops work.
   Preserve pidfd identity, listener/READY coverage, leases, watchdog and reaping.

7. Acceptance/evidence: disjoint/exhaustive execution ownership and positive work
   on primary/helper/network GPU/CPU; remote executes assigned IDs only, CPU serves
   uncovered17984 only, CPU resident bitmap/arena identical in all sessions, one
   arena allocation and warm reuse111096115200 bytes with zero new CPU weight
   network bytes/load/hash/allocation/expiry/unload. G0 assigned/cached/logical
   expert reads/expert-network/new-persisted bytes each17176473600; G1 expert
   reads/network/new-persisted0 and reused17176473600. Separate physical I/O,
   transport, staging, RSS and GPU allocation figures; no whole-model mirror.
   Reconcile actual route/client/worker counters and finite valid outputs.

   First-logit files must have exactly4*vocab finite float32 bytes, trusted vocab,
   no write errors. Predeclared B1/G1 screening: centered relative L1<=.03 and
   KL(softmax(B1)||softmax(G1))<=.03 nats, temperature1; center means, denominator
   max(sum(abs(centered baseline)),1e-12), stable double log-softmax. Within-mode
   drift maxabs<=1e-5*max(1,maxabs(reference)). Report maxabs/RMSE/top1/margin and
   token agreement; these observations do not prove all-token/bitwise parity.
   Compare like-for-like B1/G1 timing with actual tokens and transfer/setup/READY,
   prompt/decode/request/wait/worker/helper timing categories separately. No speedup
   gate or cosmetic rerun. Missing evidence is incomplete, not a claimed PASS.

8. Failure cancels dependent phases and stops owned work with unchanged bounded
   TERM/grace/KILL/reap; never relax a guard to continue. Complete owned GPU/engine
   cleanup and bounded evidence closeout inside the remaining active allowance;
   per-role/controller evidence<=256 MiB, no model blobs in archives. Confirm GPU
   returns within the existing30-second accounting bound. CPU retention ONLY if
   already explicitly approved, earliest of success+30 min, actual idle expiry,
   original admission+2 h; failure stops it, otherwise default-stop. Keep bounded
   cache only by explicit prior decision, no accumulating roots or broad deletion.
   Do not affect other services. Return phases, exact identities, active-time and
   request/token consumption, measurements, cleanup and evidence hashes privately.
   Stop at report, with no merge, new placement, rebuild or permanent service.
