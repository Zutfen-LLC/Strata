# Primary startup capacity correction

This supersedes the previous active sizing and continuation instructions. Earlier
hash-pinned budgets, planner sources, amendments, source handoffs and failure
receipts stay unchanged. Implementation bf51f09e, guard policies and product
binaries are unchanged. This is configuration/test/documentation only.

The user reports the df32c511 four sizing fixtures passed. B0 loaded and verified
21273 CPU experts,111096115200 bytes, one allocation in428.006 seconds (read265.067,
verify158.646), zero CPU network weight bytes. Peak RSS105854.8125 MiB, VmSwap0,
maximum CPU-host SwapFree decline4 MiB, engine-host decline0, no new cgroup events.
It then stopped during later serve/prompt initialization: primary1076 MiB free
versus1152 floor, helper554 versus512. No engine READY, inference or remote GPU
execution occurred. CPU bitmap identity was not exported before failure. All
owned products/guardians/forwards were reaped, arena released and GPUs returned
to baseline within.846 seconds. These are user-reported observations; no private
logs were inspected or runtime work performed here.

## Selected capacity and allocation reasoning

Choose **20 primary experts per layer**,960 total; keep helper2247 and remote3289.
One fewer per layer (1008) removes239.0625 MiB of requested storage, projecting
only163.0625 MiB above the floor from the reported1076 MiB sample. Two fewer remove
96*5222400=501350400 bytes=478.125 MiB, projecting1554.125 MiB free and402.125 MiB
above the unchanged floor. These projections assume other allocations unchanged;
neither is a measured free-memory result or a guarantee of full startup fit.

Pinned source review supports this larger bounded reduction:

- `generate.cpp:6112-6135` sizes native per-layer slots from actual per-layer
  blob strides. `ExpertCache::slot_stride` rounds blob+guard to256 bytes.
- `expert_cache.cpp:563-580` sums those strides and obtains one arena allocation;
  it does not allocate one driver block per expert. Here5222400 is already256-byte
  aligned. The isolated frozen environment omits STRATA_MMQ_DIRECT, whose default
  mode0 adds no slot guard (`moe_mmq.cu:559-574`). Elastic cache is off and helper
  mode prevents KV-growth VMM (`generate.cpp:5192-5207`). The nominal arena decrease
  is therefore501350400 bytes; driver granularity/other allocations remain separate.
- Cache sizing is earlier than hit scratch, prompt initialization and the verifier.
  The verifier's later arena alone includes16 staging blobs=79.6875 MiB, plus
  recurrent/attention/head/quantization/merge buffers (`verify.cpp:786-833`,
  `verify.hpp:kStagingBlobs`). Prompt initialization includes64 MiB GEMM scratch
  and32 MiB workspace, often borrowed from the cache; do not add borrowed bytes
  again as independent VRAM (`prefill.cpp:1489-1500`). Verifier initialization
  occurs later at `generate.cpp:9214`; library/context costs are not a proven
  fixed payload multiple. The failure's exact last completed allocation is unknown.

The extra per-layer reduction provides room for those remaining allocations
instead of relying on a163 MiB margin. It stays within the unchanged CPU envelope:

| Quantity | New value |
| --- | ---: |
| Primary | 960 (20/layer) |
| Helper | 2247 |
| Full CPU complement | 21369 |
| CPU payload bytes | 111597465600 |
| CPU payload MiB | 106427.63671875 |
| CPU CLI memory-budget MiB | 106428 |
| CPU budget slack bytes | 380928 |
| Remote GPU assigned experts | 3289 |
| CPU execution complement in G phases | 18080 |

The CPU cap is below107000 MiB; it cannot fit even one additional expert. If live
primary sizing clamps or actual ownership differs, fail before CPU arena allocation
rather than accepting a different partition. Admission107000+16384 MiB, RSS108024
MiB, runtime8 GiB headroom and all other protections remain unchanged. The previous
RSS plus478.125 MiB is only a projection, not evidence of the next run's peak.

## Helper changes and unchanged remote plan

Primary20/layer is a subset of primary22/layer because each layer admits its first
ranked eligible experts (`generate.cpp:6329-6365`). The helper scans the same
profile, skips primary holdings, and takes the first2247 (`remote_experts.cpp:159-169`);
there is one fixed helper, no stages, auto sizing, unranked tail or adaptations.

Helper IDs can change: dropped primary IDs can enter the helper and push some
old helper IDs into CPU. It would be wrong to claim helper identity is unchanged
or that the CPU adds exactly the dropped primary IDs. However **combined local
ownership can only shrink**: every old CPU ID had2247 eligible old helper IDs
before it, all of which remain nonprimary when primary shrinks. That CPU ID cannot
enter the new helper. New primary is also inside old primary. Thus new local union
is a subset of the previous local union, and old CPU complement is a subset of
the new complement. This also holds transitively from the original2263 helper.

Keep all existing remote IDs and the exact92156-byte plan, SHA256:
`a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578`.
Its3289 experts/17176473600 payload bytes remain disjoint from the new local union.
Pinned `generate.cpp:6740-6763` retains exact overlap/source/layout/digest checks;
the plan encodes the assigned set, not CPU bitmap. The subsequent CPU-membership
check still passes for the unchanged assignment. The full source manifest remains
valid. No metadata planner, plan rewrite, full-weight hash or payload movement is
needed for this correction. Engine setup derives a fresh CPU bitmap/partition;
record that identity and keep it identical across B0/B1/G0/G1.

The focused sizing entry now has five cases, including a nonuniform ranked fixture
where helper IDs change but the union inclusion and remote disjointness hold.
It covers rendered flags, budget coupling, new/old arena status and unchanged
resource floors. Authored NOT RUN here; earlier four-case PASS does not cover
this revision. The static review does not establish runtime admission or parity.

## Complete Hermes continuation

Continue the authorized experiment using this document's immutable commit as
ARTIFACT_COMMIT. No additional preparation gate. Use existing approved private
bindings: ROLE_ENGINE_A two local GPUs, ROLE_GPU_B network GPU, ROLE_CPU_C CPU
worker, ROLE_CONTROLLER controller. Keep implementation bf51f09e/main34a59d83
and PR2 draft/unmerged. No build/install/product change or security/network setup.

Retrieve only small planning files through existing authorized GitHub access:
`https://raw.githubusercontent.com/Zutfen-LLC/Strata/ARTIFACT_COMMIT/docs/network-gpu-worker/planning/`
Fetch current commands.py, cache_status.py, test_integration.py,
test_placement_sizing.py, integration README and SHA256SUMS.integration-v1,
this PRIMARY-STARTUP-SIZING-CORRECTION.md and SHA256SUMS.runner-v1. Verify full
publication-receipt hashes/sizes and retained dependencies against the manifests;
fetch only missing listed small files into a planning-relative FILES_ROOT.
No mixed revisions, repo clone, new supervisor or dependency/model download.
Preserve all earlier evidence and other files/assets.

Run ONE focused test on ROLE_CPU_C with the approved interpreter:

```sh
"$PYTHON" -B -I "$FILES_ROOT/supervisor-integration-v1/test_placement_sizing.py"
```

Reviewed entry limits128 MiB address space,5 CPU seconds,10 wall seconds and1 MiB
regular output; bound captured stdout/stderr1 MiB each using existing tooling.
No live proc/plan/model reads, workloads, subprocesses, sockets, GPU operations,
signals to other processes or live namespace/cgroup/PID1 prerequisites in fixtures.
Record exact argv/hashes/actual start/end/RC/output. Require exit0 and zero errors,
failures/skips with exact marker:
`PLACEMENT_SIZING_FIXTURES_PASS_V1 tests=5 live_signals=0 workloads=0`.
Do not rerun69+3, listener8 or the old four-case check. On focused failure stop and
preserve evidence; on PASS continue directly and automatically through the trial.

All prior owned work is reported gone, no arena remains. Confirm using existing
scoped ownership receipts, without stopping another service. Reuse and verify
the exact bf51f09e live binary receipts on all roles, including GPU binary, and
original profile/full manifests/private bindings/namespace provenance. Keep the
exact remote plan SHA/size/IDs above at both roles. No planner or manifest
regeneration, broad model hashing or preparation weight transfer is necessary.

Only active sizing flags now: `--expert-cache 960 --expert-cache-per-layer`,
`--expert-cache-device1 2247`, CPU `--memory-budget-mib 106428`.
Require actual primary960/helper2247/CPUfull21369 and111597465600 arena bytes.
G phases retain remote3289 and CPU-uncovered18080. Record the new CPU bitmap and
partition identity from actual setup/evidence when available; no identity was
exported from the failed run, so do not invent a comparison hash. Require the same
new CPU bitmap/arena in all four sessions. Unexpected counts/clamps refuse launch.

Time: retain cumulative615.809 seconds active+cleanup used, leaving4784.191 of5400.
The old absolute wall cutoff is withdrawn. All-stopped coding/retrieval/focused
fixtures/handoff time is excluded. At resume readiness capture local monotonic T
and wall correlation. In the existing approved private invocation use the existing
`Controller(..., start=T-615.809)` API, preserving all-owner finally cleanup.
Deadline becomes T+4784.191; launch/TERM cutoff T+4739.191. Unadjusted launch(config)
or coordinated(...) would reset5400 and must not be used. Count new owned transport/
guardian startup and teardown after T; no cross-host raw monotonic comparisons,
clock pauses, deadline reset or running-process lifetime extension.

Run unchanged serial B0 application-cold CPU, B1 resident CPU, G0 cold remote GPU
cache, G1 warm remote GPU cache. CPU worker is new; engine SETUP initiates loading,
no standalone preload. One arena retains all21369 blobs including remote duplicates.
New engine per phase and new remote GPU process per G phase; same warm cache G1.
Stop/reap engine/GPU workers between phases,11-second gaps after B0/G0. Engine caps
1200/600/1800/900 seconds; GPU caps1900/1000; CPU absolute7200 seconds, CPU setup
600000 ms, GPU setup1200000 ms, idle1800000 ms. Respect required full remaining
phase envelopes and cleanup reserve; a phase may remain NOT RUN if time is short.

Identical settings and request bytes each phase, prompt-cache0/full reads:

```text
GEN 64 9419
GEN 64 14773,6353
GEN 64 9419
QUIT
```

Usage remains0/12 requests,0/768 generated tokens; max64/request. No inference
retry or extension. Preserve all other flags/settings. Keep primary1152/helper512
MiB floors, CPU107000+16384 MiB admission/RSS108024/runtime8 GiB; engine12 GiB
admission/4 runtime/RSS24/anonymous8; GPU host4 GiB admission/2 runtime/RSS2, GPU
reserve4096 MiB/cache18432 MiB, cold disk22.25 GiB/runtime4 GiB. Preserve strict
fixed-admission SwapFree decline>64 MiB, cgroup events/headroom, ownership/pidfds,
coverage/READY, leases, watchdog and bounded TERM/grace/KILL/reaping. Missing required
coverage or any existing guard failure stops work. No reserve relaxation, hidden
CPU retry, live rebalancing, auto-sizing or unapproved failure injection. The
478.125 MiB requested-cache saving is provisional; actual live guards decide fit.

Acceptance: reconcile disjoint/exhaustive execution ownership and positive actual
work on both local GPUs/network GPU/CPU. CPU serves only uncovered18080 in G phases,
while resident full21369 remains unchanged across all sessions. One CPU allocation,
warm reuse111597465600 bytes, zero new load/hash/allocation/expiry/unload and zero CPU
network weight bytes. G0 assigned/cached/logical expert reads/network/new persisted
each17176473600 bytes; G1 reads/network/new persisted0, reuse17176473600. Separate
physical I/O/transport/staging/RSS/GPU allocation; reconcile client/worker/route/helper
counters and finite valid outputs. No whole-model mirror or unassigned transfer.

First-logit files exactly4*vocab finite float32 bytes using trusted vocab; missing/
short/write-error evidence fails gate. B1/G1 predeclared screening: centered relative
L1<=.03, KL(softmax(B1)||softmax(G1))<=.03 nats at temperature1. Center means, denominator
max(sum(abs(centered baseline)),1e-12), stable double log-softmax. Within-mode drift
maxabs<=1e-5*max(1,maxabs(reference)). Report maxabs/RMSE/top1/margin/token agreement;
no bitwise/all-token parity claim. Compare B1/G1 like-for-like with actual tokens;
separate setup/transfer/cache/READY, engine/prompt/decode/request/wait/worker/helper
timing. No speedup gate or cosmetic rerun. Preserve original criteria except the
explicit sizing/time amendments here; missing evidence is incomplete.

Failure stops dependent phases and all owned work under unchanged cleanup. Release
owned engine/GPU tasks and close out evidence inside remaining active allowance;
GPU return accounting<=30 seconds, evidence<=256 MiB per role/controller, no weights
in archives. CPU retention only if already explicitly approved: earliest success+
30 min, actual idle expiry, original admission+2 h; failure stops it, default-stop.
Bounded cache retention only by explicit prior decision, no accumulating roots or
broad deletion. Preserve unrelated services and historical evidence. Return actual
phases, identities/ownership, active-time/request/token use, measurements, cleanup
and evidence hashes privately. Stop at report; no merge/rebuild/permanent service.
