# Fixed two-host CPU group: Hermes handoff

Base source: `c9cef19a35369d5a3721d11515d524ebd6476da1`. Candidate is the full published PR head
containing this file; bind `CANDIDATE` to that exact SHA, never a moving branch. Source, fixtures and
commands are authored. **Builds, executable tests and inference: NOT RUN on Geekom/Valinor.**
No speedup or numerical acceptance is established. No fleet work was dispatched by this coding task.

The change adds only the opt-in fixed CPU endpoint group, necessary per-host use of the existing
guardian/config driver, focused fixtures and bounded host capture/replay diagnostics. Wire messages, worker implementation and default
one-endpoint route stay unchanged. It does not add discovery, rebalancing or a network GPU mode.

## Prerequisites and focused checks

The original full-model comparison at 9ed478e5543885e019205512f2450ab4336d393e has completed.
The parent reports that the split arm failed every frozen first-logit screen and was slower; the
local baseline/restore first logits were bit-identical. This is supplied evidence, not execution
or raw-receipt inspection by this coding thread. **Do not rerun that campaign or either duplicate-arena
diagnostic.** Only the record0 real-weight microfixture procedure below is current. All other executable commands
in this file are historical, not authorization to repeat prior work.

### Current status: resident-monitor incident; duplicate-arena diagnostics withdrawn

The supplied sequential attempt at `dc772896dcd78ca8e3dbed331e821ee8f24c4f69` stopped during the
first temporary subset load at157s. No actual-subset numerical result was produced. Full replay
lowered the original worker's VmSwap from422148KiB to276464KiB; it later reached1060228KiB.
The external guard terminated replay, and the temporary children were reaped.

Subsequent supplied triage confirms that the **original resident's own monitor** recorded sustained
owned VmSwap above1GiB for30s, then caused the guardian to stop/unload the116135731200B arena and
exit0. Observed swap1060228KiB exceeded its1048576KiB threshold. Recorded minimum MemAvailable
was38157612KiB (about36.4GiB). No kernel/cgroup OOM event was found. The external guard targeted
the replay timeout process only; it was not the1800s deadline. All original and temporary processes
were confirmed absent after the incident, with the service port free. This is a **resource stop, not a numerical
failure**. It does not establish subset mapping or group correctness.

The source agrees with the confirmed signal path: the retained monitor sends through its worker
pidfd, and the guardian owns worker cleanup. The diagnostic controls only its forked temporary
children and never sends STOP/UNLOAD to the full endpoint. No production, monitor, protocol or
diagnostic code change is needed for this incident correction.

The prior handoff incorrectly instructed the external guard to tighten its ceiling after a swap
decrease. That was an added interpretation, not the parent's fixed-baseline policy. **The baseline
is the original worker's freshly measured, stable prelaunch VmSwap and stays fixed for the run.
Decreases never lower that ceiling; an increase above the initial ceiling requires stopping.**
Record initial, maximum and final values. The independent resident monitor retains its own original
thresholds and may stop the worker regardless of the diagnostic guard. At1060228KiB, both the
initial422148KiB diagnostic ceiling and the monitor's1GiB threshold were exceeded; ratcheting is
not required to explain either stop. Neither guard policy is changed here.

**Stop duplicate-arena experiments, including the sequential one-half mode.** The previous
`--local-subsets` procedure and config template are withdrawn from runtime use; their source remains
for history, not authorization to retry. Do not attempt recovery sweeps, swapoff, mlock, settings
changes or another diagnostic load. Parent owns restoring only the normal full resident with four
loaders/twelve inference workers and the unchanged original guard policy. This coding task launches
nothing and makes no zero-swap or timing-qualification claim.

### Current next step: record0 real-weight microfixture, no full resident connection

The parent supplies a completed G1 PASS using the retained engine/worker and `ddb0d652` helpers:
one original prompt0 GEN256 completed213 tokens and102724 entries, exactly A0. Its993280B raw-F32
first-logit file was bit-identical to A0; all reported error metrics were0. READY reported one group
member, cache hit/reused116135731200B and zero setup local/network bytes. The original full arena
remained allocation1 with zero unload/expiry/swap, and engine owners were reaped. Staging was fixed
through four existing `@asset` tokens, without source change. A post-summary wrapper KeyError was
corrected read-only, without rerun. These are supplied Hermes results, not execution or private
receipt inspection by this coding thread. They do not clear the original failed split comparison.

The remaining unconfirmed seam is actual subset loading/lookup or two-peer integration. Source
uses global `(layer, expert)` keys for loading, digest verification and inference lookup; no static
mapping defect was found. The new mode of the existing `cpu_group_numeric_diagnostic` fixes the next
input to captured record0, layer0/T2/K10. It derives peer ownership from the COMPLETE original CPU
bitmap before filtering to that record's referenced experts. IDs are never renumbered.

The independent reference reads one real Q8 blob at a time, verifies its trusted global-ID digest,
evaluates native gate/up/down rows without the worker pool, and reduces signed weighted rows in
frame-entry order. Its full result must bitmatch the captured reply before any child is created.
Only then do two owned loopback workers load their disjoint selected IDs through the original
manifest/local source path. Each actual peer must bitmatch its independent reference; the real
`ExpertWorkerGroup` must bitmatch the manual peer sum. The first READY peer uses the existing setup
keepalive while the second loads; both exchanges drain before comparisons. Group setup reuses the
small subsets, with no reload. No worker protocol, production arithmetic, capture format, engine,
GEN, remote endpoint, full resident connection or new sequencing framework is added.

Bounds are enforced in code: at most20 distinct selected experts, each5222400B, and104448000B
(99.61MiB) TOTAL arena payload across BOTH children. Each child budget is its exact assigned payload
rounded up only to MiB. Reference payload staging is one5222400B blob, released before children;
row scratch is204800B plus bounded native packing/hidden buffers and capture/manifest storage.
The existing guards each cap parent+one-child RSS at1GiB, so combined owned RSS is conservatively
bounded by2GiB. They require MemAvailable>=8GiB, owned VmSwap0, finite-cgroup headroom and unchanged
host/cgroup OOM counters, with a180s work deadline. Each child log is exclusive0600 and capped1MiB;
the output directory is new/private. Cleanup signals/reaps only positive retained child PIDs, including
failure during second-child startup. No full or half-complement arena is loaded or replaced.

On the CPU fixture host, retain the existing cache/compiler/native options. Build only the new
focused test and existing diagnostic target; no CUDA/native-OFF/product rebuild is requested:

```sh
cmake -S "$SRC" -B "$CPU_BUILD"
cmake --build "$CPU_BUILD" --target cpu_capture_micro_test cpu_group_numeric_diagnostic --parallel 4
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^cpu_capture_micro_test$'
```

The small Q8 regression checks original ownership despite skipped GPU/unselected IDs, shape/ownership
and single-active-peer refusals, digest/full-reference failure before children, positive-owned cleanup
when the second executable exits before listening, and successful peer/group bits. Its success case
delays second setup11s, beyond the first peer's10s idle deadline. Fixture payload files are larger
than tiny stdout limits: bound test output through the retained drain/timeout guard rather than a
small inherited file-size limit. All executable checks are **NOT RUN on Geekom/Valinor**.

After that test passes, parent selects runtime on the CPU fixture host, preserving the separate
full resident host. Bind these existing inputs; do not copy a full model:

| Binding | Required input |
| --- | --- |
| `DIAGNOSTIC` | New CPU `cpu_group_numeric_diagnostic`, exact candidate SHA and mechanically verified executable hash |
| `WORKER_A`, `WORKER_B` | Retained compatible original peer0/peer1 worker ELFs, already used by the earlier same-host diagnostic; verify their complete hashes (reported prefixes `e9f25fa5` / `3b41fa8b`), not newly built fixture workers |
| `CAPTURE` | Existing37544692B capture, SHA256 `218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208`; transfer only this file if absent |
| `MANIFEST` | Original trusted local manifest, original full hash, at most8MiB; identity/index must match the capture |
| `LOCAL_ROOT` | Existing valid local Q8 assets on the fixture host; source files are read-only and not staged/copied |
| `NEW_MICRO_OUTPUT` | New absolute directory beneath the existing disk-backed private run root, not tmpfs; parent directory already exists |

The two ELF sizes and actual manifest size come from their retained file receipts, not guesses.
Before weight reads, bounded `MICRO_RANGE` lines list every selected global ID, role, source basename,
exact offset/length and expected logical source-file size (at most60 ranges, total<=104448000B).
If an existing source file is absent/invalid, the mode refuses before workers. Return those precise
missing bindings and ranges; do NOT transfer entire shards or synthesize altered manifests. The
current `LocalSource` requires the original manifest's filenames/sizes; sparse-range staging is not
implemented by this change. Reuse the valid assets from the earlier subset run where available.

Run once under the retained external owned-process/output guard, after live fixture-host headroom,
cgroup, swap and OOM checks. Do not infer safety from MemAvailable alone. No port/PID of the original
full resident is accepted by this CLI:

```sh
timeout --signal=TERM --kill-after=10s 240s "$DIAGNOSTIC" \
  --worker-a "$WORKER_A" --worker-b "$WORKER_B" \
  --capture-record0 "$CAPTURE" "$MANIFEST" "$LOCAL_ROOT" "$NEW_MICRO_OUTPUT"
```

Require the independent-full/capture bit gate with `workers_started=0`, then both actual-peer bit
gates and actual-group/manual bit gate. Final `CPU_CAPTURE_MICRO` must report ok/reference/peer0/
peer1/group/owned_reaped all1, payload<=104448000, GEN0 and full_endpoint_connections0. Each child
must have one arena allocation, exactly its selected loaded/verified bytes, no upload/failure/unload/
expiry and two INFER requests (manual then group). Return source/binary/manifest/capture pins,
selected IDs/ranges and original/micro slot receipts, all gates, descriptive regrouping error,
resource/OOM and owned-child cleanup receipts. `peak_owned_rss_bound` sums two independently sampled
parent+child maxima; it is a conservative bound, not a simultaneous RSS measurement. Preserve every
failure and do not repeat automatically. No inference performance claim is made.

Interpret failures in order: reference mismatch stops before workers and questions input/weight/kernel
alignment; a peer mismatch witnesses selected-subset loading/execution; matching peers with a group
mismatch witnesses the group path. An all-pass result covers only these selected real inputs/weights
and local two-peer integration. `MICRO_SLOT` exposes the original split rank and the smaller fixture
rank: full-size compacted slot ranks, the remaining layers, remote state/network behavior, live
two-peer engine integration and closed-loop amplification remain untested. Frozen screens and the
original failed split verdict stay unchanged. The withdrawn full/half duplicate-arena tests stay withdrawn.

### Completed G1 live-engine request — supplied PASS; historical commands

The parent supplies a preliminary PASS for the `b224ef87f73d5da9724cb1f8f2a87859fc4adfbd`
single-member saved replay in5s; its full receipt is pending. Do not repeat it. The next discriminator
uses the retained original engine, one full resident worker and the original prompt0 GEN256 line.
This exercises live engine/group interaction without two-peer state or another arena. The original
split-model numerical/performance verdict remains failed.

This additive change touches only the existing Python harness, focused regressions and docs.
Top-level `cpu_route` defaults to `"legacy"`; explicit `"group"` requires exactly one
`--remote-cpu-worker` in base argv, bound to the same worker/control endpoint. The opposite flag,
duplicates, invalid route values or endpoint/manifest mismatches are refused before staging or
owner startup. The selected route survives `arm_config`, including existing single-peer rewriting.
The omitted-field legacy argv and existing two-peer sequencing remain unchanged. No new sequencer,
worker guard, engine feature or C++ build is introduced.

Hermes runs these focused Python tests once on its controller, with exact candidate source and
the retained Python. No model, GPU, native binary or package installation is needed:

```sh
cd "$SRC/tools/validation-harness/tests"
timeout --signal=TERM --kill-after=5s 120s python3 -B -m unittest -v \
  test_harness.HarnessTests.test_cpu_route_legacy_default_preserves_argv \
  test_harness.HarnessTests.test_cpu_route_single_group_survives_staging_and_arm_selection \
  test_harness.HarnessTests.test_cpu_route_invalid_conflicting_and_duplicate_flags_refused \
  test_harness.HarnessTests.test_cpu_route_exact_endpoint_and_manifest_bindings \
  test_peers.PeerTests.test_explicit_group_route_survives_single_peer_argv_rewrite \
  test_peers.PeerTests.test_membership_config_keeps_exact_total_and_legacy_primary_path \
  test_harness.HarnessTests.test_worker_controls_never_queue_behind_engine_session
```

These tests are authored, **NOT RUN on Geekom/Valinor**. After they pass, copy the existing private
warm-attach config and change only the following fields. Use the new source pin for harness helpers,
but keep the retained `9ed478e` engine and full worker binaries; verify their complete recorded
hashes rather than relying on the supplied prefixes `66750645` and `e9f25fa5`.

| Config field | One-request value/delta |
| --- | --- |
| `run_root` | New private output root; never overwrite an earlier bundle |
| `cpu_route` | `"group"` |
| engine `argv` | Replace the one `--remote-cpu-experts` token with `--remote-cpu-worker`; keep its existing numeric loopback endpoint and all other arguments |
| `attach_root`, `attached_worker_idle` | Current recovery receipt root, whose `WORKER/identities.json` must match the live guardian/worker/monitor; `true` after exclusive idle verification |
| `final_worker`, `failure_worker` | Both `"retain"` |
| `worker_bytes` | `116135731200` |
| `assets.engine`, `assets.worker`, manifest/model/profile/request assets | Retained original bindings; no binary replacement or asset regeneration |
| `assets.reference` | Existing retained A0 summary JSON with the original request1 token array and worker-entry count |
| `arms` | Exactly one copied warm arm, named `G1`, `restart:false`, `requests:[0]`, `reference:"reference"` |
| arm workers/loaders/masks and engine/worker policy | Keep12/4, caller0/pool1..12 and all retained engine/GPU/environment settings |
| arm `placement_markers` | Keep960 primary/1378 helper/22238 CPU markers; replace any legacy CPU-ready marker with `fixed CPU group ready peers=1`; require `cache_hit=1 reused=116135731200 B` and `setup network=0 B local=0 B hits=0 misses=0` |
| `capture_first_logits` | `true`; remove global `STRATA_DUMP_FIRST_LOGITS` so the driver supplies a unique prefix |
| `env.STRATA_CPU_CAPTURE` | Remove; no new host capture |
| `worker_peers`, arm `cpu_peers` / `peer_bytes` | Omit; this is one attached full resident, no remote controls |

Keep `--remote-cpu-local-manifest`, `--remote-cpu-resident`, native assets, primary/helper GPU placement,
prompt cache, greedy/speculative settings, budgets and all deadlines unchanged. Do not retain both
CPU endpoint flags. `requests:[0]` sends the exact first GEN256 line from the retained request asset,
then waits for its complete DONE and QUIT/reaping. Its A0 output is213 tokens;256 is a cap, not an
expected token count. The existing reference gate compares the complete token array and entries.

Before launch, Hermes must verify the current original guardian/worker/monitor identities, owned
listener, complete idle forever-resident116135731200B arena, VmSwap0 and healthy unchanged resource
guard. Preserve the fixed prelaunch swap ceiling, MemAvailable>=8GiB and unchanged OOM counters.
The retained engine has normal resident negotiation, **not a reuse-only switch**: an empty arena
could load. Do not launch if the live resident preflight fails. A mismatched complete cache is
refused by the worker without replacing it. Same placement/manifest/geometry and one group member
produce the same full partition (`ordinal % 1`), so require an observed cache hit/reused116135731200B
and zero new local-load/upload bytes at READY before GEN. Do not issue worker STATUS/STOP/UNLOAD
while the engine owner is alive; inspect the existing READY log and retained session receipts.
The configured placement markers enforce those reuse/zero-load observations before the existing
driver sends GEN; a missing marker fails the arm without changing its sequence.

Use the unchanged staging/run entry points after the focused tests and parent dispatch:

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$PRIVATE_CONFIG"
python3 -B "$RUN_ROOT/run.py" --run-root "$RUN_ROOT"
```

The first operation mechanically stages/hashes the new helpers and unchanged assets once; it starts
no owner. After complete engine-owner reaping, require the original resident still complete, idle and
forever, with resident/loaded/verified/peak bytes and allocation/unload/expiry counters unchanged,
zero new local-load/upload bytes, no failures and VmSwap0. Preserve stdout, first-logit file and
failure receipts even if the token reference gate fails. No automatic repeat or restore load.

Compare the new `G1-first-logits.0` to the retained A0 prompt0 raw-F32 file with the existing bounded
`read()`/`compare()` functions (the comparator CLI expects three pairs). This invokes one pair without
adding a script or changing any numerical threshold. The retained row is993280B of raw F32, with no
count header:

```sh
PYTHONPATH="$SRC/tools/validation-harness" python3 -B -c '
import json, pathlib, sys
import compare_logits as c
a, b = map(pathlib.Path, sys.argv[1:])
metrics = c.compare(c.read(a), c.read(b))
metrics["bitwise_equal"] = a.read_bytes() == b.read_bytes()
print(json.dumps(metrics, indent=2, allow_nan=False))
raise SystemExit(0 if metrics["bitwise_equal"] and metrics["numeric_pass"] else 1)
' "$A0_PROMPT0_FIRST_RAW_F32" "$RUN_ROOT/G1-first-logits.0"
```

Return one flat receipt: source/helper/engine/worker/config pins, exact sent line, DONE213 and complete
token equality, raw-F32 length/finite/bit comparison plus unchanged error metrics, placement/affinity,
observed setup cache hit, zero new load/upload bytes, final owned cache/resource accounting and
cleanup. This is one diagnostic request, not a performance campaign. A failure is preserved; a pass
does not clear two-peer/subset/remote behavior or the original failed split comparison.

### Completed single-member saved replay — preliminary supplied PASS; historical commands

The parent reports normal full residency restored with the retained worker binary, four loaders and
twelve inference workers. That is a supplied snapshot, not a live check by this coding thread.
Hermes must reverify the owned receipt, listener identity, complete forever-resident116135731200B
arena, exclusive idle session, current swap and original monitor health immediately before replay.
No worker replacement, second worker, additional arena, engine, GEN or settings change is requested.
Duplicate-arena procedures remain withdrawn.

The new `--single-group` option first runs the existing capture/request/logit validation and direct
full/masked replay unchanged. All full replies must bitmatch before anything masked; the existing
detail gates must pass. It then closes that session and verifies cache accounting before opening an
actual **one-member `ExpertWorkerGroup`** with the identical full partition and trusted manifest.
`resident_reuse_only=true` refuses empty/mismatched residency; fallback/upload is disabled. Every
record, including zero-work records, goes through actual group post/finish with the original
activations, IDs, weights, masks, T/K and layer. Fresh session/sequence identity is supplied by the
existing tier. Every result must be finite and bitwise identical to the saved full reply. No numerical
tolerance or production arithmetic changes are included.

The group drains/joins its existing tier, disconnects without STOP/UNLOAD and verifies complete idle
forever residency again. Resident/loaded/verified/peak bytes, allocation, unload and expiry counters
must equal the original snapshot. Exactly one additional cache hit and one arena's reused bytes are
expected for this group session; local reads/uploads must stay zero. Scratch is one bounded request,
reply, ID conversion and group row at a time, reusing the existing capture in memory. No masked
reference arrays or child processes are added by this mode. The direct phase and group phase each
have a180s deadline, with5s cache-control and10s group-frame limits.

On the CPU fixture host, use the existing build cache/options at the exact new source pin. Reconfigure
only to discover the new test target; do not change compiler or native options:

```sh
cmake -S "$SRC" -B "$CPU_BUILD"
cmake --build "$CPU_BUILD" --target cpu_capture_test cpu_capture_group_test cpu_capture_replay --parallel 4
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^(cpu_capture_test|cpu_capture_group_test)$'
```

`cpu_capture_group_test` owns one small real Q8 fixture worker, with one arena allocation. It checks
T2/T1 and K10, mixed ownership/weights, zero work followed by active work, full-validation failure
preventing group setup, empty/stale-cache refusal, group bit-mismatch cleanup preserving the primary
error, valid reconnect after failure and final owned-child reaping. It reuses existing fixture/lifecycle
helpers and needs no model or external endpoint. The unchanged `cpu_capture_test` covers corrupt,
truncated, oversize and malformed capture refusals. Avoid a tiny inherited file-size cap that would
kill fixture creation; bound stdout/stderr using the retained validation policy. **These checks are
authored, NOT RUN on Geekom/Valinor.** No CUDA/engine rebuild is needed for this diagnostic-only change.

After those focused gates pass, mechanically stage/hash only the replay executable using the retained
transfer procedure. Preserve all worker/engine binaries and private assets. The six positional input
bindings stay exactly those of the completed replay: absolute saved capture, trusted manifest, request
asset, retained A0 raw-F32 first-logit file, numeric loopback address and current owned full-worker port.
Verify the37544692B capture SHA256
`218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208`.
No config template, weight staging or subset flag is needed. Parent dispatches one invocation after
live checks under the existing external guard and unchanged resident monitor:

```sh
timeout --signal=TERM --kill-after=10s 420s "$REPLAY" \
  "$CAPTURE" "$MANIFEST" "$REQUESTS" "$RETAINED_A0_FIRST_RAW_F32" 127.0.0.1 "$FULL_WORKER_PORT" \
  --single-group
```

The external guard targets replay only. Measure original/replay VmSwap before launch; both must be
zero for this restored-resident run and remain zero. Keep the original prelaunch ceiling fixed, never
ratchet after decreases. Require MemAvailable>=8GiB, unchanged host/cgroup OOM counters and bounded
replay RSS/output using the retained guard. On failure preserve receipts, reconcile the replay PID,
verify the original worker/guardian/monitor and cache, and do not retry or send STOP/UNLOAD.

Return exit status, source/replay/worker/config/capture hashes, the two summary lines and final owned
residency/resource receipt. Require `CPU_CAPTURE_REPLAY ok=1 resident_unchanged=1 full=960` with its
existing gates true, then `CPU_SINGLE_GROUP_REPLAY ok=1 resident_unchanged=1 records=960 cache_hit=1`;
active+zero must equal960. The first line's `grouped` counter is the older masked arithmetic, while
the second line is actual one-member group execution. If the first phase fails, no group is opened.

A pass isolates the real-input group post/finish wrapper from direct replay. It does not test two-peer
fan-out, actual subset slot maps, remote state, the live engine/helper-GPU merge or closed-loop error
amplification. It cannot clear the original failed screens or rule out accumulated rounding across
layers. No second-endpoint alias to the single-session resident is proposed.

### Prior duplicate-arena attempt — historical evidence, do not repeat

The supplied run at `6351cde18db499b0c6a28fcad01e6228098914aa` ended with a **resource stop, not
a numerical failure**. Its CPU build and2/2 CTests passed after an external RLIMIT_FSIZE wrapper
was corrected to permit synthetic fixture payloads. Full replay again passed960 replies. The first
temporary half completed58067865600B/11119blobs in66.05s; the second was incomplete when the external
guard observed422148KiB of swap in the original full worker and terminated replay at176s. Both
temporary children were reaped. No actual-subset/group numerical outcome was produced. The original
cache stayed logically complete with unchanged identity; this does not mean physical VmSwap0.

The parent supplied a completed PASS at `7c0dfce4f88856856c6d33a281824b1aae7dff1b`, whose direct
parent is `80119730422bfe7071c97420af8f5866344162c9`. CPU checks2/2 and CUDA/native-OFF links passed.
One original prompt produced one token/DONE, warm reuse of116135731200B and first logits bit-identical
to retained A0. The saved capture is37544692B, SHA256
`218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208`.
Replay returned0 in3s: windows20, records/full/grouped960, maxabs9.53674316e-7, maxrel0.333,
relativeL2 6.112e-8, entry_bits1, peer_bits1, detail_envelope1. First regrouping bit difference was
record0, prefill position0/T2/slice0/layer0/K10; the largest was record47/layer47. The complete full
arena remained unchanged: one allocation, zero failures/unloads/expirations. These are supplied
Hermes results; neither execution nor private receipt inspection occurred in this coding thread.

Those results hold baseline activations/IDs/weights fixed. They do not establish closed-loop error
propagation, subset slot-map correctness, real concurrent group behavior or remote peer state.
The failed original model screen remains failed. **Do not recapture, launch an engine or rerun GEN.**

The withdrawn sequential implementation kept one temporary half at a time, with58067865600B peak
temporary arena and60GiB additional owned-RSS budget. Even that reduced footprint did not preserve
the original worker's swap policy. Its focused fixture was authored; no desktop tests were run.
Neither duplicate-arena attempt produced an actual-subset numerical verdict. The capture format,
protocol, arithmetic and original acceptance screens remain unchanged.

### Completed real-input capture and resident replay (historical commands; do not repeat)

The parent supplies the following completed evidence, not execution by this coding thread. At
8011973 the seven-case K10/T1 diagnostic passed all 140 metric lines, including both retained
worker binaries, full/subset masks, native row/batch references and actual group/manual merging.
Only regrouping versus unsplit differed: maximum absolute errors were 1.53e-5 (dense/shifted),
3.81e-6 (sparse), and 4.58e-5 (cancellation), within the unchanged published envelope. This does
**not** clear the failed model comparison. Its synthetic activations/weights and T1 shape miss real
values and multi-token prompt batches.

Retained alignment also found matching three GEN lines, prompt tokens/hashes, GEN256 lengths,
engine binary, model/manifest/profile assets, no resume/reuse, and correct first-logit ordinals.
The first-logit files are **raw F32 with no count header**. No unintended configuration difference
was found. No intermediate CPU inputs or partials were retained. The next discriminator captures
those actual host inputs, without CUDA graph nodes, callbacks, events, copies or synchronization
changes. Production inference arithmetic and SEW1 remain unchanged.

Bind `CANDIDATE` to the exact commit containing this section. Use retained build caches, unchanged
CMake options, and no model operation until the focused gates pass. Build-target versus CTest names:

```sh
# CPU fixture host: reconfigure only to discover the new targets, preserving all cached options.
cmake -S "$SRC" -B "$CPU_BUILD"
cmake --build "$CPU_BUILD" --target cpu_capture_test cpu_capture_replay expert_tier_test --parallel 4
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^(cpu_capture_test|expert_tier_test)$'

# Engine host: outside inference, retained CUDA/native-ON cache, existing ccache and parallelism.
cmake -S "$SRC" -B "$CUDA_BUILD"
cmake --build "$CUDA_BUILD" --target strata --parallel 8

# Incremental native-OFF link check on its existing validation host/cache.
cmake --build "$NATIVE_OFF_BUILD" --target strata --parallel 2
```

`cpu_capture_test` has no socket or model. It checks multi-token format round trips, fixed capacity,
all-full-before-masked ordering, early mismatch refusal, zero-work, complete layer/token coverage,
exclusive publication, missing slices/layers, corrupt/truncated/oversize input, non-finite inputs,
wrong ownership and lifecycle/bounds refusals. These new checks are **authored, NOT RUN locally**.
Do not rebuild or replace the retained worker binary for this diagnostic. Mechanically hash/stage
the replay executable to the engine host through the existing approved transfer operation; preserve
the old binaries. The tool links only CPU libraries and does not need model files copied.

Use the existing merged config driver and its retained local warm configuration. Change only:

| Private config binding | Required value |
| --- | --- |
| engine asset | newly built candidate engine, mechanically staged/hash verified |
| worker asset/argv, manifest and model assets | exact existing full resident worker and validated assets |
| `attach_root`, `attached_worker_idle` | existing owned receipt root; true only after exclusive idle use established |
| arms | one local warm entry, `restart:false`, `requests:[0]`, no `reference` token-array gate |
| workers / loader_threads / masks / engine policy | retained 12 / 4 and unchanged measured policies |
| CPU route | original single `--remote-cpu-experts` endpoint; no group/remote GPU endpoint |
| peer configuration | remove top-level `worker_peers` and arm `cpu_peers` / `peer_bytes`; no remote control bindings |
| `capture_first_logits` | true (driver supplies its unique first-logit prefix) |
| `env.STRATA_CPU_CAPTURE` | new absolute private filename, absent along with its `.partial` sibling |
| final_worker / failure_worker | retain / retain |

Keep the original retained request asset: its first **plain GEN256** line is sent unchanged, with
no keyed GEN options, resume, session restore or warmup. Capture includes every prompt window and
the first decode window. After that first logit row, diagnostic mode emits only its first sampled
token and normal DONE; it then accepts QUIT. This is deliberately not a complete token-array or
performance run. The original requested length is retained in the capture metadata. A second GEN
is refused. No harness code or per-experiment sequencing script is needed.

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$CAPTURE_CONFIG"
python3 -B "$CAPTURE_RUN_ROOT/run.py" --run-root "$CAPTURE_RUN_ROOT"
```

Before starting, establish the existing resident is exclusively idle and complete with **forever**
residency. Both capture and replay refuse a finite TTL (avoids expiry between status and setup),
an empty cache or a cache miss. The server's existing retained-key mismatch refusal preserves the
old arena. No LOAD/UPLOAD fallback is available in these diagnostic paths. Capture allocates one
96MiB buffer plus bounded codec scratch before callbacks, writes no files in callbacks, and
publishes a private mode0600 file exclusively after the existing first-logit host copy. Its integrity
digest detects accidental corruption, not malicious edits. Never publish raw activation files.

Wait for the engine broker to finish QUIT, reap the engine/monitor, and check the existing owned
idle receipt **before** any worker control/replay connection. Never probe STATUS during an engine
session. Keep worker/pool affinity, binary and arena unchanged. Under the retained bounded output
guard, invoke exactly once (outer timeout200s, tool budget180s plus bounded frame deadlines):

```sh
timeout --signal=TERM --kill-after=10s 200s "$REPLAY" \
  "$CAPTURE" "$MANIFEST" "$REQUESTS" "$RETAINED_A0_FIRST_RAW_F32" "$WORKER_NUMERIC_HOST" "$WORKER_PORT"
```

All file bindings must be absolute. The tool validates the entire bounded format, exact original
GEN line/token binding, trusted manifest/index identity, and captured first-logit digest against
the retained A0 raw file **before contacting the worker**. A logit mismatch stops interpretation.
It then attaches using a fresh resident cache-hit session, rewrites only session/sequence IDs,
and first replays **every full-mask reply bitwise**. Any mismatch stops before masked work; report
its record index, do not treat subsequent grouping as valid. Only after that full pass does it replay
the two deterministic alternating masks on this same full arena and add active peers in configuration
order. Zero-work peers send nothing. This isolates batching/masking/reduction on actual values,
including T>1, without different slot maps, network peers or a second copy of the model.

Output is bounded summary metrics: maximum absolute/relative error (relative denominator floor1e-12),
relative L2 (norm floor1e-30), first bit-differing record and its phase/position/T/slice/layer/K.
If a numerical discrepancy exists, one record with the largest absolute error is decomposed into
at most T*K<=80 individual contributions, with two masked reconfirmations. Report entry-order and
per-peer bit identity and the unchanged `1e-4 + 4e-6*sum_abs_contributions` envelope for that record;
this is not an envelope gate over un-decomposed records. No threshold is tuned after results.
Non-finite/shape errors, full replay mismatch, decomposition mismatch or failed resident checks
fail the tool. Ordinary regrouping bit differences remain visible; a passing tool never clears
the original failed model screen or proves network/subset slot identity.

The tool disconnects normally and reconnects only after that session closes, checking the arena
remains complete/idle with unchanged bytes, loaded/verified counts, allocations, expiry and unloads.
It never sends STOP, UNLOAD, UPLOAD or LOCAL_LOAD. Preserve any failure and receipts; do not retry
uncertain operations or reload the model. Return the source/binary pins, focused test result,
capture size/hash, one first-logit comparison, replay summary/first differing record, and final
resident receipt. No complete A/B/A rerun, worker replacement or model reload is requested.

### Completed synthetic discriminator (historical commands; do not repeat)

The parent reports identical ggml source bytes at the pinned revision, matching effective compiler
flags, and a fixed retained K2/T8 fixture passing both retained worker ELFs bitwise (20,480 floats).
Those results narrow the search but do not establish equality at every shape or explain the model
failure. This follow-up changes only tests, the fixture generator and the manual build target.
Production routing, arithmetic, protocol and all numerical thresholds are unchanged.

Use the existing CPU/native-ON build on the designated fixture host. Bind the full new diagnostic
source SHA as CANDIDATE, and verify clean source as below. WORKER_A and WORKER_B are existing,
compatible, preserved worker ELFs from the failed comparison. Keep their exact hashes and the same
fixed reference build/runtime environment in the private receipt. Do not rebuild or replace those
workers. No GPU or model assets are needed.

~~~sh
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
cmake --build "$CPU_BUILD" --parallel 2 --target cpu_group_numeric_diagnostic worker_local_test
~~~

There is deliberately **no CTest entry** for the manual diagnostic: its two ELF inputs must be
explicit. Under the existing owned process-tree/RAM/output guard, run these once, in order, with a
180-second wall cap and 2 GiB RAM cap per invocation (no automatic repeat):

~~~sh
OMP_NUM_THREADS=2 "$CPU_BUILD/worker_local_test" --binary "$WORKER_A" --load-threads 4
OMP_NUM_THREADS=2 "$CPU_BUILD/cpu_group_numeric_diagnostic" --worker-a "$WORKER_A" --worker-b "$WORKER_B"
~~~

The first invocation checks the shared generator's unchanged default two-expert fixture after its
bounded extension; it does not repeat both earlier cross-binary checks. The new diagnostic constructs
32 Q8_0 experts per layer at H2560/FF640 across two layers, with only20 CPU experts per layer. Every
T1 request has10 **distinct** IDs. Full synthetic CPU arenas are208,896,000 B per worker; each split
subset is104,448,000 B. The generated fixture payload is334,233,600 B, not a model-sized asset.
Each child has a256MiB arena budget,12 inference workers and4 loader threads; only fresh loopback
port0 children can be controlled. Output is a fixed number of metric lines; child logs cap at1MiB
each. Successful completion prints child reaping and zero real-model loads. Failure cleanup signals
only the children it forked, with bounded TERM then KILL/reap. Leave the real resident worker alone.
Do not set STRATA_WORKER_FIXTURE_KEEP; normal exit removes the synthetic files.

Seven fixed cases cover dense/shifted routing, sparse CPU/GPU ownership, work on each peer alone,
zero CPU work, and arithmetic cancellation. IDs0–9 have distinct seeded weights; IDs10–31 contain
adjacent identical expert pairs so the signed cancellation case has a known small residual.
Signed weights are a deliberate numerical stress case, not a router-policy change or normalization.
The diagnostic also checks copied queued inputs, inactive-peer INFER counts, cancellation before
connection, and valid reuse afterward. It does not claim to test in-flight native-kernel cancellation.

The comparisons run in this order:

| Stage label | What a failure distinguishes |
| --- | --- |
| common-pool-vs-independent-rows | Native twelve-worker row execution differs from independent single-entry native calls |
| peerN-batch-vs-common-rows | Changing the native batch membership changes a routed row, before weighting/reduction |
| binaryA/B-full, binaryA/B-maskedN, binaryA-vs-binaryB-full | Real retained worker arithmetic/masking differs at K10/T1 or from the common reference |
| subsetA/B-vs-common-batch-mask | Actual subset ownership/setup/slot mapping differs from a full-arena masked batch |
| group-vs-manual-peers | The production group differs from independent concurrent tier posts and configuration-order addition |
| group-vs-grouped-common-reference | The combined real group differs from the same grouping over independent native rows |
| reference-grouping-vs-unsplit, group-vs-unsplit, group-vs-actual-fullA | Quantifies reference/actual regrouping against both independent and actual full-worker baselines |

All row, masked-reply and same-grouped-reference gates are **bitwise**. Only grouping versus unsplit
uses the already-published 1e-4 + 4e-6*sum_abs_contributions envelope. Metrics report maximum
absolute/relative error, relative L2 (norm floor1e-30), RMSE, first differing index and hexadecimal
float values. Relative errors can be large for the deliberately near-zero cancellation result;
the fixed absolute/scale gate remains decisive. Non-finite results fail. A bits=0 result at an
unsplit comparison is reported even when its fixed envelope passes; do not hide it as bit identity.

Return the first differing stage and all fixed-case metrics, exit status, duration/peak memory,
exact diagnostic/worker hashes, and child/file cleanup. An all-pass result narrows the failure to
inputs or behavior absent from this fixture; it does **not** clear the model's failed logit screen.
Do not tune thresholds, rerun models, add seeds or start another campaign. Broader tracing needs a
separate decision after these results. **This diagnostic is authored and NOT RUN on the desktop.**

### Historical original validation procedure — do not repeat

Parent/Hermes owns execution and source/binary pins. Keep the healthy existing full worker on02
resident until all build/fixture gates pass. Do not probe, unload, stop or replace it for preflight.
Use retained caches/assets and existing build directories. Do not reconfigure policies or install
anything. On each selected checkout establish clean exact source first:

```sh
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
```

On the existing Hermes Python controller, run the synthetic suite once under its retained bounded
output guard. It needs Linux /proc and ordinary loopback sockets, no model, GPU or native binary:

```sh
cd "$SRC/tools/validation-harness/tests"
timeout --signal=TERM --kill-after=10s 240s python3 -B -m unittest -v test_harness test_peers test_numerics
```

The new peer tests exercise real tiny children: remote-role staging/pre-START binding, alternate
loopback readiness, fast STOP reconciliation, explicit abort/reaping, lost-channel uncertainty,
bounded transport stderr, and config-only local/split selection. Existing affinity/owner tests stay.
No local synthetic test demonstrates real network throughput or full-model correctness.

On05 use the existing CPU/native-ON test build; build-target names and CTest names are different:

```sh
cmake --build "$CPU_BUILD" --parallel 4 --target strata-expert-worker cpu_worker_group_test expert_tier_test expert_worker_test worker_local_test worker_service_residency_test
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^(cpu_worker_group_test|expert_tier_test|expert_worker_test|worker_local_binary_test|worker_service_residency_test)$'
```

The group fixture covers malformed/overlap/missing ownership, one/two peers, T8/T1/T2/T1, changing
K/masks, zero-work, copied input lifetime, concurrent fan-out, reversed completion, failure draining,
non-finite/overflow refusal, setup keepalive, resident reconnect and explicit geometry/partition
refusal. It compares grouped bits exactly. Its independent unsplit gate is frozen before execution:
`abs(delta) <= 1e-4 + 4e-6 * sum(abs(weighted entry contributions))`, with max absolute/relative,
scale-relative and RMSE printed. This scale covers cancellation; the existing single-tier bitwise
gates are unchanged. Preserve any first failure; do not loosen gates to pass.

On02, outside inference, use its retained CUDA/native-ON cache, ccache and approved bulk parallel8:

```sh
cmake --build "$CUDA_BUILD" --parallel 8 --target strata strata-expert-worker worker_dispatch_test cpu_group_dispatch_test
ctest --test-dir "$CUDA_BUILD" --output-on-failure -R '^(worker_dispatch_test|worker_dispatch_host_zero_test|cpu_group_dispatch_test)$'
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$CUDA_BUILD/strata" --native-experts on
```

The new CUDA fixture uses real group/local services, the real dispatcher and mapped combine at the
production K10 contract (T1 on its two-expert synthetic pack). It asserts no local weight fallback,
zeroed worker rows, exact grouped reference and exactly one mapped contribution. Multi-token
grouping is covered by the CPU fixture, old cache/worker merge gates by the existing CUDA tests.

On03 do the small incremental native-OFF boundary check with its existing build (no model):

```sh
cmake --build "$OFF_BUILD" --parallel 2 --target strata worker_disabled_dispatch_test
ctest --test-dir "$OFF_BUILD" --output-on-failure -R '^worker_disabled_dispatch_test$'
python3 -B "$SRC/tests/worker/worker_disabled_cli_test.py" --engine "$OFF_BUILD/strata"
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$OFF_BUILD/strata" --native-experts off
```

These commands request no HIP/SYCL build. Their backend build/link validation remains NOT RUN until
the parent selects available hardware; do not claim it from CUDA or CPU results. If a named retained
build lacks the required configure options, report that exact prerequisite rather than silently
substituting a different backend. Stop after the first failed gate and keep receipts/cleanup.

## One local / split / local comparison, only after gates pass

Use candidate engine and matching candidate worker binaries for every arm: this compares ownership,
not different source releases. Retain the old baseline binaries separately. Bind one private config
to the [PEERS.md contract](../../tools/validation-harness/PEERS.md), stage mechanically once per host,
then run that static driver. No per-arm wrapper edits, automatic retries or generation warmups.
The six explicit load/warm entries yield exactly three retained GEN256 requests in each A0/B/A1.
Cold load/READY timing stays separate from inference. Loading cache conditions must be disclosed.

Engine plus RTX3060/3060Ti stay on02. Keep primary960/helper1378 placement, Q8 manifest/assets,
same retained three prompts, worker12 and loader4 on each participating host, turbo and identical
engine/worker masks/policy. Verify post-swap02 usable251.5GiB, 8x32GB,2133MT/s and05 usable94.2GiB
from parent runtime receipts. All22238 CPU experts occupy116135731200B local; B alternates them in
layer/expert order,11119 and58067865600B per peer.05 receives only that subset, never the116GB arena.
Budget each host from actual headroom; do not infer safety from installed RAM alone.

Use private actual numeric LAN bindings and the existing exact source allowlist. No new SSH/control
route, firewall/NIC/access permission or tunnel is assumed. If an authorized command channel from the
engine controller to the existing second-host guardian cannot be bound, stop before START and report
it. Do not invent addresses or change access to make the experiment run.

The current resident full bitmap cannot stand in for a split subset. After prerequisites pass,
parent names the owned worker replacement/handoff; the driver must own the retained guardian handle
before a restart. It refuses arbitrary PID adoption and occupied unowned ports. If current ownership
cannot be transferred, report the concrete handoff blocker before touching residency. Cold loads and
explicit replacement are necessary at A0/B/A1 partition changes; matching warm sessions reconnect
without reload. Final A1 stays resident;05 is stopped/reaped before A1. Keep useful weights until the
next named owned replacement, not an unrelated cleanup.

Report per-peer entries, requests, resident/loaded/verified bytes, allocation/unload/failure counters,
service and exchange timing, endpoint order, and group post-to-finish/caller-wait wall scopes. Peers
overlap, so never sum their elapsed scopes as wall time. Transport and slower peers can dominate;
equal expert counts do not establish balance. Report all nine request times and tokens, per-arm
aggregate and B versus both A0/A1 with drift disclosed. A flat or slower result closes this round;
do not add tuning arms or rerun until selecting a win.

## Numerical and correctness gates fixed before the run

Set `capture_first_logits: true` for every arm. It uses the existing
`STRATA_DUMP_FIRST_LOGITS` diagnostic: raw F32 first next-token row per matched prompt, `.0`/`.1`/`.2`.
This does **not** compare logits at every continuation position. Missing, malformed or non-finite
rows fail the comparison. Run the same source helper twice, A0 versus B and A0 versus A1:

```sh
python3 -B "$SRC/tools/validation-harness/compare_logits.py" \
  --baseline "$ENGINE_RUN_ROOT/A0-first-logits.0" "$ENGINE_RUN_ROOT/A0-first-logits.1" "$ENGINE_RUN_ROOT/A0-first-logits.2" \
  --candidate "$ENGINE_RUN_ROOT/B-first-logits.0" "$ENGINE_RUN_ROOT/B-first-logits.1" "$ENGINE_RUN_ROOT/B-first-logits.2"
python3 -B "$SRC/tools/validation-harness/compare_logits.py" \
  --baseline "$ENGINE_RUN_ROOT/A0-first-logits.0" "$ENGINE_RUN_ROOT/A0-first-logits.1" "$ENGINE_RUN_ROOT/A0-first-logits.2" \
  --candidate "$ENGINE_RUN_ROOT/A1-first-logits.0" "$ENGINE_RUN_ROOT/A1-first-logits.1" "$ENGINE_RUN_ROOT/A1-first-logits.2"
```

Frozen screening limits: max absolute logit error `<=1e-2 + 1e-4*max(abs(baseline logits))`, relative
L2 `<=1e-4`, max absolute softmax probability error `<=1e-3`, KL(A0||other) `<=1e-5`. Max relative
element error uses denominator floor1e-12 and is reported, not used as a cancellation-unstable gate.
These are prospective screening limits, not measured acceptance; no after-result threshold tuning.
Report top1 equality and full256 token arrays separately, including first divergence. Keep exact
baseline outputs. Bind the retained prompt task oracle before execution; record task correctness
separately from token equality and numeric pass. If those prompts have no task oracle, mark task
correctness UNASSESSED and leave full-model acceptance pending rather than inventing one.

Return one flat receipt with exact source/binary/config hashes, host policy/topology, commands,
pass/fail/skip/NOT RUN, first-failure output, cleanup/uncertainty, all nine measurements, numeric
metrics, token comparison and task verdict. No broad provenance framework, further campaign, merge
or deployment is authorized by this handoff.
