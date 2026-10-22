# GPU-COMBINED-V1-20261022 — prepared diagnostic, not execution permission

This additive experiment continues transport candidate 3dc8363bd0794de6e8ea283eba46bc43d5a84da9
and product base bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5. No merge/deployment or
numerical repair. Preparation/publication is approved; **the proposed new model
window still needs execution approval**. A written handoff or config is not that
approval. No local Python, scripts, syntax checks, tests, builds or fleet work
ran on Valinor/Geekom. New sources/tests below are **authored NOT RUN**.

## Evidence, scope and useful limit

Supplied GPU-FRAME-V1 result at 3dc8363b: CPU-only 05 build took 10 seconds;
correctness and loopback three-arm comparison passed without source changes.
Binary SHA-256 3ee11883ca8bf5deb354a0c45b2d112ea30ee2a430b64c30d77a16071561a739.
Split median RTT T0/T1/T2 was about 44 ms; contiguous .067/.328/.528 ms and
split+fixture NODELAY .100/.268/.478 ms. T8 was 2.018/2.089/2.030 ms respectively:
comparable performance, not demonstrated zero overhead. Segment sizes were not
measured. This strongly supports a framing/ACK interaction on loopback; neither
relay nor real inference speedup was measured. Relay was NOT RUN because prior
forwards were absent. These are supplied results, not newly inspected evidence.

Carry contiguous framing into two real passes. First measure A/B without any
capture environment variables. Then a fresh engine/GPU worker runs one A token
with capture enabled. One CPU arena remains resident across these two passes
and remains loaded after a healthy measurement completes, as explicitly requested.
Do not use a dummy relay benchmark as a prerequisite.

CUDA captures its first actual GPU-owned T=1 request, before autoregressive
divergence: exact request binding/input/ids/weights/mask, every GPU-owned row and
the already weighted partial; one selected expert's actual resident device blob;
actual Q8_1 input and hidden bytes used by its kernels. Capture happens after
the normal computation; it does not substitute a kernel or change numerical
flags. Engine captures the matching request, helper sum before network partials,
GPU and CPU partials/masks and host sum after merging. Capture is opt-in via
STRATA_GPU_CAPTURE_DIR (04) and STRATA_GPU_MERGE_CAPTURE_DIR (03), only on the
one-token pass. No whole-model dump or per-token trace framework.

Files are created once, never replaced, in new absolute 0700 owned directories:
`cuda.capture` and `merge.capture`. Each file is capped at 12 MiB. T=1/K<=10/H<=2560
limits row data to 100 KiB; only one resident blob <=8 MiB is copied, normally
5222400 bytes. Request is about 10 KiB. The worker checks the captured device
bytes against the selected assigned digest. Capture overhead (extra device copy,
serialization, fsync) belongs exclusively to this separate pass. Missing capture,
non-T=1 first boundary, wrong identity or unreadable output is a diagnostic
failure, not permission to capture later or retry inference.

On 05, replay the captured input and frozen assigned blobs through existing
ExpertWorkerService::direct_native_partial / native_gu_rows / native_quant_h /
native_down_rows. The accepted LocalSource/GpuAssetSource reads and validates
only the GPU-owned experts of this one layer, one blob at a time (<=10 *8 MiB
logical source reads), using retained source/manifest/plan bindings. Selected
source bytes must equal the actual captured device bytes. Replay reports every
expert's output difference and selected input/hidden quantization differences.
It decodes quantized formats; it does not implement another expert reference.
CPU phase calls must reproduce direct_native_partial bitwise. Separate CUDA
gate/up and float SwiGLU buffers are not reliably retained by fused kernels, so
they are explicitly unavailable; actual hidden quantization and down outputs
remain sufficient to localize before/after the down projection.

Replay also uses existing accumulate_partial on captured GPU rows and CPU-native
rows to compare weighted-once partials. It verifies the engine's same-request
GPU partial and disjoint CPU mask, and compares the observed before+GPU+CPU host
sum with the captured after sum. This observes the host merge boundary, not an
independent proof of the final device combine or every model layer. If this first
layer agrees, the mismatch is not fully localized; report that rather than
claiming all layers are correct or launching another capture.

## Proposed window and exact streams

Propose **1500 seconds / 25 minutes**, from T before any owned transport/peer
startup through capture retrieval, guarded 05 replay and engine/GPU/transport teardown.
Keep the real 45-second global cleanup reserve, five-second challenge transit,
15-second owned cleanup, original setup limits and fresh clipped leases. No
running engine/GPU extension. All-stopped build/fixture/config preparation is excluded.
The separately authorized resident CPU service and its existing watchdog/monitor
continue after this measurement deadline; they do not keep the verdict pending.
Report their ongoing lifetime and memory use separately, rather than claiming
all owned work stopped or hiding resident service time in the measurement total.

A conservative estimate retains old slow inference: CPU 427.1 + two warm GPU
setups 210.2 + A/B 302.8 + capture 15 + two engine/startup allowances 140 + gap 11
+ retrieval 10 + replay 90 + cleanup 45 = **1251.1 seconds**. About 249 seconds
remain for variance. This is a planning estimate, not measured new performance.
No assumption that contiguous framing removes relay delay.

The companion launcher binds a single fresh profile over supervisor
311561689bb4b1f8a109821601e3a6bda07067e6 with the additive
`supervisor-residency.patch`, before owners start. Its existing serial G0/G1
mode and Session/Job guards are reused. Apply the patch only to a new staged
dependency copy; no historical checkout or receipt is edited.
Internal labels below refer to new private run roots, never the retained G0/G1:

```text
G0 = PERF_AB_64:    GEN 64 9419
                   GEN 64 14773,6353
                   QUIT

G1 = CAPTURE_A_1:   GEN 1 9419
                   QUIT
```

Fresh engine/GPU maxima follow the 1500-second window. CPU is clipped during
initial guarded launch, then receives explicit independent service ownership
when its pinned, monitored listener is proven, before model setup starts.
The final retention receipt separately verifies complete idle weights after
both engines/GPU workers are reaped. GPU setup 1800000 ms remains an upper bound
clipped by the window. The existing serial 11-second gap remains. CPU starts fresh
(the reported prior arena was released), one 111597465600-byte allocation, then
reconnects with zero additional load/hash/allocation for capture. No assumption
of a pre-existing resident service; never displace an unrelated listener.
The new explicit `--residency-forever` keeps only complete verified resident
sessions without an idle expiry. `--residency-ms 0` still disables residency;
finite TTLs retain their old meaning. Combining a positive TTL with forever is
refused. Bounded frame/setup deadlines remain. CACHE_STATUS reports UINT64_MAX
for idle_remaining_ms only when a complete no-expiry arena exists.

Old history remains **6651.9 active seconds and exhausted 11 requests /704 charged
tokens**. New scope is at most **3 started requests /129 charged tokens** (64,64,1).
Charge each started request its full allowance even if interrupted. Report old,
new and combined actual time/usage separately; no reset or unused-token retries.

Require normal READY, DONE64/DONE64 for PERF and DONE1 for CAPTURE, rc0, finite
248320-float first logits, complete routing and actual owned cleanup. Preserve
formal B1/new PERF A/B centered rel-L1<=.03 and KL<=.03; prior NUMERICAL_FAIL is
not relabeled. Compare PERF A against capture A first logits using unchanged
1e-5*max(1,maxabs(reference)) drift. Numerical failure is the reason for this
diagnostic: if products/guards are healthy, continue the already planned capture
and replay even when PERF fails numerical thresholds. Product/guard/identity
failure stops dependent work. No threshold change or additional inference.

## Complete Hermes preparation and execution handoff

1. Verify the immutable candidate and GPU-COMBINED-V1.SHA256SUMS. Reuse the known
   3dc8363b framing fixture PASS; do not rerun its loopback/relay campaigns.
   Stage a new copy of the retained 31156168 supervisor package and its shared
   telemetry/guard files at the same private FILES_ROOT layout, verifying its
   existing manifest before applying the narrow experiment patch:

   ```sh
   git -C "$SUP_ROOT" apply --check --unidiff-zero "$EXPERIMENTS/supervisor-residency.patch"
   git -C "$SUP_ROOT" apply --unidiff-zero "$EXPERIMENTS/supervisor-residency.patch"
   ```

   Patched agent.py SHA-256 71801d2d6af81a3f379d4074f8ef1f500d146043340a924453796388a535498d;
   controller.py SHA-256 13d0bbc84e39761b17fe09500d763f6f7521b69fac314c1ed2472f3670a686c9.
   SUP_ROOT is FILES_ROOT/supervisor-integration-v1. This dependency is separate
   from the candidate source; do not follow its superseded old run prompt.
   Reuse accepted model, binary-base, plan and baseline receipts; no full model
   hashing, planner, sizing/listener or unrelated validation campaign.

2. Reuse each host's existing dedicated experiment source/build tree and compiler
   configuration, preserving unrelated work. Keep old validated executables and
   evidence at their original paths. Before relevant incremental builds set
   CMAKE_RUNTIME_OUTPUT_DIRECTORY to a new private candidate-bin directory; do
   not overwrite retained executables. Apply only the immutable candidate diff
   to a clean dedicated source checkout, retaining its prior Git state. If no
   suitable isolated/dedicated incremental tree exists, report that concrete
   prerequisite rather than silently doing a clean full rebuild. Existing ggml
   source/build/dependencies and compiler/backend flags must be reused; no
   dependency downloads, installs or configuration of network/security settings.
   All build/config work is all-stopped, serial (-j1), <=900 seconds per host,
   existing build-memory guard retained (cap <=8 GiB), logs <=1 MiB each.

   With SRC/BUILD/CANDIDATE_BIN bound privately per host, preserve all existing
   CMake options and reconfigure only source revision, output destination and
   test target availability:

   ```sh
   cmake -S "$SRC" -B "$BUILD" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$CANDIDATE_BIN" -DSTRATA_BUILD_TESTS=ON
   # 05, existing native-ON CPU configuration, no CUDA:
   cmake --build "$BUILD" --parallel 1 --target gpu_capture_test strata-gpu-capture-replay worker_service_residency_test
   # 03, existing accepted CUDA/architecture configuration:
   cmake --build "$BUILD" --parallel 1 --target strata
   # 04, existing accepted CUDA/architecture configuration:
   cmake --build "$BUILD" --parallel 1 --target strata-cuda-expert-worker
   ```

   Run only the new focused checks once on 05:

   ```sh
   "$CANDIDATE_BIN/gpu_capture_test"
   "$CANDIDATE_BIN/worker_service_residency_test"
   "$PYTHON" -B -I "$EXPERIMENTS/test_combined_profile.py" --supervisor-root "$SUP_ROOT"
   ```

   Require rc0, no fixture failures/errors/skips, markers
   GPU_CAPTURE_FIXTURES_PASS_V1 models=0 cuda=0 and
   CPU_SERVICE_RESIDENCY_FIXTURES_V1 failures=0 models=0 cuda=0 and
   GPU_COMBINED_PROFILE_PASS_V1 tests=9 workloads=0. C++ fixtures: 512 MiB AS,
   10 CPU/30 wall seconds each, captured output <=1 MiB; synthetic files <=32 MiB in a fresh temp
   directory. Profile fixture: 128 MiB AS, 5 CPU/10 wall seconds, 1 MiB output.
   C++ per-file RLIMIT_FSIZE is 32 MiB so it permits the synthetic shard/capture
   files; bound stdout/stderr separately with the existing fixture runner.
   Stop on relevant build/fixture failure. Ordinary in-scope launcher/fixture
   corrections may be made with exact source/hash distinction and focused rerun;
   do not rebuild/retest unrelated campaigns or conceal failures.

3. **Before executing the new window, confirm separate execution approval.**
   Until then stop after preparation; do not set execution_approved=true or start
   products/transports. Upon approval, set these fields in controller and each
   private agent/replay config:

   ```text
   diagnostic: GPU-COMBINED-V1-20261022
   supervisor_commit: 311561689bb4b1f8a109821601e3a6bda07067e6
   diagnostic_revision: the immutable candidate commit
   approved_window_s: 1500
   execution_approved: true
   gpu_completion: true
   retain: true                      # controller
   service_residency: true            # controller and CPU agent only
   retention_approved: true           # CPU agent only; false for engine/GPU
   ```

   Legacy implementation=bf51f09e in these supervisor configs identifies the
   frozen base/retained baseline contract, not a claim that new binaries are old.
   Record diagnostic_revision plus each new source/build/binary hash; bind 03
   BIN_ENGINE_A and 04 BIN_GPU_B/binary_sha256 to newly built candidate outputs.
   Bind 05 BIN_CPU_C/binary_sha256 to the newly built candidate strata-expert-worker
   (required for the explicit no-expiry option), preserving the old validated ELF.
   Baseline object still
   binds the actual seven retained B0/B1/G0 files and their accepted hashes from
   archives 35a57b3a... and 511471cd... using the original full hashes in the
   pinned 31156168 contract. Retain completed G1 evidence (bundle
   b993dbef8e26846c4b6623754c21efbf4839aca317c244a99bac5b781c85af75)
   separately for same-mode comparisons. Do not manufacture old supervisor PASS.

4. Reuse the exact retained private role mappings and approved transport argv.
   The established topology is 03 loopback -> reverse SSH -> controller loopback
   -> local SSH forward ->04/05 loopback. Copy the same four forwards argv arrays
   and role-peer SSH prefixes from the accepted controller config; do not infer
   new hostnames, routes, options or ports. Recreating only those approved owned
   forwards belongs inside the new approved window. If exact bindings/authority
   are unavailable, report the missing target/permission before execution.
   Replace each peer's remote agent invocation with:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/combined_diagnostic.py" --supervisor-root "$SUP_ROOT" --agent-config "$ROLE_CONFIG"
   ```

   No private role details may be published. New run_root values must not exist;
   capture directories are created beneath 03/04 run_root during CAPTURE only.
   CACHE_B remains the same complete exclusively owned real directory. Normal
   configured binary/plan/namespace checks apply; no new provenance campaign.
   Keep primary960/helper2247/CPU21369/uncovered18080/remote3289, CPU budget106428
   MiB, payload111597465600 B and plan92156 B SHA
   a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578.
   All original host/cgroup/RSS/anonymous/swap/identity/watchdog/error protections
   and strict local admission1152/512 MiB remain; valid local runtime crossings
   remain warning-only. Remote reserve4096 MiB, cache18432 MiB and existing disk
   guards stay. Both passes require validated reuse W=17176473600 with zero new
   product source-weight reads/network/persist. Diagnostic copying and replay
   source reads are separately accounted, never mislabeled zero I/O.

5. Record controller T immediately before owned startup, then run:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/combined_diagnostic.py" --supervisor-root "$SUP_ROOT" --controller-config "$CTRL_CONFIG"
   ```

   PERF has both capture variables absent. CAPTURE enables them only in new 03/04
   jobs. The coordinator stops/reaps engines/GPU workers, commits healthy CPU
   idle residency and returns its receipt promptly. The CPU's same independent
   watchdog and monitor continue on05; private resident.json records all three
   process identities and the original admission/baseline. No persistent forward
   is needed for that supervision. Owned forwards close normally. Preserve complete
   engine/GPU cleanup/baseline return receipts and a fresh CPU live-status receipt.
   Extract actual DONE
   prompt/decode milliseconds, CPU worker_req_wall deltas and CUDA request_ms /
   request-count deltas for each PERF request; final worker service_ns is a
   separate cumulative executor scope. Do not sum overlapping scopes or compare
   old CPU-tier wall against new generation wall. Report actual relay topology,
   total/mean real GPU exchange time and end-to-end tokens/s. Capture/replay are
   excluded from these performance comparisons and included in window elapsed.

6. With engines/GPU workers stopped and the CPU service still protected, collect only 04 run_root/capture/cuda.capture
   and 03 run_root/capture/merge.capture into one new private 0700 directory on05.
   Use retained role SSH/file-transfer bindings through the same controller;
   record small-file SHA-256 before/after transfer. Do not transfer model/cache
   directories or expose captures publicly. Bind 05 replay config role=ROLE_CPU_C,
   new run_root, normal namespace receipts, binary_sha256 for the new replayer,
   and exactly five absolute replay_argv strings:

   ```text
   [strata-gpu-capture-replay, capture_directory_on05, frozen_plan_on05,
    retained_LOCAL_ROOT_C, retained_MANIFEST_C]
   ```

   On the same controller clock compute REMAINING_S = T+1500-45-now immediately
   before dispatch, then invoke through the retained 05 role command. Bound that
   owned transport invocation by the same original deadline; do not queue a stale
   duration for later execution. Allow at most the existing five-second dispatch
   margin; if establishing that transport takes longer, cancel before replay and
   obtain a fresh remaining duration. Never send a raw remote clock cutoff:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/combined_diagnostic.py" --supervisor-root "$SUP_ROOT" --replay-config "$REPLAY_CONFIG" --remaining-s "$REMAINING_S"
   ```

   The adapter reuses Session/Job, identity pins, parent death and true-parent
   cleanup. A separate small ROLE_REPLAY_C policy admits 8.5 GiB available and
   preserves 8 GiB headroom, original swap/event/cgroup checks and a 512 MiB
   RSS/anonymous limit. This avoids asking for a second full-arena admission while
   the original CPU arena and its unchanged stronger guard remain live. The replay
   cap is <=90 seconds and the fresh remaining duration, anchored before local
   config/import/admission/hash preflight; those costs cannot renew the window.
   It refuses if no safe interval remains. The replayer itself
   caps AS512 MiB, CPU60/wall90 seconds, output1 MiB. Require rc0, complete cleanup
   and GPU_CAPTURE_REPLAY_COMPLETE_V1 in its workload.out. Completion means the
   diagnostic ran; its printed differences are not a numerical acceptance PASS.
   If the original window has no safe interval, stop/report replay NOT RUN;
   no extension, counter reset or inference retry.

7. Leave exactly the healthy CPU worker, its existing monitor and detached
   guardian resident on05. Once guarded listener ownership is granted, controller
   EOF, measurement cancellation or lease expiry detaches this CPU owner rather
   than killing a healthy arena. Resource/identity/coverage faults still clean up;
   an explicit CPU STOP or SIGTERM still stops it. Partial/failed setup never
   promotes incomplete weights; status must say empty if they were released.
   There is no idle timeout or inherited trial/two-hour cap after the explicit
   service ownership transition. Original
   host/cgroup/RSS/swap/event/identity/coverage protections continue with their
   original admission baseline. Memory pressure, lost coverage, process failure,
   namespace/identity change, or unreadable/full storage still triggers owned
   cleanup. No restart loop or production service installation is added.
   Numerical acceptance failure alone does not unload this healthy arena.

   The resource and watchdog journals retain two generations of <=64 MiB each,
   with private ownership checks; the original admission/baseline is also kept in
   resident.json. Rotation prevents a hidden log-cap lifetime expiry while keeping
   storage bounded. The opt-in worker detail log has an 8 MiB lifetime quota:
   after one quota marker it suppresses further detail lines while keeping live
   status counters and the rotating watchdog/resource journals. Main shutdown
   summaries remain; quota exhaustion does not unload the arena. Default/finite
   modes retain their existing logging. Missing/stale status is a fault/unknown,
   never proof of residency. Query worker cache status only between engine sessions;
   its existing single-client service may not answer a second connection while
   inference is active. A busy/timeout result must not be mislabeled empty/dead.
   Use the retained authorized05 SSH prefix to execute these commands on05's
   loopback; no permanently open controller tunnel or new listener is needed:

   ```sh
   # Honest status: verifies watchdog/worker/monitor identities and <4s coverage,
   # then queries the worker. Use the same private CPU agent config after the run.
   "$PYTHON" -B -I "$EXPERIMENTS/combined_diagnostic.py" --supervisor-root "$SUP_ROOT" --resident-status "$CPU_CONFIG"
   # Explicit unload: frees weights, keeps the guarded empty worker available.
   "$PYTHON" -B -I "$SRC/tools/worker_control.py" unload --host 127.0.0.1 --port "$CPU_PORT_C"
   # Explicit stop: worker releases arena/exits; its true-parent guardian reaps
   # worker and monitor, writes WATCHDOG_FINAL_V1 and exits. Verify that receipt.
   "$PYTHON" -B -I "$SRC/tools/worker_control.py" stop --host 127.0.0.1 --port "$CPU_PORT_C"
   ```

   For a later separately authorized guarded engine job, reconnect to the same
   retained ENDPOINT_CPU_C_ON_A using the frozen `--remote-cpu-experts` binding,
   `--remote-cpu-local-manifest "$MANIFEST_A" --remote-cpu-resident` and the same
   model/partition. Recreate only the already approved forwards when needed.
   Existing SETUP/READY gives a fresh session; require cache-hit/reused111597465600,
   zero new load/hash/network/allocation before claiming reuse. This handoff grants
   no extra GEN request. The serial trial controller always starts its CPU role:
   **do not rerun it as a reconnect command or start another CPU worker**. A later
   engine owner must attach the retained service; no reconnect orchestration
   redesign or persistent network access is included here. After explicit unload,
   an authorized setup loads the same partition again. After explicit stop, restart
   requires normal fresh guarded admission. Never signal a reused numeric PID.

8. Return a single concise report: actual run/build/source hashes and focused
   outcomes; uninstrumented PERF timing/numerics versus retained B1/G1; capture
   layer/expert/sequence and same-request bindings; per-expert, quantization,
   weighted-partial and host-merge differences; unavailable intermediates and
   unresolved layers; product cache and CPU reuse; separate replay/capture I/O,
   old/new/combined measurement time and usage plus ongoing resident-service age;
   errors, actual engine/GPU reaping/baseline restoration, live CPU/guardian/monitor
   identities and status/reconnect/unload/stop bindings, and private evidence hashes.
   Preserve prior NUMERICAL_FAIL and failures=1 attribution uncertainty. Stop all
   bounded diagnostic/transport work; retain the healthy protected CPU arena,
   full cache-B and unrelated services. No merge, deployment, threshold tuning, extra model
   request, activation campaign or another supervisor iteration.
