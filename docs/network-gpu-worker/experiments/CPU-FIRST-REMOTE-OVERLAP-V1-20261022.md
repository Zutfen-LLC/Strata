# CPU-FIRST-REMOTE-OVERLAP-V1-20261022

Approved additive candidate from e122215a28791062259f391a810d232e10c542d7 on
experiment/gpu-first-layer-v1. No merge or deployment. This author performed only
static source/text/git/hash/publication work. All new builds, fixtures and model
validation are **NOT RUN** here; Hermes executes the bounded continuation below.

## Change and evidence

Supplied completed practical comparison: G reproduced the five B answers exactly,
four strict passes and the same substantively correct fenced-code format failure.
B0 batch wall was 65.66 s, G0 95.24 s. The reconciled G0 GPU synchronous client
chain was 35.775 s over 5502 RPCs; its CPU chain was about 6.98 s lighter. Prompt
time increased 21.11 s and decode 8.52 s. These observations support the source's
serialization issue, not a nonoverlapping wall-time partition or promised speedup.

expert_source.cpp previously completed synchronous GpuWorkerTier::post before
ExpertWorkerTier::post started its existing CPU thread. CPU post already copies
activations, IDs, masks and weights into its own request. The candidate submits
that CPU work before the synchronous GPU post, using a small shared submission
function tested directly by the standalone fixture. It adds no production thread,
socket abstraction, timer or scheduler. Ownership validation/masks and request
contents stay unchanged; success retains CPU-finish then GPU-finish, and the
existing deterministic helper/GPU/CPU merge arithmetic stays untouched.

If GPU post returns false or throws after CPU submission, the dispatch drains the
posted CPU request before returning/rethrowing. A CPU-drain failure is logged while
the original GPU failure remains primary. Neither network partial reaches the
merge on this path. Existing socket deadlines and outer guards bound failure
handling; no running deadline or join budget is extended. Other existing early
failure paths retain the tier's own leftover-request/destructor joins. CPU post
refusal prevents GPU submission. No kernels, quantization, routes, wire bytes or
production inference flags change. The 04 GPU worker and 05 CPU worker do not
depend on this dispatcher/header and must not be rebuilt or replaced.

Static review checked the failure paths and existing finish/merge order. A report
identity gap was corrected: standalone reporting requires the candidate engine
role, valid source revision and exact candidate ELF pin. Static review is not an
executed test or evidence of performance improvement.

## Focused preparation and gates

Use existing private roles, donor sources/dependencies and dedicated incremental
trees. Preserve old ELFs, old G/B evidence, the full 04 cache and the healthy CPU
arena/monitor/guardian. Do not execute workloads on Valinor/Geekom, clean a build,
download/install dependencies, change network/security settings or credentials,
create a new route, transfer/refill/hash the full model, replan, merge or deploy.

1. Obtain this immutable revision and verify CPU-FIRST-REMOTE-OVERLAP-V1.SHA256SUMS.
   Source production delta from e122215a is limited to expert_source.cpp and
   cpu_first_remote_post.hpp. CMake adds only the standalone fixture target;
   other changes are focused fixtures, candidate binding/reporting and procedure.
   Retain actual source revision, compiler/cache settings and new ELF hashes in
   private build receipts. Keep existing runtime/numerical flags and ggml cache.

2. On 05 use the accepted CPU-only incremental source/build tree and a new owned
   BIN05 output directory. Build only the standard-library fixture, serially:

   ```sh
   cmake -S "$SRC05" -B "$BUILD05" -DSTRATA_BUILD_TESTS=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN05"
   cmake --build "$BUILD05" --parallel 1 --target cpu_first_remote_post_test
   (
     ulimit -v 131072
     ulimit -t 5
     timeout --signal=TERM --kill-after=5s 10s "$BIN05/cpu_first_remote_post_test"
   )
   ```

   Build <=120 wall seconds / 512 MiB RSS/anonymous, conservative host headroom
   under existing 05 controls; no native-worker/engine target is built. Fixture
   <=128 MiB AS / 10 wall seconds / two threads / 1 MiB output. Require rc 0 and
   `CPU_FIRST_REMOTE_POST_V1 cases=4 failures=0 models=0 cuda=0`.
   It invokes the exact production submission function with controlled asynchronous
   CPU and synchronous GPU endpoints: in-flight CPU before GPU completion,
   copied-buffer ownership, success, GPU false/throw drains, and CPU-post refusal.
   It uses no model, socket or GPU. The fixture deliberately compiles without
   STRATA_NATIVE_EXPERTS and links only Threads; its target is available with
   native-OFF too. Actual changed worker dispatch remains inside the native and
   network-GPU guards, so no new disabled-build worker symbol dependency is added.
   Do not spend a separate full native-OFF engine campaign on unchanged guards.

3. On 03 reuse the accepted CUDA/native-ON incremental source/build tree and
   compiler/ggml donor cache. Stage this exact source in that dedicated clean
   experiment checkout; preserve other work and prior ELFs. Use a new private
   CANDIDATE_BIN. No clean rebuild or substitute dependency fetch if prerequisites
   are absent: report the concrete target/prerequisite instead.

   ```sh
   cmake -S "$SRC03" -B "$BUILD03" -DSTRATA_BUILD_TESTS=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$CANDIDATE_BIN"
   cmake --build "$BUILD03" --parallel 1 --target strata gpu_worker_dispatch_test
   ```

   Build <=900 wall seconds / 8 GiB build RSS/anonymous, one job, logs <=1 MiB.
   Keep backend, architecture, optimization and native options from the retained
   cache; do not silently reconfigure a different backend. Do not build a worker
   executable. Hash the new candidate strata ELF and preserve the old accepted ELF.

   Run the updated real dispatcher integration fixture on 03's approved engine
   GPU mapping under its existing fixture/resource guard, with no product running:

   ```sh
   CUDA_VISIBLE_DEVICES="$GPU_SELECT_A" timeout --signal=TERM --kill-after=5s 180s "$CANDIDATE_BIN/gpu_worker_dispatch_test"
   ```

   Require rc 0 and `CPU_FIRST_DISPATCH_V1 failures=0 (real dispatcher, CPU tier, network CUDA and merge)`.
   The fixture uses synthetic bounded Q8_0 weights, real CPU service/client,
   network CUDA executor and RemoteExpertOpt combine on the local test host;
   primary-resident rows are emulated as before. It verifies same-owner merged
   results bitwise, then GPU loss after CPU submission: exactly one original CPU
   share, joined request, zero remote sum and untouched result rows. No fallback
   or partial commit. Its original numerical checks remain. Synthetic source is
   about 63 MiB, assigned CUDA weights about 10.5 MiB plus scratch/context; report
   actual runtime/context overhead. Keep existing strict admission and GPU/host
   guards, <=2 GiB fixture RSS/anonymous and <=1 MiB logs. Do not impose a tiny
   CUDA address-space limit that prevents driver initialization. Confirm only
   fixture-owned cleanup and GPU baseline before product startup.

4. Reuse the e122215a staged absolute supervisor package including shared planning
   modules and its three published patches. The controller postimage remains
   fa89b9554442a71bc54b7d1098d31831b76a19ba0a15c93cfc5cda9c3a49cce3;
   agent 4307f89265617cf18c6ac747b1b19193457172c93dfaa902db81e011ff924849,
   policy d3b1de8805aef47041df1b81726cec6e6c7623ef117df22c1420695bec660cb8.
   Do not modify the live CPU guardian's historical files. On 05 run only the new
   candidate-binding/report mocks, not prior passing suites:

   ```sh
   EXPERIMENTS="$SRC05/docs/network-gpu-worker/experiments"
   "$PYTHON" -B -I "$EXPERIMENTS/test_cpu_first_overlap.py" --supervisor-root "$SUP_ROOT"
   ```

   Require rc 0 and `CPU_FIRST_PROFILE_V1 tests=4 workloads=0`, no failures/errors/
   skips; internal caps 256 MiB AS / 5 CPU / 10 wall seconds, no subprocess/socket/
   PID signaling, output <=1 MiB. All preparation/gates above are separate from
   the new model window and consume zero GEN requests. On focused PASS proceed
   automatically; this is not another approval gate. Ordinary in-scope launcher
   fixes are allowed with exact differences and failed attempts retained.

## Automatic seven-request validation

The owner explicitly grants **one fresh 1500-second window**, start before
controller/relay startup through cleanup, with the existing **45-second reserve**.
Use the retained exact plan SHA
1c81fa79b9a26f66b0a4f7b2e6eae92858c82f47de827f08751b312948a21792.
No B rerun, new prompt, retokenization or hidden warmup. G0 is the five GEN32
tasks; G1 is the two supplied-answer GEN1 probes, each phase ends QUIT. At most
**7 started requests / 162 generated tokens**, charge interrupted starts in full.
Historical actual usage is **31 / 941**; new maxima are **38 / 1103**. This is a
new explicit grant, not a reset of the exhausted earlier request ceiling.

5. Resolve original B and accepted G artifacts/config snapshots from retained
   private receipts, verify bounded copies against those receipts, and keep their
   separate roots. Original B is e0d11fc7; accepted G is e122215a, archive SHA
   740ef5f45b98947d10a43845156e945d4c0616d1d4fb90ad17a58e28b167d764.
   The profile rechecks B/prior G completeness and exact plan/config association
   before transport startup. Do not regenerate token IDs or replace old evidence.

6. Revalidate the original resident CPU service by its live identity/resource
   receipts, complete 111597465600-byte / 21369-expert partition and UINT64_MAX
   idle TTL. Historical numeric PIDs are not signal authority. No duplicate,
   restart, unload, replacement or cold fallback. Keep 04 cache 17176473600 B,
   placement 960 / 2247 / 21369 / uncovered 18080 / remote 3289, CPU budget
   106428 MiB, and the same 92156-byte plan SHA
   a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578.
   Preserve strict admission, runtime local VRAM-warning policy and substantive
   host/cgroup/swap/RSS/GPU/disk/identity/cleanup guards. Refresh stale admission
   receipts through the existing authorized channel only, without touching the
   resident owner. No new privilege or network authority is granted.

7. Use fresh configs/run roots with diagnostic=CPU-FIRST-REMOTE-OVERLAP-V1-20261022,
   diagnostic_revision=<this immutable commit>, supervisor_commit=311561689bb4b1f8a109821601e3a6bda07067e6,
   approved_window_s=1500, execution_approved=true, identical retained plan path/SHA,
   and these explicit bindings on all new configs:

   ```text
   retained_engine_sha256 = old accepted B/G engine ELF SHA
   candidate_engine_sha256 = new 03 candidate strata ELF SHA
   candidate_base_revision = e122215a28791062259f391a810d232e10c542d7
   candidate_source_revision = this immutable commit, equal to diagnostic_revision
   ```

   ROLE_ENGINE_A binary_sha256 and BIN_ENGINE_A bind the NEW candidate; the agent
   still verifies that exact ELF. The old accepted pin remains distinct and is
   checked against retained B/G configs. Only this diagnostic identifier permits
   that reviewed scheduling candidate; legacy G continuation remains strict about
   the accepted hash. Record source/build receipt and actual pins, not a claim
   that the new executable is byte-identical to the base bf51f09e implementation.
   Keep implementation=bf51f09e as the inherited base identifier, gpu_completion=true,
   retention_approved=false and service_residency=false. RUN_A equals new run_root.
   CPU peer remains the existing observer, referencing original resident_config.
   04 GPU worker binary and 05 resident binary stay pinned to their accepted ELFs.

   Controller uses the e122 config layout, new controller_evidence directory,
   retained_b={root,engine_config}, plus prior_g={root,engine_config} pointing to
   accepted G artifacts/config local to the controller. Keep its existing three
   peer roles and at most four exact approved temporary controller-mediated
   loopback forwards. Replace only profile/source paths and engine candidate pin:

   ```sh
   # 03/04 peer command, with host-local EXPERIMENTS/SUP_ROOT/ROLE_CONFIG:
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --agent-config "$ROLE_CONFIG"
   # 05 peer, never a CPU-start agent:
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --observe-config "$OBSERVER_CONFIG"
   # Existing approved controller:
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --controller-config "$CTRL_CONFIG" >"$EXECUTION_RESULT"
   ```

   Preserve the single monotonic execution-clock receipt, clipped caps and failed
   FINAL/stderr/transport-exit diagnostics. No running extensions, reset or extra
   requests. Numerical/quality outcomes do not cancel a healthy prescribed stream;
   real product/resource/ownership faults still stop safely. Both warm GPU setups
   validate/load the cache separately. Cache mtimes do not prove transfer/read zero.

8. Stop owned engines/GPU worker/observers/temporary relays, restore GPU baselines,
   and retain healthy guarded CPU residency and full 04 cache. Report by host/role;
   do not claim all Strata processes/listeners disappeared. Retrieve bounded new
   outputs, clocks, two probe logits/logpos files, counters and cleanup diagnostics.
   On 05 use the new comparison wrapper (decode only; no retokenization):

   ```sh
   (
     ulimit -v 1048576
     ulimit -t 30
     timeout --signal=TERM --kill-after=5s 60s "$PYTHON" -B -I "$EXPERIMENTS/cpu_first_overlap.py" --plan "$PLAN" --pack "$PACK_META" --b-root "$B_ROOT" --b-engine-config "$B_ENGINE_CONFIG" --prior-g-root "$PRIOR_G_ROOT" --prior-g-engine-config "$PRIOR_G_ENGINE_CONFIG" --g-root "$G_ROOT" --candidate-engine-config "$CANDIDATE_ENGINE_CONFIG" >"$COMPARISON_RESULT"
   )
   ```

   Compare candidate G with both retained B and prior G; label separate-run timing.
   Preserve actual decoded text, strict/format/substantive observations and cap
   exhaustion. Require same-mode output IDs, fixed-input probe logits and reference
   likelihood unchanged; any drift is SCHEDULING_REGRESSION, investigate later,
   without another request or threshold relaxation. Historical NUMERICAL_FAIL and
   .03 centered rel-L1 / .03 KL screening stay unchanged. State measured prompt,
   decode and warm batch improvement/regression; do not promise recovered time.

   Keep setup, steady exchange and executor counters distinct. CPU request_ms/
   worker_req_wall is post-to-finish wall and now includes overlapping GPU wait;
   it is not pure CPU service time. Never add overlapping CPU/GPU times as a wall
   partition. Use cumulative counter deltas. failures=2 is consistent with two
   ordinary session EOFs, not individually proven; verified=4W is two local
   read/hash passes per setup across two sessions, not four VRAM loads or zero
   local reads. service_ns includes executor host/transfers/kernels/sync/reduction,
   not pure kernels. Require product counters for zero new source transfer/persist
   or mark unmeasured. Report new elapsed, historical ledger, cleanup and ongoing
   residency honestly; preserve failed evidence and do not start another campaign.
