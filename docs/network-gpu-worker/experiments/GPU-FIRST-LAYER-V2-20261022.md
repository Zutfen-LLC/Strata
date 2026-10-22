# GPU-FIRST-LAYER-V2-20261022 — prepared repair, execution not granted

Additive correction to failed candidate 37b873fab6579b682e0e01217241f1b7d01524e3
on experiment/gpu-first-layer-v1. Preserve that commit and its failed evidence.
The arithmetic, trace locations, frozen placement and numerical thresholds are
unchanged. Preparation/publication are approved; the parent must separately
approve this proposed execution. All new code, builds and tests are **NOT RUN**
by this author. Desktop work was static source/text/git/hash/publication only.

## Evidence, source finding and limits

Supplied V1 result: builds and CPU fixtures passed. B0 attached to the existing
111597465600-byte arena in 0.004 s and reached READY, then its first traced verify
window stalled. GPU rang 1 / served 0 / plan A 0 / copies B 0, the pool claimed
0 of 0 jobs, no CPU inference traffic was reported, and the trace directory was
empty. The 60-second engine stall watchdog aborted rc -6. G0 and replay were not
run. This is supplied evidence, not a newly inspected fleet report.

The source establishes a launcher omission. Verifier::run already uses
HostGraphLaunch when STRATA_VERIFY_ASYNC_LAUNCH=1. Its documented purpose is to
let the CPU service thread serve doorbells while an instrumented cudaGraphLaunch
blocks until graph completion. The V1 profile enabled the trace but omitted
this switch; the inherited commands module builds an explicit environment, so
shell inheritance cannot repair the omission. The added D2H nodes are downstream
of graph waits for host plan/CPU rows. A blocking launch/query on the same host
thread would prevent those flags from being raised.

That explains a source-supported dependency risk; it does **not** prove the exact
call that blocked in V1. Neither a specific D2H node nor the mapped-mask copy is
established as the cause. Do not describe CUDA kernels or quantization as fixed.

The repair sets STRATA_VERIFY_ASYNC_LAUNCH=1 and STRATA_VERIFY_TRACE=1 for both
traced engines, using the existing supported launch path. Opted-in trace init
now refuses a missing async switch before allocating trace buffers. The existing
bounded handshake ring gains LAUNCH-BEGIN, QUERY-BEGIN and QUERY-DONE alongside
WINDOW, LAUNCHED, RANG and SERVED. On another stall, preserve those events and the
existing GPU breadcrumbs; do not infer that a downstream copy executed solely
because the first doorbell rang.

No trace copy, final-device-combine location, routing, mask, arithmetic, kernel,
quantization, scheduling flag other than this diagnostic launch switch, or
ownership/resource policy is changed. Default untraced launch behavior remains
unchanged. This is a candidate correction until the real CUDA gate and product
request execute. A successful small gate establishes this seam, not whole-model
correctness.

## Tiny CUDA gate before any model request

The existing host_graph_launch_test gains --layer-trace. It captures a real CUDA
graph with four sequential host/device handshakes, H=2560 and K=10, and the same
five trace-copy shapes/directions: per-entry rows, shared output, pre-add output,
post-add output and the mapped ownership-mask alias. The CPU supplies each
layer's mask and plan/copy/served flags only after its GPU doorbell. All copied
values and all four serviced layers are checked exactly. Dummy kernels replace
expert arithmetic; no model, worker service, arena request or network is used.

--blocking-launch is fixture-only: the test translation unit wraps the real
cudaGraphLaunch with cudaStreamSynchronize before returning. This forces the
documented blocking-launch ordering without relying on a sanitizer/driver to
trigger it. Production HostGraphLaunch is unchanged. The synchronous negative
control must be released by the eight-second watchdog and fail with the exact
CPU-service assertion. The threaded arm must complete without that release.
Both use the actual CUDA graph and D2H operations. A second positive arm omits
the forced blocking wrapper, exercising the installed runtime's normal launch
and query behavior too. Both positives call HostGraphLaunch::query before
doorbell service and while waiting for the next ring, as the verifier does.
No timing threshold is used to declare a performance improvement.

Explicit buffers are 122880 device bytes, 532640 pinned trace bytes and one small
mapped control record; CUDA context/driver overhead is additional and must be
reported. No whole-model or large synthetic allocation is introduced. Run under
the existing 03 fixture admission/resource controls, with the external 20-second
per-command timeout below; verify only fixture-owned process/context cleanup.
Do not touch the resident 05 worker or unrelated GPU processes.

## One complete proposed continuation

Propose at most **90 seconds for the new focused gates**, then, only on PASS, the
existing **600-second attached B0/G0/replay window**, including its real 45-second
cleanup reserve. Thus new diagnostic runtime is bounded by 690 seconds plus the
separately bounded incremental build. The controller's 600-second clock begins
before its own transports/observer startup; no running lease or deadline is
extended. The gate uses no GEN allowance. No model retry is included.

1. After the parent grants execution, obtain this immutable revision and verify
   GPU-FIRST-LAYER-V2.SHA256SUMS from the repository root. V1 manifests/reports are
   historical and belong to their original immutable revisions. Use existing
   ROLE_CONTROLLER/private role receipts and the same temporary authorized
   controller-mediated relays. No desktop execution, new route, credential,
   network/security setting, persistent tunnel, dependency download, merge or
   deployment. Preserve 04 cache-B and all prior failed/successful evidence.

   Keep the existing admission-receipt age rule. The parent reports that the old
   05 administrator receipt reaches 24 hours at 2026-10-22 12:14:41 UTC. At actual
   admission, if a required receipt is stale, refresh that receipt through the
   existing approved administrator channel and bind it in the new admission
   config. Refreshing the receipt is distinct from restarting or modifying the
   resident service: do not rewrite its ownership receipt, original baseline,
   guardian files or namespace identity to make a check pass. Recheck the same
   host/boot/namespaces through the existing procedure; no new privilege or
   security setting is authorized. If that approved channel is unavailable,
   report the exact pending admission target and stop before dependent work.
   Do not waive age or silently substitute service health for admission evidence.

2. Reuse the V1 staged absolute supervisor root at
   311561689bb4b1f8a109821601e3a6bda07067e6 with supervisor-residency.patch and
   first-layer-supervisor.patch already applied. Do not modify the live CPU
   guardian's historical files. If a fresh stage is needed, apply the same two
   immutable patches, in that order, with git apply --check --unidiff-zero then
   git apply --unidiff-zero. Expected postimage SHA-256:

   ```text
   agent.py      4307f89265617cf18c6ac747b1b19193457172c93dfaa902db81e011ff924849
   policy.py     d3b1de8805aef47041df1b81726cec6e6c7623ef117df22c1420695bec660cb8
   controller.py 13d0bbc84e39761b17fe09500d763f6f7521b69fac314c1ed2472f3670a686c9
   ```

3. On 03, reuse the accepted CUDA incremental tree/dependencies and a new private
   CANDIDATE_BIN. Preserve prior ELFs. Serial build, at most 900 seconds and 8 GiB
   build memory under existing controls; logs at most 1 MiB each. No clean rebuild
   or dependency fetch if that tree is unavailable: report the prerequisite.

   ```sh
   cmake -S "$SRC" -B "$BUILD" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$CANDIDATE_BIN" -DSTRATA_BUILD_TESTS=ON
   cmake --build "$BUILD" --parallel 1 --target strata host_graph_launch_test
   ```

   No new 04/05 binary behavior changed. Reuse the V1 04 GPU worker and 05
   strata-layer-diagnose built at 37b873fa, binding their real retained ELF hashes
   in the private configs. Never rebuild, replace or restart the live 05 worker.
   Reuse the supplied passed CPU serialization/residency fixtures; do not rerun
   old warning/periodic/framing campaigns.

4. Before model startup, run only these new/changed checks. On 05:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/test_first_layer_profile.py" --supervisor-root "$SUP_ROOT"
   ```

   Require rc 0 and FIRST_LAYER_PROFILE_PASS_V2 tests=7 workloads=0, no failures,
   errors or skips. Its caps remain 128 MiB AS / 5 CPU seconds / 10 wall seconds,
   no process, socket or fleet workload creation, output at most 1 MiB.

   On the actual engine GPU on 03, using the existing fixture guard and no other
   product started, use the same approved CUDA_VISIBLE_DEVICES/device-0 mapping
   as ROLE_ENGINE_A. Run the negative control first, then both positive arms (the
   explicit return-code capture also works under a fail-fast shell):

   ```sh
   NEGATIVE_RC=0
   timeout --signal=TERM --kill-after=5s 20s "$CANDIDATE_BIN/host_graph_launch_test" --layer-trace --blocking-launch --synchronous >"$NEGATIVE_LOG" 2>&1 || NEGATIVE_RC=$?
   test "$NEGATIVE_RC" -eq 1
   rg -F -x 'FAIL: traced graph CPU service before launch completion' "$NEGATIVE_LOG"
   timeout --signal=TERM --kill-after=5s 20s "$CANDIDATE_BIN/host_graph_launch_test" --layer-trace --blocking-launch >"$POSITIVE_LOG" 2>&1
   rg -F -x 'FIRST_LAYER_CUDA_HANDSHAKE_V2 layers=4 served=4 trace_fields=5 blocking_launch=1 models=0 PASS' "$POSITIVE_LOG"
   timeout --signal=TERM --kill-after=5s 20s "$CANDIDATE_BIN/host_graph_launch_test" --layer-trace >"$RUNTIME_LOG" 2>&1
   rg -F -x 'FIRST_LAYER_CUDA_HANDSHAKE_V2 layers=4 served=4 trace_fields=5 blocking_launch=0 models=0 PASS' "$RUNTIME_LOG"
   ```

   Negative control must exit 1 with
   `FAIL: traced graph CPU service before launch completion` after its bounded
   release, not timeout, signal, API failure or a different assertion. The forced
   positive must exit 0 with
   `FIRST_LAYER_CUDA_HANDSHAKE_V2 layers=4 served=4 trace_fields=5 blocking_launch=1 models=0 PASS`.
   The normal-runtime positive must also exit 0 with the matching blocking_launch=0
   marker. Preserve logs, elapsed time and context/buffer memory. Check fixture-owned
   cleanup before product start. A missing GPU/toolchain, failed gate or ambiguous
   result stops here with a concrete report. No full diagnostic is spent to
   investigate a failed gate. Ordinary in-scope fixture/launcher corrections may
   be made and reported; do not silently replace the test with CPU serialization.

5. On gate PASS, automatically continue the existing guarded attached controller
   with fresh private run roots. Set diagnostic=GPU-FIRST-LAYER-V2-20261022,
   diagnostic_revision=<this immutable revision>, approved_window_s=600,
   supervisor_commit=311561689bb4b1f8a109821601e3a6bda07067e6, and
   execution_approved=true only after the parent's execution grant. Use current
   first_layer_profile.py for all peers/controller. Engine/GPU agent configs keep
   gpu_completion=true, retention_approved=false and service_residency=false.
   Record the new 03 ELF hash separately from the frozen implementation binding.

   Engine/GPU peer command:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/first_layer_profile.py" --supervisor-root "$SUP_ROOT" --agent-config "$ROLE_CONFIG"
   ```

   05 peer is the read-only observer, with role=ROLE_CPU_C, new run_id and
   resident_config=<retained original CPU config absolute path>:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/first_layer_profile.py" --supervisor-root "$SUP_ROOT" --observe-config "$OBSERVER_CONFIG"
   ```

   Historical PIDs are not signal authority. Require pinned worker/monitor/
   guardian identities, fresh resource coverage, complete 111597465600-byte arena
   with UINT64_MAX idle TTL, and exact resident SETUP hit before any engine load.
   During an active request the observer checks identity/resource receipts without
   competing for the worker's single cache connection. No duplicate worker,
   restart, unload or silent cold-load/rebudget fallback. If absent/incompatible,
   report the failed check and stop; an approximately 429-second cold load is not
   included or authorized here. Recreate only the same at-most-four approved
   temporary forwards. If exact bindings/authority are unavailable, report that.

   Record T before controller startup and run:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/first_layer_profile.py" --supervisor-root "$SUP_ROOT" --controller-config "$CTRL_CONFIG"
   ```

   It runs fresh B0 = GEN 1 9419 then QUIT (no network GPU), then fresh
   G0 = GEN 1 9419 then QUIT (warm GPU cache). Require READY / DONE 1 / rc 0 and
   finite 248320-float first logits for each. No extra request or retry. Preserve
   primary 960 / helper 2247 / CPU 21369 / uncovered 18080 / remote 3289, CPU budget
   106428 MiB, payload 111597465600 B and the 92156-byte remote plan SHA-256
   a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578.
   CPU must reuse the same complete partition with zero new load/hash/network/
   allocation; G0 must reuse all 17176473600 assigned cache bytes with zero new
   source weight reads/network/persist. Keep all substantive host/cgroup/swap/
   RSS/identity/watchdog/cleanup/remote-GPU/disk guards. Local valid runtime VRAM
   reserve crossings remain warning-only, admission strict. A numerical failure
   alone does not destroy healthy residency or cancel the planned saved-input
   diagnosis. Another stall stops without G0/retry; preserve the new breadcrumbs.

6. Preserve first logits/routing and compare against retained B/G 9419 first
   logits to detect trace-induced changes. Formal centered rel-L1 <= .03 and
   KL <= .03 remain unchanged. Previous .0352/.0500 failures remain real; no
   quantizer repair or acceptance relaxation is included. Prior .0099 expert
   replay measured unquantized versus decoded quantized hidden, shared at that
   boundary, not established GPU-specific rounding. Decoded equality is not a
   claim of equal encoded bytes. Traced timing is diagnostic overhead.

   After engines/GPU are stopped, transfer only bounded records from 03 trace-B0,
   03 trace-G0 and 04 trace-G0 into three new 0700 directories on 05. Preserve
   before/after hashes. Records remain <=256 KiB each and <96 MiB in aggregate;
   pinned engine staging remains at most 17044480 B. Do not copy model/cache trees.

   The unchanged replayer uses the real native CPU kernels on each saved B/G
   input at the earliest material input/output boundary, or earliest bit difference
   if no material drift. It binds actual G rows to exact request bytes, routes,
   weights and masks, and reports expert/subset/local/host/final-device differences.
   No new CUDA allocation or GEN is required. It reads at most 160 MiB of routed
   expert payload, normally 104448000 B; separate those reads from trace I/O and
   product weight-transfer metrics. Helper per-expert rows and residual/head
   internals remain unavailable; do not turn an unresolved boundary into PASS.

   Replay config: role=ROLE_CPU_C, new run_root, normal namespaces, retained V1
   replayer hash, and seven absolute replay_argv strings:

   ```text
   [strata-layer-diagnose, B_directory, G_directory, GPU_directory,
    frozen_GPU_plan_on_05, retained_LOCAL_ROOT_C, retained_MANIFEST_C]
   ```

   On the original controller clock compute REMAINING_S=T+600-45-now immediately
   before dispatch. Use the existing bounded 05 SSH prefix, original deadline and
   at most five seconds dispatch margin; cancel stale dispatch. Execute:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/first_layer_profile.py" --supervisor-root "$SUP_ROOT" --replay-config "$REPLAY_CONFIG" --remaining-s "$REMAINING_S"
   ```

   Existing Session/Job/pidfd/reaping protection applies; replay caps remain
   120 wall seconds / 90 CPU seconds / 512 MiB AS and RSS/anonymous / 1 MiB output,
   with 8 GiB host headroom alongside the resident arena. Config/import/admission/
   hash time consumes the grant. Require rc 0, complete reaping and
   FIRST_LAYER_DIAGNOSTIC_COMPLETE_V1 (unchanged tool marker). If no safe interval
   remains, report replay NOT RUN; no overtime or generation retry.

7. Report source/build/fixture hashes, negative and positive gate outcomes,
   launch/query/doorbell evidence, first-token logit verdict and earliest bit/
   material layer differences, routed IDs/weights/ownership and same-input replay
   results. Separate diagnostic completion from numerical acceptance. Historical
   usage is the supplied **15 started requests / 833 generated tokens**, including
   one failed started V1 B0. The last confirmed older time was 7541.0 s, plus
   **approximately 117 s** for that failure; do not publish a fabricated precise
   combined receipt. Report fresh gate/build/continuation times separately.
   Report ongoing guarded CPU-service age separately from measurement time.
   New allowance is at most two started GEN1 requests, each charged its full
   one-token allowance; cumulative starts at most 17 and generated tokens at most
   835. Never reset historical accounting or spend a replacement request.

   Release engines, GPU worker/allocations, observer and owned temporary relays.
   Keep cache-B and the healthy resident 05 worker/monitor/guardian/listener live
   under original guards. Report cleanup by role; do not claim all ports free.
   The attached controller above is the reconnect path for these two requests.
   Explicit idle controls remain available via the approved 05 SSH command:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/combined_diagnostic.py" --supervisor-root "$SUP_ROOT" --resident-status "$OLD_CPU_CONFIG"
   "$PYTHON" -B -I "$SRC/tools/worker_control.py" unload --host 127.0.0.1 --port "$CPU_PORT_C"
   "$PYTHON" -B -I "$SRC/tools/worker_control.py" stop --host 127.0.0.1 --port "$CPU_PORT_C"
   ```

   Unload/stop are not trial teardown and are not authorized merely by this
   continuation. No new service lifetime or persistent access is created. Return
   the report promptly rather than waiting for residency to end.
