# GPU-FIRST-LAYER-V1-20261022 — prepared, no execution granted

Additive experiment from 0a54d0db28f884d731ccbc6ada55c62f3f976651, with frozen
product base bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5. No merge, deployment,
kernel repair, numerical threshold change or performance tuning. Preparation
and publication are approved. The parent must approve the proposed new Hermes
execution before it starts. New code, builds and focused tests are **NOT RUN**
by this author. Desktop work was static text, review, git and hashes only.

## Supplied evidence and the remaining question

GPU-COMBINED completed at 0a54d0db. Supplied PERF A/B decode was 36.399/35.437 s,
GPU exchange 28.443 s across 5291 requests and GPU warm setup 106.4 s. This is
reported full-path improvement; this diagnostic makes no additional performance
claim. Numerical centered rel-L1 .0352/.0500 still fails the unchanged .03
threshold. KL passed. PERF A was bitwise identical to retained G0/G1, and its
one-token capture was identical to PERF A. Evidence bundle SHA-256:
e89f5e44265c6b378906ffb35ed463e5d6a14b92c7aa8b0f7b171f344d4cdcf4
(2810755 bytes). These are supplied observations, not a newly inspected report.

The layer-1 expert-86 replay found decoded CPU/CUDA input and hidden quantization
equal, expert rel-L1 about 9.5e-8, and exact weighted GPU-row and host-merge sums.
Those were decoded-float comparisons, not proof that encoded quantization bytes
were identical. The .0099 comparison was CPU unquantized SwiGLU versus decoded
quantized hidden: shared quantization loss at that boundary, not evidence of
GPU-specific rounding causing the final failure. Do not change quantizers.

The missing observation is the final device combine and the B/G layer path.
This candidate records one fresh 9419 prompt through those boundaries in B and G,
reports earliest bit differences separately from material drift, then replays
only the selected differing layer with its saved actual inputs. No A/B64 replay.

The known fixture-only rename is retained: local service_setup rename hashes to
547c814ad7cb26eda294d02bb69268e80abf2b3e783fb692c93f6f4fbaf38538,
matching the supplied 547c814a prefix. Absolute supervisor-root resolution is
also retained. No private patch or fleet evidence was fetched to manufacture a
longer reported remote hash. Do not rerun those previously passed campaigns.

## What is captured, compared and replayed

Opt-in STRATA_FIRST_LAYER_TRACE_DIR on 03 records host-XXX.trace (actual MoE
float input, routed IDs/weights, dispatch kind, helper/CPU/GPU partials and host
sum), partition-000.trace (the exact CPU bitmap), and window-000.trace (actual
token/position). Device-XXX.trace contains actual unweighted local per-entry
rows, shared output, local combined output before remote addition, final device
output after remote_opt_->combine, and the device ownership mask. Kind values
are -1 local CPU, 0 primary VRAM, 1 PCIe, 2 helper, 3 CPU on 05, 4 CUDA on 04.

Verifier::record_window adds bounded asynchronous D2H nodes to the existing
CUDA graph/stream. It does not synchronize inside capture, change flags, replace
kernels or reorder existing arithmetic. Files are emitted only after the normal
window synchronization. The trace requires a complete first T=1/position 0/token 9419
window, at most 128 layers, H<=2560 and K=10. Other execution modes are not claimed.
Each fresh engine writes its first window only. Traced timings are diagnostic
overhead and must not be presented as an uninstrumented benchmark.

STRATA_GPU_LAYER_ROWS_DIR on 04 saves the actual first T=1 request, per-expert rows
restored to original entry order, and weighted partial for each observed layer.
These rows already exist on the worker host after normal execution. No resident
weight blob or whole-model dump is added. Each process writes an entry once into
a new private 0700 directory; existing files are never replaced.

Each record is <=256 KiB; fixed layer/name limits bound all records to <96 MiB
across B/G/04, normally substantially less. Engine pinned staging is at most
128*((10+3)*2560+10)*4 = 17044480 bytes, through the existing host-allocation guard.
Other trace work uses bounded per-record buffers. Host/cgroup/RSS/swap/VRAM/disk
protection remains active. Trace failures refuse the diagnostic, never silently
produce an empty boundary.

strata-layer-diagnose checks finite complete records, window/geometry association
and exact B/G CPU bitmap equality with 21369 experts. It reports all layer input,
routing and final-device differences plus local reduction, host sum, device add
and ownership-mask diagnostics. It reports both first bitwise difference and
first maxabs >1e-5*max(1,maxabs(reference)), the existing drift scale used only to
classify this diagnostic. Formal centered rel-L1<=.03 and KL<=.03 stay unchanged.
Expected CPU-to-network-GPU ownership changes are visible, not labeled a routing
bug merely because ownership differs between B and G.

For the earliest material input or output boundary (otherwise the earliest
bit difference), replay reads at most ten routed expert blobs per saved input,
one at a time, with existing manifest/range/digest validation. Existing native
CPU kernels run on the saved B input and saved G input separately. G GPU-owned
entries compare against actual saved CUDA rows with exactly matching request
inputs, IDs, weights and mask. Primary rows and CPU/helper/GPU weighted subsets
are also compared where available. No new GPU allocation or GEN request is needed
for replay. Maximum logical source reads are 160 MiB, normally 104448000 bytes.
Helper per-expert rows were reduced before return; they remain unavailable.

This can distinguish expert computation from weighting/masking/final addition
at the observed boundary. Tiny upstream input differences are not identical-input
claims. If every observed MoE boundary agrees while logits differ, the residual,
attention/head path remains unresolved; report that, not a correctness PASS or
permission for another capture campaign.

## Resident attachment and proposed budget

The existing healthy CPU service stays resident on 05 under its original guardian
and monitor. Historical numeric PIDs are never authority to signal processes.
Reverify its resident.json identity receipt, fresh resource coverage, full arena
111597465600 bytes and UINT64_MAX idle status through the retained private CPU
config. The previous blanket "no processes / all ports free" sentence applies
to stopped engine/GPU roles, not this explicitly retained worker and its listener.

The new 05 peer is a read-only observer over the existing authorized SSH command.
It pins existing identities and verifies fresh monitor samples; it never creates,
signals, unloads, stops or replaces the CPU service. It queries cache status only
before/after engines, because the worker has one active connection. During an
engine request it checks identities and fresh guard records without opening a
second cache connection. Complete append records and bounded journal rotation
are reconciled; malformed, unreadable or stale evidence still fails.

Each engine sets STRATA_REQUIRE_CPU_RESIDENT_HIT=1: exact resident SETUP binding
must succeed before any local load/upload. Incompatible or missing residency
fails closed. The actual captured CPU bitmaps must also match between B/G.
Do not restart/unload this service or rerun the old serial CPU-start controller.
If absent/incompatible, report the target and failed check; an additional cold
load costs about 429 s and needs a separately considered fallback budget.

Propose a **600-second / ten-minute measurement window**, T before owned
transport/observer startup through engine/GPU cleanup, transfer and guarded replay.
Keep 45 s global cleanup reserve and original clipped leases/caps. Conservative
planning: two 90 s engine allowances + 120 s warm GPU setup + two 20 s first requests
+ 11 s gap + 30 s retrieval + 120 s replay + 45 s cleanup = 546 s. This estimate is not
a measured new run. Actual prior CPU reconnect was .004 s; no cold CPU allowance
is silently included. No full-maximum-envelope admission or running extension.

Exact streams, each in a fresh engine:

```text
B0 = baseline with resident CPU, no network GPU: GEN 1 9419; QUIT
G0 = distributed with resident CPU and warm 04 cache: GEN 1 9419; QUIT
```

The public internal phase labels refer only to new run roots. New allowance is
**two started requests / two tokens**. Charge interrupted starts in full. Historical
14 requests / 833 tokens / 7541.0 active seconds remain recorded; cumulative maximum
becomes 16 requests / 835 tokens. Report new elapsed and ongoing resident-service
age separately. Engine/GPU preparation is outside the measurement window while
the existing CPU service remains guarded and resident, not falsely "all stopped".

## Complete Hermes handoff

1. After the parent grants the proposed execution, use the immutable candidate
   and GPU-FIRST-LAYER-V1.SHA256SUMS. Keep main, draft PR2, validated binaries,
   retained baseline/capture evidence and full cache-B untouched. Use existing
   ROLE_CONTROLLER and private role/namespace/plan/model receipts. No desktop
   execution, dependency downloads, new credentials/routes/security settings,
   persistent tunnels, merge, deployment or full-model hashing/replanning.

2. Stage a new 31156168 supervisor/shared-guard package without changing the live
   CPU guardian's historical files. Verify the retained package manifest, then
   apply the two small published patches in order. SUP_ROOT must be absolute:

   ```sh
   git -C "$SUP_ROOT" apply --check --unidiff-zero "$EXPERIMENTS/supervisor-residency.patch"
   git -C "$SUP_ROOT" apply --unidiff-zero "$EXPERIMENTS/supervisor-residency.patch"
   git -C "$SUP_ROOT" apply --check --unidiff-zero "$EXPERIMENTS/first-layer-supervisor.patch"
   git -C "$SUP_ROOT" apply --unidiff-zero "$EXPERIMENTS/first-layer-supervisor.patch"
   ```

   Postimage agent.py SHA 4307f89265617cf18c6ac747b1b19193457172c93dfaa902db81e011ff924849,
   policy.py SHA d3b1de8805aef47041df1b81726cec6e6c7623ef117df22c1420695bec660cb8,
   controller.py SHA 13d0bbc84e39761b17fe09500d763f6f7521b69fac314c1ed2472f3670a686c9.
   The extra patch only permits the exact B0/G0 order and separates engine/GPU
   expected phase lists; GPU starts at G0, never a fabricated B0 service.

3. Reuse existing dedicated incremental source/build trees, dependencies and
   compiler/backend settings on 03/04/05. Preserve old ELFs. Set a new private
   CANDIDATE_BIN before building. Serial only, <=900 s per host; 05 build memory
   <=2 GiB and 03/04 <=8 GiB under the existing guards; logs <=1 MiB each. Keep
   enough host headroom for the live CPU arena; do not disable its guard or unload
   it to make a build fit. No full clean rebuild if the suitable incremental tree
   is unavailable: report that concrete prerequisite.

   ```sh
   cmake -S "$SRC" -B "$BUILD" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$CANDIDATE_BIN" -DSTRATA_BUILD_TESTS=ON
   # 05 existing native-ON CPU configuration, no CUDA:
   cmake --build "$BUILD" --parallel 1 --target layer_trace_test strata-layer-diagnose
   # 03 existing accepted CUDA configuration:
   cmake --build "$BUILD" --parallel 1 --target strata
   # 04 existing accepted CUDA configuration:
   cmake --build "$BUILD" --parallel 1 --target strata-cuda-expert-worker
   # New focused checks only, on 05:
   "$CANDIDATE_BIN/layer_trace_test"
   "$PYTHON" -B -I "$EXPERIMENTS/test_first_layer_profile.py" --supervisor-root "$SUP_ROOT"
   ```

   Require rc0, FIRST_LAYER_FIXTURES_V1 failures=0 models=0 cuda=0 and
   FIRST_LAYER_PROFILE_PASS_V1 tests=7 workloads=0, no failures/errors/skips.
   C++ fixture 512 MiB AS/10 CPU/30 wall s, synthetic files <=32 MiB, output <=1 MiB;
   Python fixture 128 MiB AS/5 CPU/10 wall s, output <=1 MiB. Reuse already passed
   framing/periodic/residency campaigns. The old corrected residency fixture is
   preserved but not a new prerequisite. Do not rebuild/restart the live 05 worker.

4. Create fresh private run roots/configs. All new configs bind diagnostic=
   GPU-FIRST-LAYER-V1-20261022, supervisor_commit=311561689bb4b1f8a109821601e3a6bda07067e6,
   diagnostic_revision=<immutable candidate>, approved_window_s=600 and
   execution_approved=true only after the parent's execution grant. Engine/GPU
   agent configs use gpu_completion=true, retention_approved=false and
   service_residency=false. Legacy implementation=bf51f09e names the frozen base;
   record new engine/GPU ELF hashes separately and bind the actual new binaries.

   Controller keeps the existing three peer role names and <=4 exact temporary
   forwards from the retained approved transport config. Replace 03/04 remote agent
   invocation with:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/first_layer_profile.py" --supervisor-root "$SUP_ROOT" --agent-config "$ROLE_CONFIG"
   ```

   Replace the 05 CPU peer with the observer, not an agent-config launch:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/first_layer_profile.py" --supervisor-root "$SUP_ROOT" --observe-config "$OBSERVER_CONFIG"
   ```

   Observer config role=ROLE_CPU_C, new run_id, and resident_config=<retained old
   CPU config absolute path>. That old config/receipt names the live service;
   do not substitute historical numeric PIDs or the new observer run root.
   Reuse the same approved controller-mediated loopback relays and numeric endpoint
   bindings. Recreating only those temporary owned relays is in-window. If exact
   bindings/authority are missing, report them; do not improvise a transport.

5. Record T before startup and execute the attached controller:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/first_layer_profile.py" --supervisor-root "$SUP_ROOT" --controller-config "$CTRL_CONFIG"
   ```

   It checks the resident service, runs B0 then G0, reaps engines, stops/reaps the
   GPU worker, verifies final idle residency and closes observers/forwards. CPU
   failure stops dependent work; ending the observer never stops the resident
   service. Require each engine READY/DONE1/rc0 and finite 248320-float first logits.
   Preserve primary 960/helper 2247/CPU 21369/uncovered 18080/remote 3289, CPU budget
   106428 MiB, arena 111597465600 B and plan 92156 B SHA
   a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578.
   G0 must validate full warm reuse 17176473600 B with zero new source weight
   reads/network/persist; both engines must reuse the exact CPU arena with zero
   new load/hash/network/allocation. No numerical threshold or substantive host,
   cgroup, swap, RSS, identity, watchdog, remote GPU/disk guard is changed. Local
   runtime VRAM crossings remain warning-only; admission floors remain strict.

6. Preserve first logits/routing and compare B0/G0 with the retained B/G 9419 first
   logits. Report any instrumentation-induced difference; do not conceal it or
   launch another request. Formal B/G centered rel-L1<=.03 and KL<=.03 remain.
   Numerical failure alone does not cancel the planned artifact diagnosis while
   products/guards remain healthy.

   After engine/GPU cleanup, transfer only the bounded trace records from 03's
   trace-B0, 03's trace-G0 and 04's trace-G0 into three new 0700 directories on 05,
   using existing authorized role transfers. Record small-file hashes before/
   after. Do not copy model or cache directories. Bind replay config role=ROLE_CPU_C,
   new run_root, normal namespaces, new replayer binary_sha256, and exactly seven
   absolute replay_argv strings:

   ```text
   [strata-layer-diagnose, B_directory, G_directory, GPU_directory,
    frozen_GPU_plan_on_05, retained_LOCAL_ROOT_C, retained_MANIFEST_C]
   ```

   On the original controller clock calculate REMAINING_S=T+600-45-now immediately
   before dispatch. Use the retained 05 SSH prefix with the same externally bounded
   original deadline and at most five-second dispatch margin; cancel a stale
   dispatch before replay. Run:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/first_layer_profile.py" --supervisor-root "$SUP_ROOT" --replay-config "$REPLAY_CONFIG" --remaining-s "$REMAINING_S"
   ```

   Reuses existing Session/Job/pidfd/reaping guards, the small separate replay
   role 512 MiB RSS/anonymous cap and 8 GiB host headroom alongside the live arena.
   Config/import/admission/hash time consumes the grant; cap <=120 s, CPU90 s,
   AS 512 MiB, output 1 MiB, owned cleanup retained. Require complete reaping, rc0
   and FIRST_LAYER_DIAGNOSTIC_COMPLETE_V1. If no safe interval remains, report
   replay NOT RUN; no overtime, counter reset or generation retry.

7. Return one concise report: source/build/fixture hashes, exact first-token
   verdict/logit drift, earliest bit/material input/output boundaries, routed
   IDs/weights/ownership/device masks, same-input expert and merge replay results,
   unavailable helper rows/residual/head observations, cache/CPU reuse, separate
   source-read/trace I/O and historical/new accounting. Distinguish diagnostic
   completion from numerical acceptance. Stop engine/GPU/observer/owned-forward
   roles, retain full cache-B and the healthy guarded CPU service. Report 05's
   worker/monitor/guardian and listener as intentionally live, not "all ports free".

   Existing explicit idle controls remain available through the approved 05 SSH
   command, and are not trial teardown commands:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/combined_diagnostic.py" --supervisor-root "$SUP_ROOT" --resident-status "$OLD_CPU_CONFIG"
   "$PYTHON" -B -I "$SRC/tools/worker_control.py" unload --host 127.0.0.1 --port "$CPU_PORT_C"
   "$PYTHON" -B -I "$SRC/tools/worker_control.py" stop --host 127.0.0.1 --port "$CPU_PORT_C"
   ```

   Do not execute unload/stop without that explicit intent. This attached
   controller is the guarded reconnect path for the two authorized requests;
   later runs require their own allowance. No extra campaign or performance work.
