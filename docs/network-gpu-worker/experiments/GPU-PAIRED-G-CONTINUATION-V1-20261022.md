# GPU-PAIRED-G-CONTINUATION-V1-20261022

Approved G-only completion of the paired practical evaluation. Additive to
e0d11fc765db3f8b07ea70d3535738a630fc129e; preserve that revision and failed run.
The owner approved this correction, publication on experiment/gpu-first-layer-v1
and the fresh 1500-second G-only window. This author performed static text/source/
git/hash/publication work only. New fixtures and model execution are **NOT RUN**.

## What is established

Supplied prior result: B0 completed five tasks, 51 emitted tokens, four strict
passes and a substantively correct Python answer rejected for fenced format.
B1 completed both supplied-answer probes, two tokens. No G inference started.
G0 setup stopped after the attached CPU observer failed; the resident service
and its independent monitor remained healthy. The observer's exact cause was
not retained. Do not manufacture it or weaken leases/freshness checks.
The stated 16:46:35–16:57:34 timestamps span 659 seconds, not expiry of a
1500-second grant. Keep that arithmetic correction distinct from causal evidence.

In the actual staged composition (31156168 plus supervisor-residency.patch then
first-layer-supervisor.patch), Peer.pump validates run/role, then raises on a failed
FINAL before appending the message; Peer also discards stderr. The new third patch
retains the actual FINAL/event before the same refusal, writes it independently
of the event queue, and records the last 64 KiB of each peer's stderr and its
actual controller-owned transport exit status. That status may reflect cleanup's
signal or SSH return, not the resident CPU worker's status. Stderr never counts as
protocol progress. Existing identity, sequence, lease, deadline and cleanup rules
remain. Invalid identities still fail without being accepted as valid FINALs.
No claim is made that stderr alone identifies the previous observer failure.

Peer diagnostics are <=512 KiB JSON per peer, with observed stderr byte count,
truncation and EOF flags; nonblocking drains do bounded work. Receipts are replaced
atomically so cleanup updates do not first truncate a retained failure packet.
There is no new monitor or supervisory framework. The one existing controller
clock starts before relay/peer startup and is retained as execution-clock.json,
including deadline, cleanup reserve, actual elapsed time and failure/close status.

## Scope and accounting

Run only G0's identical five GEN32 tasks, then G1's identical two supplied-answer
GEN1 probes. Both streams end with QUIT. No B rerun, new prompt, retokenization,
warmup or replacement request. The exact retained plan SHA-256 is:

```text
1c81fa79b9a26f66b0a4f7b2e6eae92858c82f47de827f08751b312948a21792
```

The existing evaluator, prompts, strict checks, output caps, common-input numerical
measurements and timing scopes are unchanged. A wrapper verifies complete retained
B output/config/plan association and finite numbered probe logits, then calls that
same evaluator with separate B/G roots. It decodes using the retained metadata;
it does not encode prompts again. Label this a **separate-run comparison**.
Keep actual text, strict-format versus substantive observations, and
OUTPUT_CAP_REACHED distinct. Do not silently upgrade the fenced B code to a strict
pass; any substantive assessment is a labeled reading of the actual answer.

Fresh window: **1500 seconds**, with **45 seconds reserved for cleanup**; one
monotonic start before controller/temporary-relay startup, no extensions or reset.
Allowance: **7 started requests / 162 generated tokens maximum**. Charge each
interrupted started request its whole 32- or 1-token allowance. Supplied cumulative
actual usage begins at **24 requests / 888 tokens**; maxima become **31 / 1050**,
below the previous 31 / 1159 ceiling. Report old elapsed evidence and new elapsed
separately. Do not fabricate a precise combined historical timer.

Keep the same accepted contiguous-frame engine/GPU-worker ELFs, placement and
runtime flags. No build or product change. Primary 960 / helper 2247 / CPU 21369 /
uncovered 18080 / remote 3289; CPU budget 106428 MiB, arena 111597465600 B; remote
plan 92156 B SHA a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578.
Preserve the complete 04 cache 17176473600 B. Require product counters for full
reuse and zero new source-weight reads/network/persist, or mark those quantities
unmeasured; unchanged file mtimes prove none of them. Do not delete/refill/mirror
the cache. Keep historical NUMERICAL_FAIL and .03 centered rel-L1 / .03 KL limits;
numerical screening alone does not cancel healthy paired task completion.

## Complete Hermes continuation

1. Obtain this immutable commit and verify GPU-PAIRED-G-CONTINUATION-V1.SHA256SUMS.
   Reuse existing ROLE_CONTROLLER, private endpoint/namespace/admission receipts,
   and the same at-most-four temporary controller-mediated loopback forwards.
   No new route/security setting/credential, dependency download, product rebuild,
   model/weight hash campaign, merge or deployment. Keep private paths, configs,
   model outputs and old evidence off GitHub. No desktop model/test/build workload.

2. Stage a new private copy of the already validated absolute SUP_ROOT on the
   required roles, without modifying any live CPU guardian's historical files.
   Preserve its shared planning modules in SUP_ROOT.parent as well as the
   supervisor directory; do not stage only controller.py.
   It must already contain the two old patches, in their original order. Verify:

   ```text
   agent.py      4307f89265617cf18c6ac747b1b19193457172c93dfaa902db81e011ff924849
   policy.py     d3b1de8805aef47041df1b81726cec6e6c7623ef117df22c1420695bec660cb8
   controller.py 13d0bbc84e39761b17fe09500d763f6f7521b69fac314c1ed2472f3670a686c9
   ```

   Apply only the new diagnostic patch to that copy:

   ```sh
   EXPERIMENTS="$SRC/docs/network-gpu-worker/experiments"
   git -C "$SUP_ROOT" apply --check --unidiff-zero "$EXPERIMENTS/peer-diagnostics.patch"
   git -C "$SUP_ROOT" apply --unidiff-zero "$EXPERIMENTS/peer-diagnostics.patch"
   ```

   New controller.py SHA-256 must be
   fa89b9554442a71bc54b7d1098d31831b76a19ba0a15c93cfc5cda9c3a49cce3.
   Agent/policy postimages stay as above. Do not apply this patch to the bare
   planning source or assume an unpatched base is the actual running composition.

3. On 05 run only the new focused fixture:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/test_paired_g_continuation.py" --supervisor-root "$SUP_ROOT"
   ```

   Require rc 0 and PAIRED_G_CONTINUATION_PASS_V1 tests=7 workloads=0, no failures,
   errors or skips. Bounds are 256 MiB AS / 10 CPU seconds / 20 wall seconds,
   <=1 MiB output with the existing bounded launcher. It uses small synthetic
   records and mocked processes; no socket, process, model or fleet workload is
   created. Reuse prior fixture passes; do not rerun unrelated campaigns.

4. On PASS, automatically check the retained B evidence. Resolve PLAN, B_ROOT,
   B_ENGINE_CONFIG and PACK_META from the retained original receipts. B_ROOT is
   the original 03 engine run-root layout containing B0/B1 output/clock, B1.logpos
   and the two B1.first-logits.N files. B_ENGINE_CONFIG is the original private
   engine config snapshot at e0d11fc7, not a newly rewritten reconstruction. Verify
   retrieved copies against existing receipts; copy only these small artifacts,
   not model/cache trees. Retain the exact plan bytes on all roles.

   Run this bounded check on 05 and make the same checked files available to the
   approved controller for its automatic pre-launch check:

   ```sh
   timeout --signal=TERM --kill-after=5s 60s "$PYTHON" -B -I "$EXPERIMENTS/paired_g_continuation.py" --plan "$PLAN" --b-root "$B_ROOT" --b-engine-config "$B_ENGINE_CONFIG" check >"$BASELINE_CHECK"
   ```

   Use existing host/swap/headroom controls, <=256 MiB AS / 10 CPU seconds /
   60 wall seconds / 1 MiB output. Require complete=true, seven requests, 53
   generated tokens, correct original revision/plan binding and finite matching
   probe/logit association. Missing/inconsistent B artifacts stop before any G
   launch; never repeat B or retokenize as a fallback. Do not run paired_tasks.py
   prepare, or its old report CLI (which regenerates the token plan).

5. Reuse the original healthy resident_config and independently guarded CPU arena.
   Revalidate live worker/monitor/guardian identities, same complete partition,
   UINT64_MAX idle TTL, resource freshness and exact resident SETUP hits. Historical
   numeric PIDs are not authority. Refresh a stale admission receipt only through
   the existing approved administrator channel. No service restart, unload,
   duplicate, replacement or cold fallback. If a genuine guard fails, preserve
   its real reason and stop; do not speculate that a lease can be relaxed.

6. Use fresh run roots/configs. Every new peer/controller/observer config sets:

   ```text
   diagnostic = GPU-PAIRED-G-CONTINUATION-V1-20261022
   diagnostic_revision = this immutable commit
   supervisor_commit = 311561689bb4b1f8a109821601e3a6bda07067e6
   approved_window_s = 1500
   execution_approved = true
   plan = absolute path to identical retained PLAN
   plan_sha256 = 1c81fa79b9a26f66b0a4f7b2e6eae92858c82f47de827f08751b312948a21792
   retained_engine_sha256 = original B engine config's binary_sha256
   ```

   New engine binary_sha256 must equal retained_engine_sha256. Bind the same
   accepted GPU worker and all previous plan/model/placement/resource settings.
   Agent configs retain implementation=bf51f09e, gpu_completion=true,
   retention_approved=false, service_residency=false. RUN_A equals the new engine
   run_root. Controller additionally sets controller_evidence=<new absolute 0700
   directory, not pre-created>, retained_b={root: B_ROOT, engine_config:
   B_ENGINE_CONFIG}; these paths are local to the controller. The old resident
   config is referenced only by the CPU observer, never a CPU-start agent.

   Use the same exact peer SSH prefixes/endpoint bindings as approved previously.
   Engine/GPU peer command:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --agent-config "$ROLE_CONFIG"
   ```

   05 observer command:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --observe-config "$OBSERVER_CONFIG"
   ```

   Start the controller once; its retained clock covers all owned relay/peer
   startup, validation, G0/G1 and cleanup:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --controller-config "$CTRL_CONFIG" >"$EXECUTION_RESULT"
   ```

   It runs G0 five GEN32 requests then G1 two GEN1 probes. One GPU worker is owned
   from G0 through G1, then stopped; cache validation/loading occurs for each
   engine connection. All fresh caps clip to the remaining window; no running
   extensions. The 45-second reserve remains real. A repeated observer/product
   fault stops safely with its packet, bounded stderr, transport exit status and
   clock receipt. Ordinary in-scope launcher mistakes may be corrected and
   documented without new approval, but record failed attempts and do not hide
   extra requests or spend replacements beyond this schedule.

7. Release owned engines/GPU/temporary transports; preserve healthy CPU residency
   with its original independent guardian/monitor. Confirm cleanup by host/role,
   not a blanket all-processes/all-ports statement. Retain full 04 cache and prior
   failed evidence. Retrieve only bounded new outputs, two probe logit files,
   logpos/clock/counter/cleanup records and all controller peer diagnostic JSONs.

   After G, run the unchanged evaluator through the no-retokenization wrapper on
   05. B_ROOT stays the retained B root; G_ROOT is the new retrieved G root:

   ```sh
   (
     ulimit -v 1048576
     ulimit -t 30
     timeout --signal=TERM --kill-after=5s 60s "$PYTHON" -B -I "$EXPERIMENTS/paired_g_continuation.py" --plan "$PLAN" --b-root "$B_ROOT" --b-engine-config "$B_ENGINE_CONFIG" report --pack "$PACK_META" --g-root "$G_ROOT" >"$PAIRED_RESULT"
   )
   ```

   Keep output <=1 MiB. Report separate-run B/G task pairs, raw decoded text,
   strict passes/format failures/substantive observations/cap exhaustion, engine
   prompt/decode and observed warm-wall scopes, CPU/GPU participation and network
   setup versus steady-state counters. The common-input probes compare identical
   supplied sequences, not divergent generated continuations. Numerical failure
   is recorded without redefining historical thresholds. Unavailable counters are
   unmeasured, not inferred zero from cache mtimes. Include the new monotonic
   receipt and accurate usage/cleanup/residency; no model retry or further campaign.
