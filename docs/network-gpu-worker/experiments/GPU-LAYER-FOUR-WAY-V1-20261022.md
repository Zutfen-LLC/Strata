# GPU-LAYER-FOUR-WAY-V1-20261022 — approved offline follow-up

Additive from d97e2801e99a33bb3d9a546c1adbe4dfc1808176. Reuse the saved V2
captures and existing CPU-only replay machinery. No new capture, GEN, engine,
GPU worker, relay, service, production kernel/quantizer/threshold change, merge
or deployment. All new builds, fixtures and replay are **NOT RUN** by this author.
Valinor work was static source/text/review/git/hash/publication only; no Geekom
or fleet execution occurred. This authorized offline task needs no new approval
checkpoint. Hermes performs the bounded 05 execution through the existing channel.

## Supplied result and question

The d97e2801 offline run reportedly passed: build 8 s, fixture <1 s, replay 1 s,
10 blobs / 52224000 B / zero GEN. Seed source index 1 (the second layer) had
bitwise-identical B/G inputs, ten IDs/order, exact routing weights, shared output,
local routed rows, local pre-remote output and helper partial. The only ownership
change was entry 4 / expert 86 / weight 0x1.d6ec94p-5, CPU in B and GPU in G.
These identifiers are supplied observations, not selection constants.

Native-FMA shared-last reproduced the actual local output; float64 did not.
Native replay CPU partials matched both modes bitwise. G GPU partial differed
from its CPU surrogate by 2.9e-10. Earlier predicted final delta matches were
2131/2560 coordinates (native) and 2148/2560 (hybrid), residual max 1.86e-9 against
actual max 3.73e-9. Surrogate local rows differed by <=1.1e-8, so this follow-up
stops regenerating those already-observed matching local components.

These are supplied results, not a newly inspected private report. Resolve the
retained diagnose-full.out/report and V2 trace paths from existing receipts.
V2 source evidence archive SHA-256:
ebffeafcc21daf6a1a63f620b5bc001cf59b9fb1f55889cdfb2ce6cf84d24aa3.

## Exactly four supported cells

The tool keeps the seven absolute path arguments and earliest BIT-different
output / byte-identical input selector. It verifies the seed's input, IDs/order,
weights, captured local/shared/rows/helper and ownership mask controls. It derives
exactly one 3->4 CPU-to-GPU change from the two owner arrays, with no other owner
change or baseline GPU ownership. It prints the actual entry/expert/weight and
binds the saved G CUDA rows to exact request input/weights/IDs/mask/partial bytes.
Missing data, changed controls or an unsupported owner pattern refuses the
comparison with a concrete reason. Do not hardcode 1/4/86 to force a result.

Hold the captured identical local pre-remote output L and helper partial H fixed.
Only baseline CPU-owned expert rows need native replay. Existing kernels compute
them once from the common input, one bounded blob at a time. Local/helper-owned
experts are not probed. All other CPU rows, routing IDs/order and weights remain
fixed; the sole substituted row is the derived moved expert. The native row comes
from the existing single-entry, unit-weight service probe, including its signed-
zero behavior; the CUDA row is the saved unweighted expert output.

The four predictions are:

| Cell | Moved row | Reduction ownership/order |
| --- | --- | --- |
| CPU_baseline (00) | CPU-native | baseline B |
| CPU_distributed (01) | same CPU-native row | distributed G |
| CUDA_baseline (10) | captured CUDA row | baseline B |
| CUDA_distributed (11) | same captured CUDA row | distributed G |

For each cell, the actual worker's accumulate_partial applies weights once in
original entry order to the CPU and GPU masks separately, starting each partial
at zero. No entries are re-sorted, subtracted out of a rounded captured partial,
or merged into a new all-expert order. Then, with explicit float rounding at each
addition, remote = RN(RN(H + GPU_partial) + CPU_partial), final = RN(L + remote).
The captured L/H are copied, not regenerated from surrogate expert rows. No
helper per-expert data is needed or invented for this fixed-partial experiment.

Every predicted CPU/GPU/remote/final component is compared to BOTH observed B and
G components. Cell 00 is the baseline-like replay; cell 11 is distributed-like.
Cells 01/10 are counterfactuals, and all four remain predictions until compared.
Each cell's delta relative to predicted 00 is compared with observed G-B using
double differences of the original floats: exact coordinate count, max actual/
predicted delta and residual. Check diagonal component matches as well as final
delta; cancellation alone is not a faithful reconstruction.

For each component, with Yxy denoting the corresponding cell, the report gives:

- row effect at B = Y10-Y00; at G = Y11-Y01;
- grouping effect at CPU row = Y01-Y00; at CUDA row = Y11-Y10;
- average row main effect = ((Y10-Y00)+(Y11-Y01))/2;
- average grouping main effect = ((Y01-Y00)+(Y11-Y10))/2;
- interaction = (Y11-Y01)-(Y10-Y00).

Effects use double arithmetic on cell floats and report nonzero coordinate count,
maxabs and RMS. They are conditional counterfactual effects, not observations,
quality metrics or an end-to-end causal intervention. Max absolute residual
alone cannot establish a universal 1–2 ULP bound. Formal NUMERICAL_FAIL remains;
the full-logit 3.5–5% centered rel-L1 discrepancy is not explained merely by
matching the seed, and is not a measured quality loss. Do not tune thresholds,
change kernels or launch a consistent-order full-model intervention here.

## Complete Hermes procedure

1. Obtain this immutable revision and verify GPU-LAYER-FOUR-WAY-V1.SHA256SUMS.
   Preserve prior sources/binaries, diagnose-full.out, reports, logits, captures
   and hash receipts. Resolve the existing three owned 0700 B/G/GPU trace roots,
   frozen GPU plan, LOCAL_ROOT_C and MANIFEST_C from retained execution/replay
   receipts; do not guess paths or fetch/extract another model. Use the existing
   approved controller-to-05 command only; no relays, new access, network/security
   settings or dependency downloads. Do not contact the resident inference socket.

2. Reuse the accepted native CPU incremental tree and dependencies on 05. Use a
   new private CANDIDATE_BIN and preserve old ELFs. Serial build <=600 s, <=2 GiB
   build memory, logs <=1 MiB under existing resource controls. Preserve the live
   arena's guards/headroom. Missing suitable build state is a reported prerequisite,
   not authority for a clean rebuild, dependency install, service unload/restart.

   ```sh
   cmake -S "$SRC" -B "$BUILD" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$CANDIDATE_BIN" -DSTRATA_BUILD_TESTS=ON
   cmake --build "$BUILD" --parallel 1 --target layer_grouping_test strata-layer-diagnose
   timeout --signal=TERM --kill-after=5s 10s "$CANDIDATE_BIN/layer_grouping_test"
   ```

   Require rc 0 and LAYER_FOUR_WAY_FIXTURES_V1 PASS models=0 cuda=0. The small
   focused fixture adds derived-entry refusal cases, fixed-anchor four-cell
   values and exact conditional/main/interaction expectations. Existing tiny
   selector/grouping checks remain in that same target; no old campaigns run.
   Fixture caps stay 128 MiB AS / 5 CPU seconds / 10 wall seconds / 1 MiB output.

3. On fixture PASS, automatically execute ONE guarded offline replay. Reuse the
   existing absolute supervisor/shared-guard stage at
   311561689bb4b1f8a109821601e3a6bda07067e6 with its V2 patches. Leave live guardian
   files, original ownership/baselines and worker/monitor/guardian untouched.
   Keep required admission receipts fresh through the existing approved
   administrator channel. If it is unavailable, report the exact admission
   blocker; do not waive age or create privileges/security changes.

   New private config: diagnostic=GPU-LAYER-FOUR-WAY-V1-20261022,
   diagnostic_revision=<this immutable revision>, offline_only=true,
   supervisor_commit=311561689bb4b1f8a109821601e3a6bda07067e6,
   approved_window_s=120, role=ROLE_CPU_C, fresh run_id/run_root, existing trusted
   namespaces and the newly built replayer's actual binary_sha256. Exactly seven
   absolute replay_argv strings, resolved from retained receipts:

   ```text
   [new_strata-layer-diagnose, B_directory, G_directory, GPU_directory,
    frozen_GPU_plan_on_05, retained_LOCAL_ROOT_C, retained_MANIFEST_C]
   ```

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/layer_grouping_offline.py" --supervisor-root "$SUP_ROOT" --replay-config "$REPLAY_CONFIG"
   ```

   Existing Session/Job/pidfd admission, identity, host/cgroup/swap/RSS/error and
   reaping guards remain. Start the 120-second replay grant before config/import/
   admission/hash, preserve its 15-second owned-cleanup reserve, 90 CPU seconds,
   512 MiB AS/RSS/anonymous cap, 8 GiB headroom and 1 MiB output. Bound external
   dispatch to 125 seconds including the existing channel's small dispatch margin.
   At most ten 8-MiB blobs (80 MiB) are read, one at a time; this version reads
   fewer when local/helper entries are present. Report actual count/bytes rather
   than demanding the prior 10 blobs. Frozen plan/placement and caches are unchanged.

   Require rc 0, complete owned reaping, FOUR_WAY_CONTROLS, all four cell reports,
   FOUR_WAY_DELTA and FOUR_WAY_EFFECT lines, and
   LAYER_FOUR_WAY_DIAGNOSTIC_COMPLETE_V1 numerical_acceptance_not_implied=1 new_GEN=0.
   A completion marker without supported four cells is not the requested result.
   Report missing/unsupported evidence or a failed guard concretely. Ordinary
   in-scope offline launcher/fixture corrections are allowed and must be recorded;
   no new model request or broad revalidation is a fallback.

4. Return one concise report: derived seed/entry/expert/weight and fixed controls;
   CPU/GPU/remote/final matches for each cell against observed B/G; the joint
   delta residual and conditional/main effects/interaction; any rounding or
   observability limits. Separate observations, replay predictions and the two
   counterfactual cells. Do not infer full-logit causality, universal ULP bounds
   or quality loss from the seed. Keep .03 acceptance thresholds and NUMERICAL_FAIL.

   This task adds zero GEN/tokens: cumulative usage remains 17 requests / 835
   tokens. Preserve V2's 271.3 s and the earlier 7541.0 s + approximately 117 s as
   separate reported history; report new offline elapsed separately. Previous
   offline build/test/replay timing is not a new product window or reset ledger.
   Stop/reap only new fixture/replayer supervision. Keep the healthy 111597465600-B
   CPU arena resident and the 17176473600-B GPU cache preserved. No stop, restart,
   unload, duplicate worker, numeric-PID signal, new service, merge or deployment.
