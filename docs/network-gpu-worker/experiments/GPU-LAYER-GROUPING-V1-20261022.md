# GPU-LAYER-GROUPING-V1-20261022 — offline follow-up

Additive CPU-only tool correction from b1184eb0280bed93a75e9e1b9339312cdd55f3f6.
The user authorized this offline follow-up. It adds zero GEN requests/tokens and
starts no engine, GPU worker, CPU arena, listener or transport relay. No production
kernel, quantizer, threshold, placement or capture instrumentation is changed.
All new builds, tests and replay are **NOT RUN** by this author; desktop work was
static source/text/review/git/hash/publication only. Hermes executes the bounded
offline work on 05 using the existing approved controller channel.

## Supplied observations and corrected interpretation

V2 CUDA gates passed; B0/G0 each completed DONE 1 / rc 0 and traced logits were
bitwise equal to retained untraced B/G logits. Formal centered rel-L1 .035174 is
above .03: NUMERICAL_FAIL remains. KL .00196 passes. First output bit drift was
source layer 1, followed by layer-2 input drift, maxabs <=4.8e-7. First material
input/output differences were layer 4 (reported rel-L1 1.83e-5 / 1.98e-3). Those
rel-L1 values do not define materiality: the tool's existing material classifier
is maxabs >1e-5*max(1,maxabs(reference)). Neither classifier nor acceptance changes.
Later routing divergence follows changed hidden states, not final-logit drift.

The old selector preferred material drift and replayed layer 4 using its B and G
inputs separately. Those were not a common input across modes. Its host checks
reconstructed each mode's own (helper+GPU)+CPU and before+remote; they did not show
cross-mode equality. The V2 evidence archive SHA-256 is supplied as:
ebffeafcc21daf6a1a63f620b5bc001cf59b9fb1f55889cdfb2ce6cf84d24aa3.
This author did not fetch or independently inspect private fleet evidence.

Source inspection found that the old local replay check also started with shared
output, whereas the native vector combine rounds the first weighted product,
uses FMA for subsequent routed entries in order, then adds shared output
(src/kernels/cuda/native_moe.cu, combine_k10_vec4). The non-native fallback uses
double accumulation and adds shared last (shared_expert.cu, moe_combine_kernel).
The offline tool now reports both arithmetic models against the captured actual
local pre-remote output. A model that does not reproduce that component cannot
establish its arithmetic or explain the final difference. Host std::fma is an
arithmetic model, not execution of a CUDA kernel; retained flags and observed
component matches remain relevant, including any fast-math/subnormal differences.

Remote subset reconstruction reuses the real
ExpertWorkerService::accumulate_partial entry-order reduction. CPU/GPU partials
are separately reduced, then the observed host order is (helper+GPU)+CPU, followed
by local+remote on the device. A partition change can change rounding order. This
is a hypothesis until the saved-input reconstruction reproduces the actual delta.

## What the tool now does

The seven absolute path arguments are unchanged. Default selection is explicit:
the earliest BIT-different final output whose B/G MoE input bytes are identical.
It never substitutes a later material boundary with different inputs. It prints
the selected index and requires it in the requested source layer range 1–4 before
reading expert payload. Source indices are zero based and match device-001.trace
through device-004.trace. If no such seed exists, or IDs differ, report that limit;
do not invent an input or force layer 1 from the supplied report.

The scan retains finite/geometry/window/partition/frozen-plan checks and reports
all-layer ID/weight equality. Layers 1–4 add exact hexadecimal routing weights,
IDs, owners/masks and cross-mode local rows/shared/pre-remote/helper/CPU/GPU/remote
components. SELF checks describe one mode; CROSS checks compare B with G.
Different partition components are not expected to be equal merely because their
names match. Inspect routing weights as well as IDs.

At the selected seed it validates the saved actual G GPU request against exact
input, weights, IDs, ownership and returned partial. It reads at most ten routed
expert blobs, one at a time, using existing source/range/digest validation and
the real native expert kernels. The in-memory read subset does not change the
frozen remote plan or placement. Existing single-entry, unit-weight service probes
supply the common native rows, including that reducer's signed-zero behavior.

Those SAME rows are reduced using B owners and G owners. Two clearly labeled
sets separate actual per-mode weights/shared/helper from fixed B weights/shared/
helper. Fixed controls isolate grouping order; if actual weights/shared/helper
are not equal, that counterfactual alone cannot explain the captured difference.
Changed helper membership makes fixed-helper comparison unavailable.

Helper per-expert CUDA rows and individual remote CPU rows were not captured.
The tool anchors each mode's actual helper partial explicitly; it never presents
native surrogate helper rows as observed. CPU subsets are checked using replayed
native rows. It also compares available primary/PCIe rows and actual G GPU rows.
A second, labeled hybrid uses saved actual G GPU rows for BOTH grouping orders,
holding every other native row fixed. This is a diagnostic counterfactual; those
GPU rows were not observed on B's CPU path. No GPU execution is required.

For each model/row set, component predictions are compared with actual captures.
GROUPING_DELTA compares (predicted G - predicted B) against (actual G - actual B)
in double precision from the original floats, including exact-coordinate count,
max observed/predicted delta and max residual. Report matches and mismatches.
An exact final residual with wrong intermediate components may be cancellation;
it does not prove cause or rule out missing helper behavior. A failed component
model is unresolved, not a reason to change kernels or thresholds.

## Complete Hermes procedure

1. Use this immutable revision and GPU-LAYER-GROUPING-V1.SHA256SUMS. Preserve the
   old V2 source/binaries, captures, logits and evidence. Resolve the three owned
   0700 B/G/GPU trace directories, frozen plan, LOCAL_ROOT_C and MANIFEST_C from
   retained RESULT.json/execution-result.json/replay receipts and hash receipts;
   do not guess paths. Verify the existing small-file associations/hashes. Do not
   extract/copy a model, hash full weights, replan placement or refetch evidence
   already retained on 05. No new relays/routes/security settings or credentials.

2. Through the existing approved 05 execution channel, reuse the accepted native
   CPU incremental build/dependencies and a new private CANDIDATE_BIN. No CUDA
   configuration or engine/worker targets. Serial, <=600 seconds build time,
   <=2 GiB build memory and <=1 MiB logs under existing resource controls. Preserve
   the healthy resident arena's headroom and guards. No clean rebuild, dependency
   download, unload or restart if the appropriate build tree is absent; report
   that prerequisite instead.

   ```sh
   cmake -S "$SRC" -B "$BUILD" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$CANDIDATE_BIN" -DSTRATA_BUILD_TESTS=ON
   cmake --build "$BUILD" --parallel 1 --target layer_grouping_test strata-layer-diagnose
   timeout --signal=TERM --kill-after=5s 10s "$CANDIDATE_BIN/layer_grouping_test"
   ```

   Require rc 0 and LAYER_GROUPING_FIXTURES_V1 PASS models=0 cuda=0. The new fixture
   checks seed selection, exact input semantics, a known order-sensitive grouping,
   delta mismatch detection, helper anchoring, shared-last/FMA and float64 models,
   malformed owner and nonfinite delta. Caps are 128 MiB AS / 5 CPU seconds /
   10 wall seconds / 1 MiB output. No timing assertion or old campaign rerun.
   Nothing in this procedure contacts the live worker's inference socket.

3. On fixture PASS, automatically run one guarded offline replay. Reuse the V2
   absolute supervisor/shared-guard stage at
   311561689bb4b1f8a109821601e3a6bda07067e6 with its two existing published patches.
   Do not edit the live guardian's historical files. No controller/agent/observer
   mode is needed. The new offline entry has only --replay-config and reuses the
   existing Session/Job/pidfd admission, host/cgroup/swap/RSS/error/cleanup guards.
   Keep required admission receipts fresh; if stale, refresh through the existing
   approved administrator channel, not by altering the resident service, original
   baseline/identity or age policy. Missing approved authority is a concrete
   admission blocker, not permission to add privileges.

   Create a new private config with diagnostic=GPU-LAYER-GROUPING-V1-20261022,
   diagnostic_revision=<this immutable revision>, offline_only=true,
   supervisor_commit=311561689bb4b1f8a109821601e3a6bda07067e6,
   approved_window_s=120, role=ROLE_CPU_C, fresh run_id/run_root, existing trusted
   namespaces, and the newly built replayer's binary_sha256. replay_argv is exactly
   seven absolute strings, resolved from receipts:

   ```text
   [new_strata-layer-diagnose, B_directory, G_directory, GPU_directory,
    frozen_GPU_plan_on_05, retained_LOCAL_ROOT_C, retained_MANIFEST_C]
   ```

   Execute through the same approved 05 channel:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/layer_grouping_offline.py" --supervisor-root "$SUP_ROOT" --replay-config "$REPLAY_CONFIG"
   ```

   Bound external dispatch to 125 seconds including the existing channel's small
   dispatch margin. The replay grant starts before config/import/admission/hash,
   caps at 120 wall seconds with 15 seconds reserved for owned cleanup, 90 CPU
   seconds, 512 MiB AS/RSS/anonymous, 8 GiB host headroom and 1 MiB output. This
   reads at most 80 MiB expert payload (ten 8-MiB maximum blobs), normally
   52224000 B; only one blob is staged at a time. Trace reads remain bounded by the
   retained <96-MiB format/geometry limit. No resident arena load or weight transfer.

   Require rc 0, complete owned reaping and
   LAYER_GROUPING_DIAGNOSTIC_COMPLETE_V1 numerical_acceptance_not_implied=1 new_GEN=0.
   Also preserve SELECTION, ROUTE_DETAIL, COMMON_GROUPING_CONTROL, per-component
   predictions and GROUPING_DELTA lines. A different selector, missing association,
   unavailable input or failed guard is reported concretely; never repair it by
   launching another capture/model request. Ordinary in-scope offline launcher/
   fixture corrections can be made and reported within this task.

4. Return one concise interpretation: selected seed and input/ID/weight equality;
   layer 1–4 component differences; which local arithmetic model matches captured
   pre-remote outputs; native versus actual local/G GPU rows; both grouping orders,
   controls and predicted-versus-observed residuals; all missing-row/model limits.
   Say “consistent with this observed delta” only when warranted by component and
   control matches; do not promote correlation, final-only cancellation or a
   counterfactual into a proven cause. Preserve NUMERICAL_FAIL and .03 thresholds.

   Historical V2 added 2 requests / 2 tokens / 271.3 s; cumulative usage remains
   17 requests / 835 tokens. Earlier 7541.0 s plus approximately 117 s remains a
   separate approximate history. Report offline build/test/replay elapsed time
   separately; this task adds zero GEN/tokens. Stop/reap only the fixture/replayer
   and their owned supervision. The healthy 111597465600-byte CPU worker/monitor/
   guardian remains loaded under its existing ownership/guards; the 17176473600-byte
   GPU cache remains untouched. No service restart, duplicate, unload, signal,
   threshold tuning, production kernel edit, merge, deployment or new generation.
