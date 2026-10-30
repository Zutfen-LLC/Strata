# Opt-in native-method router in the portable FFN/model

`portable_model --native-router PRIVATE_RECEIPT_PREFIX` selects the native
512-expert/top-10 routing method for every FFN. It is OFF by default. The default
host `route_logits` arithmetic and default CLI output remain unchanged.

This is methodology parity, not NVIDIA bit identity. The unchanged strict
component comparison still fails with 21/80 differing weight words and matching
80/80 IDs. The retained row-2 diagnostic at private agenthandoffs commit
`306b3710058bd20a601b073dd06045f04eeb98c0` preserves every final output/view and
has zero sum/product replay discrepancies on both devices. Five same-input exp
outputs differ by one ULP. Both reciprocal inputs differ, so that experiment
provides no independent same-input reciprocal result. This is one synthetic
row, not global correctness, full active-resource qualification or performance
evidence. No exp emulation, formula replacement or comparator tolerance change
is made. FP32 denorm behavior remains unqualified.

## Interface and execution

`PortableFfnOptions::native_router` requires the native-stage build, 512 experts,
K10, T1..8, a requested accelerator, no CPU fallback and a nonempty private
receipt prefix. The real component verifies a Vulkan backend/device and its
existing arithmetic feature/control admission. Model options expose the same
boolean and independent `native_router_receipt_prefix`. They require explicit
accelerators and strict execution. No expert placement policy is selected by
the router option; existing expert plans and CPU misses retain their ownership.
The CLI option takes a prefix value, unlike the separate native CUDA program's
boolean switch. A build/device/shape/receipt refusal never switches to host routing.

After the existing synchronous pre graph, the component consumes the live
contiguous F32 `[T,512]` logits tensor directly. There is no logits upload,
host exp, host double oracle, sorting or weight normalization on this path.
The component writes I32/F32 `[T,10]` outputs. Host readback validates every ID's
range/uniqueness and each weight's finiteness/nonnegativity, with at least one
positive weight per row. It then supplies raw original-order IDs/weights to the
existing expert callback. This validation rejects malformed results without
modifying a valid weight. Activations still use the existing host provider
boundary. Shared projections/gating, expert placement/cache limits and optional
native combine are unchanged. Router and combine options are independent.

Each T1/chunk execution owns its two output tensors, component and status
buffer. The original graph budget covers their actual aligned bytes before
scheduler admission, with each subsequent component receiving only the
remaining cap. Actual component/status bytes and outputs enter `device_graph`;
a conservative 24 KiB metadata/owner allowance plus bounded readback/route
storage enter `host_extra_bound`. Shared T1/chunk reports count their immutable
weights once and both router resources. Chunk lease and serialized-backend
rules are unchanged. Each owner atomically reserves a new
`PREFIX.layer-L.t-T.owner-N/effective.spv`; collisions are skipped, including
across recreation/replanning, and existing receipts are never overwritten.

The component does not retain the borrowed logits tensor after a call.
Exceptions unwind owners/buffers by RAII. Dispatch/result failure leaves FFN
residuals and public routes unchanged; the model's existing poison/reset rules
apply. A failed GPU operation does not acquire a new recovery guarantee from
this integration.

## Participation evidence

A `PortableRoute::native_router` tag is set only after component execution and
complete output validation. `PortableFfnWork::native_router_calls/rows` publish
only after the FFN post graph succeeds and survive reset. The model stages
actual route tags in its provider callbacks, requires them to match the requested
mode (no silent host fallback), and commits `PortableModelWork` router totals
only after the entire forward including its requested head succeeds. Partial
or poisoned forwards receive no completed-forward credit. One T1 call credits
one call/row; a T2..8 batch credits one call and T rows per layer.

Only when the option is enabled, stop/done/timing JSON adds
`native_router_method=true`, `router_numerical_parity=false`,
`native_router_calls_delta` and `native_router_rows_delta`. The placement report
adds `NATIVE_512_TOP10_METHOD` on the selected backend. These are participation
receipts, not accuracy or acceleration qualification. For a successful serial
T1 run consuming P prompt and G generated tokens, expect `48*(P+G)` completed
router calls and rows. For chunks, rows still count every consumed token but
one layer chunk is one call. Default-off totals are zero and default CLI fields
are unchanged.

## Verification

The real FFN coordinator is compiled against mocked router/combine components
for T1/T2/T8. Tests cover actual pre-graph logits/token association, raw nontrivial
rank/weight forwarding without renormalization, valid zero weights, original
expert slot consumption, router/combine coexistence, successful/failed
participation, unavailable-build and strict option refusals, duplicate/out-of-
range IDs, nonfinite/negative/all-zero weights, component/provider failure,
nonfinite rows, residual/route rollback, retained receipt collisions, owner
cleanup, chunk sharing and aggregate memory accounting. Mock arithmetic is not
a CUDA/Vulkan oracle. The model participation helper rejects mode/tag mismatches
and stages completion credit. Existing full-model synthetic CPU regression
checks default-zero router work and model option preflight refusals.

Passed locally: portable model and affected CPU/Vulkan libraries/executables
build; ten focused router/FFN checks (including the new unavailable-build test)
and the existing synthetic full-model CPU test pass. Native-stage/trace shader
checks remain passing and baseline router shader arithmetic is unchanged.
Not run: real-model router integration, an OFF/ON fleet comparison, or new
CUDA/HIP/SYCL hardware tests. No merge or default change is requested.

## One bounded same-prompt functional comparison

Use the same reviewed binary/commit, model/manifest, original case-1 complete
25-token prompt, GEN3/context256, threads/workers, strict private SDK/ICD,
dense topology/residency switches, frozen expert plan and ALL caps/reserves in
both fresh-process arms. Keep existing Q8 observation and native combine choices
identical. Only router selection changes; receipt/logit/placement/timing paths
must be fresh for each arm. Use serial T1/full-head execution for this comparison
(no new chunk/head policy). Retain existing operator preflight/supervision and
its verified external timeout. Stop on refusal/failure; do not change caps,
plans, fallback, formula or tolerances to obtain a passing pair.

Build the native-stage portable model once and run focused host tests. The
private handoff supplies a non-executing pair preparer that records one frozen
common argv/hash, plan hashes and binary identity, creates separate output paths,
and adds the router flag only to ON. Review both argv files before dispatching
exactly one OFF process then one ON process through the existing supervisor.
A timeout wrapper must preserve actual child status and supervise the owned
model process/descendants as required by the operator; this recipe does not
claim complete telemetry coverage from sparse samples.

For each arm the effective command is the existing frozen generation command,
with these per-arm evidence arguments:

```sh
timeout "$VERIFIED_FORWARD_TIMEOUT" "$BUILD/portable_model" "${FROZEN_COMMON_ARGS[@]}" \
  --expert-native-q8-observation "$NEW_ARM/expert" --native-moe-combine \
  --expert-record-routes --placements "$NEW_ARM/placements.json" \
  --dump-logits "$NEW_ARM/logits" --timings "$NEW_ARM/timings.json" \
  "${ROUTER_ARGS[@]}" >"$NEW_ARM/stdout.jsonl" 2>"$NEW_ARM/stderr.log"
# OFF: ROUTER_ARGS=()
# ON:  ROUTER_ARGS=(--native-router "$NEW_ARM/router")
```

`FROZEN_COMMON_ARGS` includes the exact common model/manifest/tokens, GEN3,
context256, device topology, plan and caps, but excludes the evidence arguments
shown above. Both arms use native combine here to match the retained f023
short-generation configuration; it is not required by the router API itself.

Report process success, consumed/generated token IDs, greedy-next decisions,
stop reason, router participation, actual GPU/CPU expert entries, memory
admission and placement/receipt counts. Compare route IDs and raw logits on the
shared consumed-token prefix. After generation histories diverge, later logits
are conditioned on different tokens and cannot isolate router arithmetic.
Report exact differing-word counts and descriptive max-absolute differences
without inventing a new acceptance tolerance. Equal generated tokens do not
prove numerical parity; changed tokens are a measured functional effect, not
a reason to hide the component's failed strict comparison. Keep real model,
activation, logit and token payloads private.
