# Opt-in portable FFN coordinator

`STRATA_PORTABLE_FFN=ON` builds `strata_portable_ffn`, a product library for the
fixed-shape T=1–8 qwen4exp **after-attention FFN sublayer**. The default build and CUDA/HIP
session paths do not call it. It is not full-model inference or a new serving
engine.

The runner composes HC normalization/mixing, routing, the shared expert,
unweighted indexed expert rows, weighted merge and HC write-back. It uses pinned
ggml graph operations with a CPU backend and an optional borrowed Vulkan backend.
No llama.cpp model/session library is linked. Callers retain the existing expert
placement and transport policy through `PortableExpertRows`.

## Build and check

Use the existing pinned dependency source, containing `ggml/`:

```sh
cmake -S . -B build-portable -DSTRATA_PORTABLE_FFN=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR=/path/to/retained/llama.cpp
cmake --build build-portable --target portable_ffn_test
ctest --test-dir build-portable -R '^portable_ffn_test$' --output-on-failure
```

The test uses synthetic weights and an independent scalar oracle, including
Q8_0 projections, split GGUF binding and mapping lifetime, routing, repeated
evaluation/reset, failed expert callbacks, CPU fallback and allocation budgets.
It does not validate V340L execution or complete-model logits.

## Product integration boundary

Open the existing model with `GgufModel::open`. `PortableFfn::from_gguf` checks
qwen4exp metadata and copies only these nine tensors for the selected layer:
`hc_ffn_{norm,down,up,inject}`, `ffn_gate_inp`, `ffn_gate_inp_shexp`, and
`ffn_{gate,up,down}_shexp`. Routed expert tensors, attention weights and the
approximately 50.7 GiB Q8_0 PLE table remain untouched by this loader. Views passed
to `create` are copied too; a runner owns its resident dense weights.

Call `reset` with the four-stream residual after attention. `step` publishes a
`PortableRoute` to the callback: hidden-width activation, selected IDs and router
weights. Return one **unweighted** hidden-width row per selected ID, in that order.
The runner applies the weights exactly once, adds the shared expert and writes
back through HC. It commits residual state only after successful, finite results.
Callback failure does not roll back side effects in an external worker; callers
must retain their existing sequence/retry rules. Instances are not thread-safe.

For Vulkan, the caller owns `ggml_backend_vk_init(device)` and passes the borrowed
pointer with `request_accelerator=true`. That backend must outlive the runner and
must not be used concurrently without caller synchronization. The library does
not choose a device, start a worker or alter a listener.

`allow_cpu_fallback=false` rejects unavailable/declined or scheduler-selected CPU
operations when an accelerator was requested. Supported graph nodes are explicitly
assigned to the requested backend, including T=1 operations with host weights;
this does not rely on the backend's automatic batch-size offload threshold.
The runner allocates on a fresh scheduler in one pass. It avoids the size-only
reservation API that resets assignments, and avoids a separate split pass that
could leave temporary copy sources in the owning graph. With explicit fallback enabled,
`placements()` reports phase, operation, tensor type/shape, actual scheduled
backend and fallback reason. Routing/weighted merge are intentionally host work,
reported separately. Reports use the scheduler's actual assignments; strict mode
still rejects any CPU computation in requested accelerator graph work. Metadata-only
views/reshapes retain their source storage and are flagged `metadata_only`; their
consumers receive explicit scheduler copies when crossing devices. They perform
no arithmetic and must not be pinned independently of their backing buffer.

`weight_bytes()` counts aligned owned weights; `graph_bytes()` counts both
schedulers' buffer reservations, including their tensor copies. Before reservation,
a conservative bound prices every node and a full source copy per consumer with
backend padding/alignment and no reuse credit. This can reject a budget that a
tighter liveness plan would fit. The automatic-only size API is unsuitable here
because it resets explicit device assignments. Actual reserved bytes must stay
within the admitted bound. These budgets do not cap driver/pipeline/backend
internal workspace or total process memory. Full-model admission must count
those separately. No throughput or bitwise CUDA equivalence is claimed.

When configured with `GGML_VULKAN=ON`, `portable_ffn_vulkan_test` is an explicit
device test (not an automatically skipped CPU CTest). It requires a real Vulkan
backend with a buffer type incompatible with ggml CPU buffers, asserts actual
MUL_MAT and HC post assignments, executes the graphs, compares scalar parity,
and checks strict mode, reset, callback rollback and budget rejection. A software
Vulkan pass is labeled as such and does not qualify AMD hardware. Use
`--device INDEX --expect-device SUBSTRING` to bind a hardware check to the retained
device mapping; see `agenthandoffs/portable-ffn-vulkan.md` for the bounded 05 run.

## Bounded chunk arithmetic checkpoint (#69)

`create` and `from_gguf` accept a trailing `tokens` argument (default 1, range
1–8). A runner owns exactly one shape and two schedulers; there is no shape
cache. `reset` accepts token-major residuals `[H,HC,T]`, dimension zero fastest.
For `T > 1`, call `step_batch` with `PortableExpertRowsBatch`. The provider is
called once with T routes and returns unweighted `[token,original slot,H]` rows.
`step` rejects a multirow runner. Both entry points publish residuals/routes only
after finite post-graph results. Failed provider work can have external side
effects; an enclosing recurrent model must poison and require reset/replay.

A T1 runner can call `create_chunk_execution(T, error)` for T=2–8. This
additive API shares the existing nine immutable tensors, allocation backend and
weight metadata arena. It performs no weight allocation or copy. Each execution
still owns its residual, routes, counters, two graph contexts and schedulers.
Calls on the pair must be serialized. The borrowed accelerator must outlive both
executions; destroying the original T1 runner does not invalidate the chunk's
weights or CPU backend. Existing factory and T1 step/reset APIs are unchanged.

There is at most one active chunk execution per weight owner. Destroy it before
planning a different tail size; nested/shared T1 clones and per-T graph caches
are not offered. The original runner's `graph_budget` caps the aggregate T1 and
chunk scheduler buffers. Planning the chunk uses that cap minus actual retained
T1 buffers, with the same conservative per-node preallocation check. Failed
planning releases the candidate's buffers and lease and leaves T1 usable. This
retains at most four schedulers and five metadata arenas for a shared pair:
one weight arena plus two graph arenas per execution.

`weight_bytes()` and `memory_usage()` remain inclusive standalone reports, so a
surviving execution continues to report the allocation it retains. For a live
T1/chunk pair, call `combined_memory_usage`: weights and the weight metadata
arena are counted once; actual graph buffers, CPU graph workspaces, host staging
and private device allowances remain additive. Only the already verified
backend-wide conversion high water uses `max`. Both executions' schedulers are
retained, with no token-temporary reuse credit. Unrelated owners are rejected.
These are FFN component ledgers; the future model still must preflight the
combined ledger and attention working state under its host/per-device caps.

The pre graph uses normalization `[H,HC,T]` (RMS over H separately), flattened HC
`[H*HC,T]`, mixed `[H,T]`, router `[NE,T]`, shared gate/up `[FF,T]`, shared output
`[H,T]` and injection `[HC,T]`. Four stream additions, per-token full softmax,
stable top-k, clamp and original slot merge order stay unchanged. The post graph
scatters each token's block into only that token's four streams. There is one
pre graph and one post graph per chunk, not T serial graph executions.
`work()` separates successful `logical_rows`/`completed_chunks` from graph and
provider attempt counts. The scheduler budget prices the actual shape before
allocation and `memory_usage()` includes bounded chunk host staging. The T1
graph topology and its memory allowance remain unchanged.

The private CPU provider's `run_batch` follows native
[`Prefill`](../src/prefill/prefill.cpp)'s expert grouping: validate all routes,
quantize each needed token once, group IDs in ascending expert order and retain
token order inside each `ExpertJobMulti`. Output pointers scatter directly to
original token/slot locations. T≤8 and K=10 bound the list to 80 jobs, below the
existing pool's 96-job scratch limit. Mapped dispatch uses one native pool call.
Verification reuses one slot in the existing ten-blob arena. The staging oracle
uses waves of at most ten jobs, so it never retains 80 weight blobs. Source
metadata invalidation, per-expert digest checks and mapped format offsets follow
the serial provider. Batch counters distinguish logical rows, actual completed
native jobs and maximum `nt`; `native_batches` counts native pool attempts.

`LocalExperts::run_batch` sends one flattened T×K indexed-row request per
participating die and sends only the remaining CPU-cold mask to grouped CPU
work. Frozen ownership, reply validation and original slot scatter remain.
The existing GPU executor computes a single-row graph per selected entry;
**this is not grouped GPU expert arithmetic**.

The CPU/Vulkan FFN fixtures include T1–8 F32 and Q8 projections, true matrix
column/graph-call receipts, callback failures, invalid rows, reset A/B/A and
budget refusal. They compare to the same backend's serial sublayer using the
existing fixture gates: CPU F32 `2e-5`, Vulkan F32 `3e-4`, Q8 `0.003`, each scaled
by `1+abs(reference)`. They print maximum absolute and relative L2 differences
and exact element counts. These small H=32 fixtures cannot establish full-model
numerical parity. Multirow matrix kernels may change reductions on another
backend or geometry. Exact route/slot ordering is required; arithmetic must be
qualified independently without adapting tolerances to observed outputs.
CPU expert tests use H=2560 and compare grouped native rows byte for byte with
serial mapped/staged rows, including shared/disjoint IDs, reordered slots,
CPU-cold masks, source invalidation and the bounded staging waves. The local
adapter tests include fake two-die placement and the real executor with CPU
graph substitution; neither is evidence of GPU execution.

## Layer chunk source checkpoint

`PortableGdnLayer::run_chunk` and `PortableQsaLayer::run_chunk` accept T=1–8
token IDs, the contiguous starting position, in/out residual rows `[H,HC,T]`,
one batch provider and a route vector. QSA uses positions rather than token IDs;
GDN uses IDs for configured PLE predecessors. These additive APIs are not called
by the model or CLI. Existing T1 `step` entry points and their diagnostic versus
production failure policies remain available.

Each layer advances the original T1 attention arithmetic in token order through
a private working cursor. GDN advances recurrence/convolution and PLE history;
QSA appends only the current KV cell and updates pool/spare/tail before that
position's attention. Future cells are never bulk-appended. The resulting rows
feed one actual T-column FFN pre/post graph pair; T1 tails retain the scalar FFN
path. The final row becomes `last_residual`. No attention fusion, precision
conversion or alternative softmax is introduced.

Diagnostic working cursors are bounded copies, and public histories remain at
the last completed call during the provider callback. Production state advances
in place, so any chunk error, including malformed input or budget refusal,
poisons either policy until reset/replay. A failure leaves input/output rows,
committed token count, last residual, diagnostic histories and completed chunk
receipts unchanged; routes are cleared. Production device inspection may show
uncommitted state. There is no device rollback promise. Reset releases the active
chunk FFN and working-state allowance while preserving cumulative chunk totals.

`chunk_work()` records only completed chunks: logical tokens and ordered attention
calls are separate from actual FFN pre/provider/post calls. A T-token completed
chunk has T attention calls and one FFN pre, provider and post call. Failed work
is excluded from these completion receipts; they are not failure-attempt timers.

One shared FFN chunk execution is retained per layer. A different T releases the
previous shape. The FFN's original two-argument factory remains; a new overload
accepts a smaller chunk-only graph ceiling so attention storage/workspace is
reserved before planning the chunk. QSA attention replans include retained chunk
buffers in the aggregate cap. Layer memory reports deduplicate weights, retain
both FFN scheduler pairs and include bounded working history/row staging under
the existing state limits. Failed plans may change the scratch high water, which
remains accounted while poisoned. No cap is raised.

The layer fixtures exercise all T1–8 sizes and tails across nine input positions,
QSA pooling endpoints 0/3/4/7/8, input EOS/image PLE predecessors, exact diagnostic
history, QSA production device inspection through a CPU-selected backend, routes,
serial continuation and A/B/A reset. They check eight actual T-column FFN matrix
nodes and one provider call per chunk. Middle-attention nonfinite input,
provider rejection/throw and an injected FFN post-graph failure must poison with
no completed publication. GDN production's private recurrence is checked through
continuation behavior rather than exposed as a new diagnostic mirror. Numerical
comparisons retain the existing Q8 gate; synthetic CPU observations do not
qualify real-model or Vulkan execution.

The next checkpoint is model and harness integration, following independent
layer review and required compilation/qualification. There is no model prefill
flag, policy binding, whole-model chunk receipt or performance claim here.
Prompt head elision remains the separate serial option. Full-model layer-major
chunking must preflight the combined layer ledgers under unchanged host/per-device
caps, leave the final prompt token/decode/terminal feed on T1, preserve unavailable
logit receipts and add model-level poison/reset, segmentation, requested-logit
and continuation gates. No hardware dispatch follows from this source checkpoint.
