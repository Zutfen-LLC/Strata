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

This checkpoint is not wired into `PortableModelSession`, the CLI or harness.
There is no new prefill flag, changed default, whole-model chunk receipt or
performance claim. Prompt head elision remains the separate serial option.
Before layer-major model integration, split the GDN/QSA attention transaction
into private ordered working cursors and completed-chunk publication. GDN must
advance recurrence, convolution and PLE predecessors in order; QSA must advance
KV, pool, spare and tail without exposing future cells. Preserve T1 rollback;
failed chunks poison until reset/replay. Retain one immutable FFN weight owner
per layer with bounded T1/chunk scheduler reuse rather than constructing extra
full-weight runners. Preflight those working states/schedulers under unchanged
model caps before enabling the chunk option or binding it in the harness.
Mixed segmentation, attention-history/EOS boundaries, model poisoning and
requested-logit/continuation checks belong to that next integration checkpoint.
