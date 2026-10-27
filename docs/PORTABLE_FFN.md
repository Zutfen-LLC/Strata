# Opt-in portable FFN coordinator

`STRATA_PORTABLE_FFN=ON` builds `strata_portable_ffn`, a product library for the
T=1 qwen4exp **after-attention FFN sublayer**. The default build and CUDA/HIP
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

## Direct end-to-end follow-on

The next product slice attaches this boundary to existing local CPU/Vulkan expert
dispatch, adds the attention half with persistent GDN and QSA state plus PLE at
its configured layer, then embedding/head and the all-trunk-layer loop. Validate
teacher-forced logits, selected experts and state reset against the current
product before full-local 05 measurement. Dense Vulkan placement and both dies'
actual execution must be recorded. Keep the current CPU-process proof first in
fleet runtime order; no hardware dispatch follows from enabling this library.
