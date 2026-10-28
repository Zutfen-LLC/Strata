# Complete portable GDN layer 0

`STRATA_PORTABLE_GDN=ON` adds `strata_portable_gdn` on top of the opt-in portable
FFN library. This executes one complete qwen4exp layer 0 for one token at a time:
HC attention mixing, Q8_0 projections, four-tap convolution, recurrent gated
delta net, RMS normalization with **sigmoid** output gating, HC write-back and
the existing FFN/expert-callback path. It is product code, not a separate engine.

Only the actual model geometry is accepted: hidden 2560, HC 4/rank 320, FF 640,
512 experts/top 10, state width 128, 16 key heads, 48 value heads, convolution
channels 10240 and inner width 6144. Other layer indices are rejected. The GGUF
entry point verifies qwen4exp metadata and the configured PLE layer list `[1]`;
it cannot silently ignore a PLE operation at layer 0. QSA and PLE execution are
not implemented in this slice.

## Calling the product layer

`PortableGdnLayer::from_gguf` binds only the selected layer's thirteen attention
and nine FFN tensors through the existing split reader. It copies no routed
expert, embedding, output-head or PLE-table payload. `create` accepts the same
raw tensor views for model-independent validation. Both own their copied weights.

Each `step` takes a fresh four-stream residual for the next token and the same
unweighted indexed expert callback used by `PortableFfn`. Successful calls update
convolution history, recurrent state, last residual and token count together.
Callback rejection, exceptions, invalid rows and nonfinite input/results leave
those public persistent values unchanged. Reset clears all state, including the
private FFN residual. Instances and borrowed backends require caller serialization.

The first implementation keeps canonical recurrent state/history in host memory.
Each token copies it into graph inputs and reads candidate state back before the
FFN callback and commit. This favors a clear rollback contract; it is not a claim
that the transfer cost is optimal. A GPU-resident transactional state scheme is
an optimization after complete-model correctness, not implemented here.

The reviewed placement/allocation policy is shared in one internal helper with
the FFN runner. Supported computation is explicitly pinned, metadata-only aliases
follow their storage, strict mode forbids CPU computation fallback, and all graph
evaluation completes synchronously before host reads. Reports include both
attention and FFN phases. Weight and graph budgets apply to their combined owned
allocations. A separate host-state budget admits the committed/candidate copies,
residuals and optional trace scratch. Driver/pipeline/internal workspace, caller
input/weight views and callback memory remain outside those budgets.

## Focused validation

```sh
cmake -S . -B build-portable -DSTRATA_PORTABLE_FFN=ON \
  -DSTRATA_PORTABLE_GDN=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_GGML_DIR=/path/to/retained/llama.cpp
cmake --build build-portable --target portable_gdn_test portable_ffn_test
ctest --test-dir build-portable -R '^portable_(gdn|ffn)_test$' --output-on-failure
```

The CPU fixture uses actual-size, synthetic sparse Q8_0 projections and F32
norms, router, convolution, decay parameters and shared-expert scalar gate. It independently computes convolution,
Q/K normalization, head broadcast (`value_head % 16`), decay, delta update and
state/output from captured intermediate vectors. State/output tolerance is
`3e-6 * (1 + abs(reference))`; convolution/normalization use `1e-5`/`2e-5`.
Rollback is checked exactly. A synthetic GGUF test closes its source mapping
before execution. No real model weights or real expert worker are tested.

With `GGML_VULKAN=ON`, the explicit `portable_gdn_vulkan_test` target asserts
device assignment of GATED_DELTA_NET, SSM_CONV and all eight attention matrix
operations, then runs multiple tokens, reset and rollback. It checks the same
independent recurrence and compares complete-layer residuals and route weights
against CPU within `0.003 * (1 + abs(reference))`, with identical selected IDs.
It requires a distinct Vulkan buffer type; a missing device is a failure.
Software Vulkan evidence and physical-device evidence must remain separate.

## Path to complete local inference

Next implement configured PLE before layer 1, then QSA attention and its indexer/KV
state before layer 3. Extend layer selection only when those operations are
supported; do not skip them in an all-layer loop. Add embedding and final HC/head,
then run the full trunk with the existing expert dispatch. Validate actual-model
teacher-forced logits, routing and reset before measuring full-local AMD inference.
There is no full-model serving path, optimized prefill, MTP or simultaneous
dual-die orchestration in this PR.
