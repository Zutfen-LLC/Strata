# Moving QSA spare-key correction — v2

This additive correction supersedes v1's zero-spare assumption. The exact
candidate is identified by the PR head and `portable-qsa-source-v2.sha256`.
Existing CUDA behavior, FFN/GDN/PLE sources, state policy and placement policy
are unchanged. Parent/Hermes owns corrected physical validation.

## Source contract and fix

Pinned upstream `3cf03257f219afbe7334045ff7c6a06ac68c627d`,
`src/llama-memory-hybrid-idx.cpp` lines322/332, zero-initializes block-member
indices and positions; line511 maps incomplete cells to that spare block.
`src/models/qwen4exp.cpp` lines609–632 consequently gathers raw cell0 four
times, averages, applies weighted RMS and RoPE at0. Strata's
`include/strata/kernels/qsa.hpp` lines200–226 states the same contract.

The first successful token now retains that normalized first raw key as the
last pooled row. Every block completion moves it to the new spare row while
replacing the old row with the newly completed block key. Initial construction
and reset remain zero before any successful token. First-token initialization,
block completion and the spare commit only after successful finite FFN output.
The already-scheduled pooling graph computes the first-key normalization;
no new backend or host normalization path is introduced. Extra candidate
scratch is within the existing host-state allowance. No maximum-capacity copy.

The scalar oracle independently normalizes the first captured raw key, checks
the spare after every position0–8 (including both block completions), and
checks ordinary tail scores as ReLU dot sums plus finite1e9 within one F32 ULP.
Callback reject/throw/malformed/nonfinite rollback now includes token0 as
well as the completion/following-token cases. Retry and reset still compare
to an uninterrupted control.

The 2055-cell high-score fixture obtains the real coordinator's committed
spare after one synthetic projection. It uses the unchanged production scoring
builder and actual TOP_K width2051, without thousands of weight projections.
Its first raw key is positive e64 and finite F32 norm weights give a spare
near e64*1e5. Query head0 is e64*1e5; other heads are0. All513 completed keys
are e64*2e4 + e65*sqrt(1e10-4e8). Completed scores are2e9, true tail scores
exceed1e10; zero spare gives only1e9. Correct membership is completed0–2047
and tails2052–2054. Exact membership, the first-key normalization and tail
scores are checked. The scoring helper extraction preserves the existing
ReLU-per-head/sum/expansion/bias expression and shares it with the regression.

## Observed RED before the production fix

Only corrected tests/oracle and score-builder extraction were applied:

- CPU `portable_qsa_test` failed:
  `high-score spare retains tails2052-2054 and complete0-2047 mismatch at 2048 actual=1.000000 expected=0.000000`.
- Strict llvmpipe `portable_qsa_vulkan_test` failed at token0:
  `completed and spare keys mismatch at 0 actual=0.000000 expected=0.506630`.

Logs: `/tmp/portable-qsa-spare-red.log` and
`/tmp/portable-qsa-spare-red-vulkan.log`. No tolerance was relaxed to pass.

## Observed GREEN after the fix

Same retained Linux/GCC16.2.1/CMake3.31.6/ggml/llvmpipe environment and isolated
builds as v1. Exact build commands:

```sh
/tmp/cmake-3.31.6-linux-x86_64/bin/cmake --build /tmp/strata-portable-qsa-build \
  --target portable_qsa_test portable_qsa_gguf_test --parallel 2
/tmp/cmake-3.31.6-linux-x86_64/bin/cmake --build /tmp/strata-portable-qsa-vulkan \
  --target portable_qsa_vulkan_test portable_qsa_test portable_qsa_gguf_test --parallel 2
/tmp/cmake-3.31.6-linux-x86_64/bin/ctest --test-dir BUILD \
  -R '^(portable_(qsa|qsa_gguf|ffn|gdn|ple)_test|gguf_(reader|split|geometry)_test)$' \
  --output-on-failure
GGML_VK_VISIBLE_DEVICES=0 VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  timeout 180s /tmp/strata-portable-qsa-vulkan/portable_qsa_vulkan_test \
  --device 0 --expect-device llvmpipe
```

Both CPU-only and Vulkan-linked builds passed all eight CTests. The strict
software-Vulkan executable passed corrected positions0–8 scalar checks,
CPU parity, first-token/block-boundary rollback/retry/reset, the new2055-cell
high-score case, and every existing selection/attention boundary fixture.
Observed new receipt:

`PASS high-score spare cached_cells=2055 width=2051 tails2052-2054 complete0-2047`

All scheduled work, including the new high-score scoring/TOP_K graph, has
actual strict Vulkan placement. Existing cached-cell checks still report
`MUL_MAT=48`, actual TOP_K at2052/2055/2056 and width2051. Whole-layer accounting
is unchanged: weights79,235,072B; peak graphs/workspace190,338,372B;
host-state bound34,973,984B at capacity9.

Logs: `/tmp/portable-qsa-spare-green-cpu.log`,
`/tmp/portable-qsa-spare-green-vulkan-ctest.log`, and
`/tmp/portable-qsa-spare-green-vulkan.log`.

## Old-head physical evidence and remaining limits

Parent supplied Hermes receipt1793142605.492489 in thread1793142298.778149:
old head `f1cdabda2212f8c6a4679af3435ac107c698eb1f` passed the original synthetic
QSA suite on both physical V340 dies, sequentially, with48 source checks,
ggml pin, eight CPU tests and cleaned build/test processes. Evidence on05:
`/home/hermes/portable-qsa-pr41-f1cdabd-evidence`. This proves those old
fixtures/pipelines execute on the hardware; it does not resolve or qualify
the moving-spare semantics. No deployed source was changed or test interrupted.

Corrected physical V340 validation is pending. The new small versioned recipe
is `portable-qsa-layer3-v2.md`; require the new high-score receipt on both dies.
No fleet workload, installs, real model weights/callbacks, model assembly,
quality, performance, CUDA/HIP/SYCL compilation or merge was performed here.
Keep the PR draft pending review and corrected physical evidence.
