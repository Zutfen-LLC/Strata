# PR42 BF16 indexer binding — v1

Additive correction after EOS checkpoint
`12a3e9b938ed0eec8e6253ffbdce3e446fe82b28`. The prior deployed source and failed
real-model results remain untouched. No fleet operation or model change.

The actual saved header contains BF16 indexer Q/K projections at all12 QSA
layers. Pinned qwen4exp loads these tensor roles directly with shapes
[2560,512] and [2560,128] (`src/models/qwen4exp.cpp:225–228`) and uses them as
ordinary projection matrices (lines599/635). There is no requirement to
re-quantize them to Q8. Existing Strata BF16 loading preserves the high16 bits
of the F32 representation (`src/core/weights.cpp`, BF16 conversion comments).

The portable loader accepts BF16 only for those two indexer roles, alongside
its legacy Q8 fixtures and explicit F32 references. It uses the existing
`dequantize_bf16`/`bf16_to_fp32` helper to widen exactly into its sole owned F32
weight buffer. One2560-element row (10240B) is temporary staging. No Q8 or
FP16 conversion is introduced. Resulting projection nodes request F32
precision and execute on the explicitly assigned backend. V340 needs no
native BF16 shader support. Host load conversion is reported separately as
`load / BF16_TO_F32_EXACT / CPU`, not a tensor-operation fallback.

Each layer's raw indexer payload is3276800B; persistent widened storage is
6553600B. Aligned F32 host allocations and actual scheduler copies enter the
existing host/per-device ledger. One-row staging is explicitly included in
host extra admission. No persistent second widened arena exists. Relative
to the old Q8 fixture, the actual layer host allocation grows by4812800B to
84047872B; comparing explicit F32 versus BF16 inputs gives identical storage
and device reservations. Existing workspace bounds cover the largest matrix
and the same graph cap bounds temporary attention/copies.

Passed locally:
- All65536 BF16 patterns widen bit-exactly, including signs, subnormals,
  infinities and NaN payload bits.
- Mixed BF16/Q8/F32 actual-H2560 layer: bit-exact same-backend parity with
  explicit F32 weights through five tokens/block completion and reset;
  weights outside FP16 range; strict role/shape/truncation/budget rejection;
  source-buffer lifetime and callback rollback.
- The same test with strict software Vulkan on llvmpipe: tensor operations
  remain on Vulkan0; only the two declared CPU load conversions occur.
  Host weights84047872B; actual device scheduler55706144B; conversion
  scratch10240B. This is software, not physical AMD evidence.
- BF16 GGUF binding survives closure of the mapping and destruction of input
  BF16 buffers, with exact explicit-F32 reference parity.
- Five focused QSA/BF16/GGUF/full-model/EOS CTests passed in34.51s. The full
  48-layer model fixture now uses the real BF16 indexer roles and retains
  independent EOS, reset/fail-stop, genuine CPU experts and scalar logits.
- Fifteen supervisor tests pass, including exact load-role coverage and
  rejection of CPU tensor work disguised as conversion. The real run requires
  all24 declared BF16 role conversions separately from Vulkan graph work.

The complete saved-header audit is `portable-model-actual-header-audit-v1.md`.
All1224 tensor roles/shapes/types and active metadata match after this fix;
no further static real-binding gap was found. This is not a claim that actual
payloads, physical memory or full-model numerical quality have passed.

Logs: `/tmp/portable-model-bf16-{cpu,vulkan,focused-ctest,supervisor}.log`.
The rebuilt current QSA Vulkan regression passed, including the high-score
spare and width2051 cases. The CPU-only CLI and BF16 fixture rebuilt, and the
CPU-only fixture passed. Logs:
`/tmp/portable-model-bf16-qsa-vulkan-regression.log` and
`/tmp/portable-model-bf16-cpu-only.log`.
The single-retry stages are `portable-model-hermes-v4.md`. No physical run
was dispatched from this task.
