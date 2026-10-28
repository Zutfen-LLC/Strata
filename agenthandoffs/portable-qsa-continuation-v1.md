# Continuation after Vulkan verification — v1

Parent requested a visibility checkpoint. Keep this checkout and builds; do not
restart the implementation. Full scope and results: `portable-qsa-local-results-v1.md`.
Versioned physical recipe: `portable-qsa-layer3-v1.md`. No fleet tests run.

Passed:
- Strict llvmpipe actual placement and complete H2560 layer3 positions0–8,
  independent scalar checks, CPU parity, block-boundary callback rollback/reset.
- Actual width2051 TOP_K with deterministic ties and full retained-cache masked
  attention at cached cells2052/2055/2056; identity cases2050/2051 and a strong
  two-cell attention regression. All tensor work stays on Vulkan.
- GGUF actual Q8_0/F32 binding, lifetime after model close, metadata/type and
  unsupported RoPE/scaling rejection.
- Eight CTests in CPU-only and Vulkan-linked builds; existing PLE, GDN and
  FFN Vulkan executables sequentially. Default-OFF target gating.

The original 12-column batched Vulkan attention gave wrong logits for heads8+
with two cached cells. Final explicit per-head F32 products fix the original
case and use FP16 retained KV. No CUDA, shared scheduler, existing FFN/GDN/PLE
source changed. CPU plan scratch and a conservative Vulkan tensor-workspace
reserve are priced. Driver pipelines/other borrowed-backend users excluded.

Final code is verified locally. An added test incorrectly equating actual
allocation with conservative admission failed and was removed; existing
budget rejection tests remain. The final corrected CPU suite passed, and the
Vulkan-linked QSA CPU executable was rebuilt/rerun after that correction.
No outstanding numerical or placement failure remains.

Useful logs:
- `${BUILD_ARTIFACT_87C31802}` — final strict QSA result.
- `${BUILD_ARTIFACT_4F449316}` — final eight CPU CTests.
- `${BUILD_ARTIFACT_F7B092C1}` — eight Vulkan-linked CTests.
- `${BUILD_ARTIFACT_C2CA6777}` — rebuilt QSA CPU test after
  removing the invalid admission-bound test.
- `${BUILD_ARTIFACT_1A2F734F},gdn,ffn}-regression.log` — strict existing device tests.
- `${BUILD_ARTIFACT_494CC583}`, `${BUILD_ARTIFACT_33C87A18}`.

Builds: `${BUILD_ARTIFACT_FF6747B2}`, `${BUILD_ARTIFACT_4336C66D}`,
`${BUILD_ARTIFACT_1A5228F7}`. Source checkout is task-7/strata.

Remaining before final delivery: inspect full final diff, verify new source
manifest, publish draft stacked on `codex/portable-ple-layer`, attach its URL,
and report the exact pushed head. Parent then reviews and owns physical tests.
The previously pushed checkpoint was
`e651ecdf94f20fb51d837ab0b6b10142e3f2aab6`. No PR exists yet.
No denied action or pending approval. Initial sandbox DNS failure was resolved
by the ordinary approved network push; no fleet credentials/access were used.
