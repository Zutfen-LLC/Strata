# Portable QSA layer 3 — implementation checkpoint v1

Historical implementation checkpoint. Superseded by
`portable-qsa-local-results-v1.md` and `portable-qsa-layer3-v1.md`.
The notes below describe the original checkpoint, before Vulkan corrections. Base: PR40 head `c3625a290066a50e397a32cdab8040038acc0b62`,
branch discovered from PR40: `codex/portable-ple-layer`. New branch:
`codex/portable-qsa-layer3`. No prior commits rewritten. No fleet operations.

## Scope and implementation

Default-OFF `STRATA_PORTABLE_QSA`, depending on portable FFN, adds
`PortableQsaLayer` for exact H2560, HC4/rank320 layer 3, one sequence and T=1.
Text positions must start at zero and increase by one. The layer consumes a
fresh four-stream residual and uses the existing `PortableExpertRows` callback.
Embedding/head and the full layer loop remain outside this slice.

The source was checked against retained upstream qwen4exp.cpp at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, especially lines 488–859, and
Strata's `include/strata/kernels/qsa.hpp`. Fixed projection/pooling/combine
graphs and an active-history-sized attention graph use the existing explicit
scheduler policy. Full K/V retain FP16; raw indexer tail and completed pooled
keys are F32. A zero spare/dead pooled key persists. Every public state change
is deferred until successful FFN completion and finite output. Candidate
history copies contain active cells only, not maximum capacity.

Generic pinned CPU/Vulkan sorts have unspecified tie ordering. The selection
graph uses TOP_K solely to recover the cutoff score, then counts strictly
greater scores and uses CUMSUM of equal scores to select ascending cell IDs.
No score perturbation. Attention uses a 0/-infinity mask and explicit
query-head/12 grouping. It reads full cached K/V rather than compacting the
selected cells into a separate cache.

## Observed local validation

Linux x86_64, GCC 16.2.1, CMake 3.31.6, retained pinned ggml. CPU library and
`portable_qsa_test` compiled; the executable passed actual-dimension full
layer positions 0–8 with scalar checks of layout, weighted RMS, rotation,
raw pooling, scores, full FP16 KV, grouped attention, sigmoid gate and HC
combine. Failure/throw/malformed/nonfinite callbacks at block completion and
following token leave public state unchanged. Retry, reset, capacity,
budgets, unavailable-backend strict rejection and fallback records passed.
Focused selection cases 2050/2051/2052/2055/2056, three score patterns each,
passed exact membership checks against scalar lexicographic selection.

Build directory: `${BUILD_ARTIFACT_FF6747B2}`.
Configure log: `${BUILD_ARTIFACT_E6BF5131}`.
Latest build log: `${BUILD_ARTIFACT_6A7A0739}`.
The build and executable passed again after clang-format.

Configure used the settings from `portable-ple-local-results.md`, adding
`-DSTRATA_PORTABLE_QSA=ON` and selecting the new build directory. No installs.

## Required next work before publication

- Add isolated Vulkan executable with actual placement checks, full layer
  CPU parity/scalar checks, rollback and focused TOP_K width2051 fixtures.
  Use a fresh build directory, retained SPIR-V include paths, explicit
  llvmpipe device selection. Do not claim physical support. A supported-op
  probe alone is insufficient. Preserve fallback and strict rejection.
- Add GGUF metadata/binding fixtures, including every required RoPE/scaling
  field and actual Q8_0/F32 names/types. Current from_gguf is implemented but
  untested. Inspect legacy RoPE scaling aliases/extensions before claiming
  unsupported scaling is fully rejected.
- Strengthen independent retry/reset control comparisons, bad inputs/types,
  and nonfinite/overflow cache cases where useful. Current projection scalar
  checks permit activation quantization rounding; attention/indexer checks
  use tight tolerances over projected trace inputs.
- Run existing FFN/GDN/PLE and GGUF regressions in CPU and Vulkan-linked
  builds, verify default-OFF target gating and unchanged production sources.
- Inspect budgets carefully, particularly scheduler/metadata/active scratch
  accounting and backend workspace. No all-capacity preallocation copy.
- Read full diff, add a versioned small operator recipe and source manifest,
  commit, push, and publish a draft stacked on `codex/portable-ple-layer`.
  Attach its URL. Keep in draft pending independent review and parent-owned
  physical tests. No merge, deployment or fleet access.

No denial, pending approval, or uncertain external mutation occurred. No
external mutation has been attempted; branch and implementation are local.
