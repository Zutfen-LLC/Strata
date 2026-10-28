> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Portable model: assembled CPU product checkpoint

Continue the existing task-8/strata worktree and branch
`codex/portable-model-session`; do not restart or edit other task worktrees.
Base remains corrected PR41 `1d8eae1c3906962932b77e366f324c51aa89819d`.
Previous checkpoint was `22137156a9eb232858a72fa30da1012c03aa0e11`.

Implemented:
- `PortableModelSession::open/step/generate/reset` and opt-in `portable_model`
  executable. Six-shard lifetime ownership; header, directory, tensor
  shape/type/extents and overlap admission. MTP payloads are not executed.
- Fresh host Q8 embedding broadcast on every token, all 48 GDN/QSA layers,
  configured host PLE, portable global HC and output.weight head.
- Real CPU provider using native_fmt/native_quant_act and
  ExpertPool::run_split_multi_native. Validated standard GGUF Q8 role staging,
  bounded 52,224,000-byte arena, retained selected-expert digests, unweighted
  rows in route order, PortableFfn as sole weighted reducer. No RPC/listeners.
- Strict accelerator placement/fallback policy inherited by every module;
  contiguous two-device split defaults to 32 to leave room for final head.
- Actual scheduler allocations queried separately per backend; explicit
  conservative CPU/state/metadata/native-backend workspace allowances and
  separate die caps. Constructor preflights each module, then reconciles
  actual buffers plus bounds. No pooled VRAM or arbitrary 2 GiB reserve.
- Poison on any token-step failure; explicit reset/replay; EOS/context and
  max-new stops; stable greedy lowest-ID ties. Raw full-vocabulary F32 dumps
  for every consumed token and final placement TSV.

Read `docs/PORTABLE_MODEL.md` for the interface, build/run commands, memory
ledger limits, identity coverage and numerical limitations.

Observed CPU evidence:
- Initial complete sparse six-shard model test passed. All 48 layers, actual
  vocabulary248064, layer1 PLE, real nonzero native Q8 CPU expert, independent
  scalar final logits, fresh-token HC, corrupt selected expert in layer10,
  poison/no retry/reset/replay, EOS and context-bound greedy generation.
  Mapped unique files:188194264224B; anonymous copied weights:4773486080B;
  staging:52224000B. No real model bytes or fleet access used.
- New independent Q8 global HC/head scalar test passed (nonuniform streams,
  nonzero down/up gates and output projections; predetermined synthetic Q8
  screening tolerance). Existing QSA/GGUF, GDN, PLE and FFN tests passed.
- Seven targeted CTests passed in35.94s at the earlier complete local build.
  The final rebuild/recheck after formatting, CLI integer validation and an
  additional 4MiB GDN metadata charge passed all seven in35.54s.
- CPU CLI --help runs; default-OFF configure completed. Default target list
  is `${BUILD_ARTIFACT_6FF2683D}`.

Build: `${BUILD_ARTIFACT_83FD5685}`, CMake
`/tmp/cmake-3.31.6-linux-x86_64/bin/cmake`, ggml checkout
`/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d`.
Logs:
- `${BUILD_ARTIFACT_A2311B57}` (first complete CPU result)
- `${BUILD_ARTIFACT_19B5E850}`
- `${BUILD_ARTIFACT_8DAABEEB}` (seven passed)
- `${BUILD_ARTIFACT_25D3B334}`
- `${BUILD_ARTIFACT_90A058D2}`
- `${BUILD_ARTIFACT_AC83AB99}`

Remaining for the next bounded checkpoint/final draft publication:
1. Read final complete diff; add rejection/memory checks where needed.
2. Build the new head/CLI in a fresh Vulkan build using the pinned dependency;
   run strict local software-Vulkan head tests and prior FFN/GDN/PLE/QSA
   regressions. The assembled new model has NOT yet run on Vulkan; do not
   claim device or physical AMD success from this CPU evidence.
3. Verify default target gating and unchanged CUDA defaults. No CUDA/HIP/SYCL
   code is linked or modified by this path.
4. Produce versioned operator commands with the exact source head, retained
   SWLOCAL1 manifest SHA and concrete model shard paths. Retained manifest:
   `${PRIVATE_ARTIFACT_32CC92CE}`, SHA256
   `8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318`.
   Its three file entries cover expert shards4–6, not all six. The parser and
   selected-expert digest path reuse that actual format and coverage.
5. Publish one draft PR stacked on `codex/portable-qsa-layer3`, attach URL,
   and report exact pushed head/tests/limits. No PR published yet.

Narrow parent inputs for final real-model recipe (not implementation blockers):
- Any global/MTP tensor-name inventory subset from existing header artifact
  F0CCSCBS0H2, to check unused inventory names against the real model.
- Only additional retained02 teacher-forced logit positions beyond the supplied
  first-generation row, if already available. Existing compare_logits.py fixed
  limits remain unchanged; parent owns numerical interpretation. Do not tune
  limits against candidate.

Parent-supplied concrete runtime/reference inputs (verified elsewhere, not
locally reverified by this task):
- Root05: `${PRIVATE_ARTIFACT_9AA3D2B0}`.
- Shard basename `Qwen3.8-Flash-Next-Q8_0-0000N-of-00006.gguf`, N1..6.
  Stat sizes in order:10946624,682434912,54400261312,49446841216,
  49668930400,34015618784 bytes. No new payload hashes.
- Actual model vocabulary248320. The synthetic test's248064 is explicitly
  synthetic; the product reads vocabulary dynamically from token_embd.weight.
- Actual retained02 LOCAL prompt IDs:
  `248045,846,198,7734,264,12654,436,34810,6297,523,1608,51425,11,440,8390,10206,11,615,321,2113,5245,11,321,75555,13,3301,1970,1132,13,248046,198,248045,74455,198,248068,271,248069,271`.
  Use context64 and a few generated tokens for the first smoke, not the
  original GEN256 request budget.
- Reference logits02:
  `${PRIVATE_ARTIFACT_91D2F803}`,
  993280 bytes, SHA256
  `4143957a612227b67ed3c47ce930365e5e8249d03ef77ee1442ca4722c33fa96`.
  Parent/operator keeps logits host-local. Never upload them or weights to Git.
- First16 retained generated IDs:
  `[71093,12305,198,1445,14856,1120,51425,1358,1005,43072,5287,1714,25,198,262,687]`.
- Actual local argv is
  `${PRIVATE_ARTIFACT_8D7D2A05}`.
  NDon input config contains distributed options and is not the actual LOCAL
  argv. Coordinator binary `${PRIVATE_ARTIFACT_FA7549CC}`, SHA256
  `931569c089a4f04114395a428c85f85227142c733f4fa0511243d05f97b46b89`.
  Binary filename alone does not prove source identity.

No fleet access, installation, merge, deployment, security changes, spending,
or destructive cleanup. No denied/pending/uncertain operation in this turn.
