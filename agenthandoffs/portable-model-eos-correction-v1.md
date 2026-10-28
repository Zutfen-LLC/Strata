> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# PR42 independent EOS semantics correction — v1

Additive correction after the first real-model launch at
`602e3fdf8f3f9981ee361e060b97f0746f8a0ee9` stopped during admission with
`portable model: tokenizer and PLE EOS metadata disagree`. Parent/operator
reported no layer execution or token/logit. The failed evidence at
`${PRIVATE_ARTIFACT_F5058E02}` remains untouched; this task
performed no fleet operation and launched no real-model retry.

## Root cause and source verification

The portable session wrongly required two independent metadata fields to be
equal. The actual artifact intentionally declares:

- `tokenizer.ggml.eos_token_id = 248046` for generated end-of-turn stopping;
- `qwen4exp.ple.eos_token_id = 248044` for PLE predecessor-history cuts;
- tokenizer BOS/padding IDs 248044 and vocabulary size 248320.

Pinned llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` reads PLE
EOS into `hparams.ple_eos_token_id` in `src/models/qwen4exp.cpp:88` and uses
that field at lines 1083–1108 to cut predecessor hash context. Current-token
EOS does not cut its own context. `src/llama-vocab.cpp:2590` independently
reads tokenizer EOS, with its key mapping in `src/llama-arch.cpp:389`.

Existing Strata `src/kernels/ngram.cpp:87` cuts predecessors using its validated
PLE constant 248044. `src/core/portable_ple.cpp` validates that PLE metadata
configuration separately. Existing product generation in
`src/program/generate.cpp` uses its independent configurable `eos_ids` on
produced tokens (default set at lines 731–733; output checks at line 11816),
not an equality with PLE metadata. Those production defaults are unchanged.

Removing only the equality would expose another same-path error: the
portable prompt loop previously stopped when an input token equalled the
configured tokenizer EOS. The retained 38-token prompt contains 248046 at
position 29 as an internal turn separator, so it must be consumed along with
the remaining assistant-prefix tokens. Generation stop checks apply to
produced tokens, not prompt input.

## Correction

Both metadata fields retain independent in-vocabulary integer checks; float,
bool, missing and out-of-range values cannot become token IDs. PLE's existing
configuration validation and n-gram math/history policy are unchanged.
`PortableTokenKind` distinguishes input/teacher-forced IDs from sampled IDs.
`step` defaults to input; `generate` and CLI classify produced tokens as
`generated`. Only a generated tokenizer EOS marks the session ended. Reset,
partial-token poisoning and all resource/device guards remain unchanged.
No tokenizer implementation, settings or CUDA production path was changed.

## Regression and local validation

- A new six-shard header-only sparse fixture uses the actual distinct EOS,
  BOS/padding and vocabulary metadata. Against the unchanged 602e3fdf runtime
  it reproduced the exact equality error (`portable-model-eos-red.log`).
  With the correction it reaches the deliberately missing manifest, proving
  those EOS fields were admitted before any weight payload is read. It also
  checks each field's missing/type/range rejection independently.
- The complete synthetic model now uses vocabulary 248320 and the artifact's
  distinct EOS fields. It tests PLE boundary 248044 as a non-terminal produced
  token, input 248046 as a non-terminal prompt separator, a deliberately
  winning output row producing tokenizer EOS 248046, EOS stop/reset, existing
  digest-failure poisoning, context bounds and scalar full-vocabulary logits.
- PLE hash oracles test that predecessor 248044 cuts context while 248046
  remains an ordinary predecessor. The nonzero 10-token integrated PLE/GDN
  sequence now includes both values, with convolution/history, rollback and
  reset checks on CPU and strict software Vulkan.
- Focused PLE, header metadata and complete model CTests passed: 3 tests,
  43.49 seconds. CPU-only CLI/model/PLE/metadata targets rebuilt, and the
  CPU-only header regression passed.
- Strict llvmpipe Vulkan PLE, GDN, QSA and head tests passed. QSA retained the
  corrected high-score spare regression. No physical corrected-head or
  real-weight numerical success is claimed from these software tests.

Logs:
- `${BUILD_ARTIFACT_C6CDA29C},green}.log`
- `${BUILD_ARTIFACT_3E74EA60}`
- `${BUILD_ARTIFACT_3B7C7733},vulkan}-build.log`
- `${BUILD_ARTIFACT_1FFA1D48}`
- `${BUILD_ARTIFACT_F5D4B927},gdn,qsa,head}-vulkan.log`

## Additional lightweight saved-header check and remaining blocker

Read the existing 215 KB Slack header inventory F0CCSCBS0H2; no fleet access
or model payload scan was used. The six exact model shards contain 1224
tensors. Their six global embedding/output HC/head/PLE tensor shapes and
encodings match the portable bindings, including vocabulary 248320 and the
flattened output norm [10240]. EOS/BOS/padding/image IDs, split counts/sizes,
PLE layer [1], compression ratios, RoPE sections and F32 epsilon agree with
the known artifact and existing admission assumptions.

This check exposed one distinct remaining encoding mismatch: every QSA
`indexer.q_proj.weight` [2560,512] and `indexer.k_proj.weight` [2560,128] is
BF16 (ggml type30) at layers 3/7/.../47. `src/core/portable_qsa.cpp:166`
currently requires Q8 for two-dimensional projection weights. Thus a real
retry after the EOS correction would later reject layer3 indexer binding.
The EOS correction does not relax that guard or silently reinterpret BF16 as
Q8. The actual type-specific binding and its reference tests are pending.
No further build/debugging stage was started after this safe checkpoint.

The exact staged retry recipe is `portable-model-hermes-v3.md`; it explicitly
holds real-model stage3 for the known BF16 binding correction. Parent must
select the new reviewed full remote SHA, prepare fresh source/build/evidence
paths, and dispatch one new bounded smoke. The retained prompt, context 64,
max-new 3, CPU expert provider, both-die placement, host/swap/device guards,
reference row at position 37 and frozen comparison limits are unchanged.
The failed 602e3fdf run is preserved as an admission failure, not a quality
result. There is no automatic retry or new model hashing campaign.
