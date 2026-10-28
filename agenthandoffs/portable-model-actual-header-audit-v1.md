# Actual full-model binding audit — v1

Source evidence: saved Slack file `F0CCSCBS0H2` (`validation host-metadata.txt`, about
215 KB), already captured on05. Only the six root
`Qwen3.8-Flash-Next-Q8_0-0000N-of-00006.gguf` header/directory records were
selected. No fleet access, payload read, new model hashing or general audit
framework was used.

A one-off local comparison checked the actual signatures against the current
portable FFN/GDN/QSA/head shape declarations, per-role type rules, integer
metadata pin lists and PLE hash constants. It covered every tensor role, not
just a sample layer. All 1224 names are unique and their computed payload
extents fit their shards without overlap. Six shard sizes total
188225033248 bytes. No further unsupported actual type, shape or active
semantic metadata was found after the explicit BF16 indexer correction.
This is header compatibility evidence, not a payload-integrity or numerical
quality claim. Runtime manifest/selected-expert digest checks remain required.

## Tensor roles

Dimensions below are GGUF order (dimension zero varies fastest). Common
roles occur at all 48 layers. GDN roles occur at 0/1/2/4/.../46 (36 layers),
QSA roles at 3/7/.../47 (12 layers), and PLE roles only at layer1.

| Global role | Actual shape | Actual type / binding |
|---|---|---|
| token_embd.weight | 2560,248320 | Q8_0, mapped host row gather |
| output.weight | 2560,248320 | Q8_0, portable output projection |
| output_hc_norm.weight | 10240 | F32 |
| output_hc_down.weight | 10240,320 | Q8_0 |
| output_hc_up.weight | 320,10240 | Q8_0 |
| per_layer_token_embd.weight | 160,320001536 | Q8_0, read-only host mapping |

| Common per-layer role | Actual shape | Actual type |
|---|---|---|
| hc_attn_norm.weight | 10240 | F32 |
| hc_attn_down.weight | 10240,320 | Q8_0 |
| hc_attn_up.weight | 320,10240 | Q8_0 |
| hc_attn_inject.weight | 10240,4 | Q8_0 |
| hc_ffn_norm.weight | 10240 | F32 |
| hc_ffn_down.weight | 10240,320 | Q8_0 |
| hc_ffn_up.weight | 320,10240 | Q8_0 |
| hc_ffn_inject.weight | 10240,4 | Q8_0 |
| ffn_gate_inp.weight | 2560,512 | F32 |
| ffn_gate_inp_shexp.weight | 2560 | F32 |
| ffn_gate_shexp.weight | 2560,640 | Q8_0 |
| ffn_up_shexp.weight | 2560,640 | Q8_0 |
| ffn_down_shexp.weight | 640,2560 | Q8_0 |
| ffn_gate_exps.weight | 2560,640,512 | Q8_0 |
| ffn_up_exps.weight | 2560,640,512 | Q8_0 |
| ffn_down_exps.weight | 640,2560,512 | Q8_0 |

| GDN role | Actual shape | Actual type |
|---|---|---|
| attn_qkv.weight | 2560,10240 | Q8_0 |
| attn_gate.weight | 2560,6144 | Q8_0 |
| ssm_alpha.weight | 2560,48 | Q8_0 |
| ssm_beta.weight | 2560,48 | Q8_0 |
| ssm_a | 48 | F32 |
| ssm_dt.bias | 48 | F32 |
| ssm_conv1d.weight | 4,10240 | F32 |
| ssm_norm.weight | 128 | F32 |
| ssm_out.weight | 6144,2560 | Q8_0 |

| QSA role | Actual shape | Actual type / binding |
|---|---|---|
| attn_q.weight | 2560,12288 | Q8_0 |
| attn_k.weight | 2560,512 | Q8_0 |
| attn_v.weight | 2560,512 | Q8_0 |
| attn_output.weight | 6144,2560 | Q8_0 |
| attn_q_norm.weight | 256 | F32 |
| attn_k_norm.weight | 256 | F32 |
| indexer.q_proj.weight | 2560,512 | BF16 (30), exact widening to F32 |
| indexer.k_proj.weight | 2560,128 | BF16 (30), exact widening to F32 |
| indexer.q_norm.weight | 128 | F32 |
| indexer.k_norm.weight | 128 | F32 |

| PLE role | Actual shape | Actual type |
|---|---|---|
| ple_key.weight | 2560,10240 | Q8_0 |
| ple_value.weight | 2560,2560 | Q8_0 |
| ple_norm_key.weight | 10240 | F32, accepted flattened norm |
| ple_norm_query.weight | 10240 | F32, accepted flattened norm |
| ple_norm_conv.weight | 10240 | F32, accepted flattened norm |
| ple_conv1d.weight | 4,10240 | F32 |

Coverage: 6 global + 16×48 common + 9×36 GDN + 10×12 QSA + 6 PLE = 1224.
The 57 family-specific signatures all match. No additional MTP/global/active
layer tensor needs to be ignored or guessed for this artifact.

## Active metadata

- qwen4exp, 48 layers, H2560, HC4/rank320; 512 experts/top10 and routed/shared
  FF640 match every current factory pin. GDN state128/groups16/time-rank48,
  inner6144 and convolution4 match. Attention heads24/KV2, key/value256,
  indexer heads4/key128/top-k2048 and full-attention interval4 match.
- Compression ratios are exactly zero except4 at each QSA layer. RoPE uses
  dimension64, sections[11,11,10,0], base1e7, and F32 RMS epsilon1e-6.
  No unsupported RoPE scaling/frequency-factor or attention-scale extension
  is present. Optional default behavior therefore matches the stored header.
- PLE is configured only at layer1, embedding head160, n-gram3, heads-per-gram8
  and convolution4. All three uint64 multipliers and all16 offsets/vocabulary
  sizes exactly match `ple_artifact_consts`. Ranges fit the table. The table
  has legitimate trailing padding; no equality between used hash rows and
  total table rows is imposed.
- Tokenizer vocabulary248320 agrees with embedding/output shapes. Tokenizer
  EOS248046 is independent of PLE boundary248044; BOS/padding248044 and image
  ID248056 are in vocabulary. The retained38-token prompt includes248046 at
  position29 and fits context64 plus3 generated tokens. Prompt separators
  are input, not generated stopping tokens.
- Six split numbers/counts and total tensor count1224 match. The retained
  expert manifest covers the expert shards4–6, not the six-shard payload
  identity. Its actual role ranges/types are checked by the existing runtime
  parser/provider; no new full-file identity schema or scan was introduced.

The BF16 roles were the sole remaining encoding mismatch found by this
complete saved-header comparison. They are now handled explicitly and
losslessly. Actual payload digests, runtime resource admission, physical
execution and numerical comparison remain the parent's bounded smoke work.

Local audit receipt: `${BUILD_ARTIFACT_8A09CD78}`; one-off
comparison script: `${BUILD_ARTIFACT_BA4DA801}`. No audit command
is a new runtime prerequisite.
