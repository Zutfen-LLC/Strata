# PR42 first real run and numerical interpretation — v1

This is a source/retained-evidence audit, with no new inference, optimization,
threshold changes or raw-logit upload. Runtime observations below were supplied
by the parent from Hermes receipt1793150510.782879 in thread1793145122.081359;
this coding task did not inspect or execute the host-local run.

## Observed result

Source `ae6489205c9eca5c072610dbacbc78c8489dffda`; run on05:
`/home/hermes/portable-model-pr42-ae64892-real`.
Product binary SHA256:
`06831fc5743559190ce56ec148187aa16f09dd3360410667adad13f6f4589c0a`.
The v4 recipe completed strict contiguous placement for48 layers and the head,
all24 exact BF16 widening markers, host PLE and real in-process CPU experts.
Generated IDs `[71093,12305,198]` match the retained CUDA prefix.
This establishes the first full local execution mechanism, not quality parity.

At prompt position37, all248320 finite logits were compared with the retained
02 first-logit row. The unchanged numerical screen **FAILED**:

| Metric | Observed | Frozen limit |
| --- | ---: | ---: |
| Maximum absolute logit difference | 0.5793266296 | 0.01 + 0.0001 times baseline maximum magnitude |
| Maximum relative logit difference | 2748.1550802 | reported diagnostic; no separate gate |
| Relative logit L2 | 0.0717160202 | 0.0001 |
| Maximum absolute probability difference | 0.0013604758 | 0.001 |
| KL(reference\|candidate) | 0.0001161040 | 0.00001 |
| Top1 equal | true | reported separately |

Relative L2 is a vector discrepancy, not an answer error rate. Maximum relative
error divides by individual reference magnitudes floored at1e-12 and can be
large near zero. Neither matching top1 nor three generated tokens waives the
failed screen. No quality pass or new tolerance is inferred.

Peak VRAM was1275203584B /1320042496B, peak process RSS40392712192B,
with zero child swap/system swap-out. Model and supervisor exited; no owned
process remains. Post-run VRAM was8339456B per die versus8327168B baseline:
12KiB higher per die, not exact restoration. Mapped file size is not residency.

## Arithmetic policies found in source

These are current-source contracts, not proof of the reference binary's exact
active settings. Its retained executable hash does not identify source alone.
The actual local `engine-argv.json` and relevant environment/startup choices
must resolve conditional paths before attributing the observed discrepancy.
Do not substitute the distributed NDon input configuration.

| Area | Portable candidate | Native CUDA source / uncertainty |
| --- | --- | --- |
| Expert arithmetic | All selected rows use `native_fmt(8,8)`, native CPU activation/hidden quantizers and `ExpertPool::run_split_multi_native`; rows unweighted in route order | GPU-resident rows use CUDA arithmetic; CPU-backed rows can use the native pool. Actual GPU coverage must be read from the retained run, not assumed to be all-GPU. CPU Q8_0 versus CUDA Q8_1 hidden packing and reduction/nonlinear differences were already isolated in the separate Q8 CPU-reference control. |
| Dense Q8 | GGUF Q8 weights retained; F32 graph inputs/outputs; pinned Vulkan chooses its own matvec kernels | Native dense Q8 MMVQ uses Q8_1 activation packing; canonical engine forms and kernels differ. Vulkan MMVQ selection also depends on integer-dot capability, geometry and available pipeline, so placement alone does not prove which quantizer ran. |
| HC/global mixer | Actual Q8 down/up matrices; F32 residual, RMS, gates, SiLU and stream reduction | `lm_head_mix` requires canonical BF16 down/up forms. Native HC Q8 experiment defaults off on CUDA; `STRATA_HC_Q8` can change this. Same formula does not imply identical stored coefficients or reduction. |
| Output projection | GGUF Q8 through ggml matvec | Loaded `NativeHead` Q8 uses Q8_1 input plus native MMVQ; canonical `lm_head` instead packs Q8_K and uses engine forms. Actual argv/loading determines the path. |
| Prompt schedule | T=1 serial, fresh HC embedding each token | `--prefill` enables a distinct batched path; current default field is zero. The retained LOCAL request alone does not establish batched prefill. Chunked GEMM/scan versus serial accumulation can differ. |
| GDN state | F32 recurrence/convolution history, host/device transfers per step | Native state and convolution histories are also F32; fused/native/batched kernel choices change arithmetic order. There is no evidence here of a blanket BF16 recurrent-state mismatch. |
| QSA | FP16 retained K/V, per-head F32 attention products, exact BF16 indexer weights widened to F32 | Native default pools are FP16; INT8/Q4/hybrid options exist and require actual run settings. BF16 indexer kernels and CUDA attention/reduction differ even when weight values agree. |
| Residual/reduction | F32 HC; sole FFN weighted expert reducer in route order | Native `BlockBuffers::R` is F32 too. CPU and CUDA dot-product trees, FMA, exponential and weighted reduction orders may differ. No universal bit-identity contract follows from F32 storage. |

Source anchors: `src/core/{portable_model,portable_ffn,portable_gdn,portable_qsa,portable_head}.cpp`,
`src/core/{native_dense,native_head,layer}.cpp`, `include/strata/core/layer.hpp`,
`src/kernels/cpu/native_expert.cpp`, and `src/program/generate.cpp` (`run_head`,
`prefill_chunk` dispatch). Pinned ggml `ggml-vulkan.cpp` functions
`ggml_vk_mul_mat_vec_q_f16` / `ggml_vk_should_use_mmvq` explicitly select
Q8_1 activation packing or an available dequantized-weight/F32-input pipeline.
Do not infer activation quantization merely from Q8 weight types.

The historical expert boundary evidence is documented in
[Q8 CPU-reference control](../docs/validation/Q8-CPU-REFERENCE-CONTROL-20261026.md).
It establishes that CPU/CUDA expert arithmetic can differ; it does not establish
that those differences explain this full-model result. Integration mistakes,
routing divergence or another semantic error remain possible.

## What the comparator establishes

`compare_logits.py` originated in9ed478e5 with fixed screening limits and no
backend-sensitive tolerance. Its surrounding fixed CPU-group campaign compared
placement/transport changes while preserving model arithmetic, including
bit-identical local baseline/restore controls. This is evidence of an arithmetic
parity screening use, not calibration as a universal cross-backend quality gate.
It is not literally restricted to the same backend: later GPU assignment
handoffs retain it and distinguish transport from CPU/CUDA kernel parity.
`quality.py` / `quality_v2.py` report it as `fixed_numeric_diagnostic` separately
from task-answer correctness. None of this changes this run's failed verdict.

## Cheapest next diagnostic: retained rows only

Have the reference owner read each existing ~1MiB row once, using the unchanged
`compare_logits.read` finite/length checks. No model run is needed. Return:

1. The actual LOCAL argv, relevant precision/prefill/KV/HC settings from retained
   environment/startup evidence, and CPU/GPU expert coverage if retained. Mark
   unavailable fields unknown; no source reconstruction campaign is required.
2. Mean and standard deviation of each row; mean(delta), standard deviation(delta),
   and centered relative L2: `norm((candidate-mean(candidate)) -
   (reference-mean(reference))) / max(1,norm(reference-mean(reference)))`.
3. Least-squares `candidate = a*reference+b`, and residual relative L2 after that
   fit (diagnostic only); top10 token IDs/probabilities for both rows, winning
   margin, candidate rank of the reference winner, and top10 overlap.

A large common offset with much smaller centered error explains why raw-logit
L2 can look much larger than probability differences. A scale effect or a
structured residual gives a different signature. These are descriptive
discriminators, not adjusted acceptance tests, and cannot by themselves prove
an integration bug or acceptable backend variation. Keep the original metrics
and limits alongside them. Do not compare candidate later positions to a single
reference first-step row.

If structured differences remain, first use any already retained boundary
captures. The decisive discriminator is the **same layer input, state, weights,
route and arithmetic policy** through a bounded CPU/CUDA/portable operation
replay, comparing intermediates before nonlinear/requantization boundaries.
A first differing semantic value under matched policy implicates integration;
differences matching an independently qualified operation's arithmetic establish
that operation's backend contribution. Synthetic parity alone cannot establish
real-model output quality. Without retained intermediates, propose one narrowly
instrumented teacher-forced capture only after the retained settings/row analysis;
parent selects and dispatches it. Do not launch repeated38-token experiments or
simultaneously change prefill, expert placement and head policy.

## Separate performance limitation

The provider stages10 experts at each of48 layers:52224000B per layer, or
2506752000B of Q8 weight copies per token, with paired FNV digest computation
on every selected expert and no verification cache. Digest work adds reads.
This makes current timing unsuitable as an AMD performance benchmark. It is
not evidence explaining the numerical discrepancy, and no optimization is
included in this audit.
