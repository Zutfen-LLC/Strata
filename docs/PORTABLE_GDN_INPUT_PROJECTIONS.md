# Optional portable GDN input projections

`portable_model --prompt-chunk-size 2..8 --gdn-batch-input-projections`
opts non-PLE GDN layers into batched input projections. Both options are required.
Omitting the new flag retains the existing execution and output formats.

Native prefill (`src/prefill/prefill.cpp`, GDN section) projects qkv, the output
gate, alpha and beta with T columns before convolution and recurrence. This
portable slice follows that boundary using the existing Q8 weights and GGML
scheduler. Independent HC preparation and injection also use T columns. HC stream
addition order, F32 projection outputs, alpha/beta nonlinear operations, and the
existing convolution, recurrence and output projection expressions are retained.
Convolution and recurrence still execute and commit one token at a time.

T1 chunks, final prompt/decode steps, and the configured PLE layer use the existing
serial graph. PLE's causal convolution changes the residual before HC mixing, so
it cannot use this independent-input graph. There is no new weight copy or expert
execution path. This does not batch GPU experts or implement a chunk recurrence
kernel.

One projection shape is retained at a time. Its graph allocation, CPU work buffer,
metadata and bounded F32 output staging are included in module admission. Each
projection column is staged into the existing sequential attention operations;
this slice does not claim zero-copy device chaining or a measured speedup.
Failed chunks retain the existing poison/reset contract and publish no completed
chunk outputs or routes. Projection submission/column observers count actual
successful batched submissions, including work preceding a later chunk failure;
reset preserves these cumulative receipts.

## Focused verification

Build and run `portable_gdn_test`, `portable_ple_test`, `portable_model_chunk_test`,
`portable_stdio_cli_test` and `portable_timing_cli_test`. Large sparse model
fixtures need sufficient temporary-file storage; use a disk-backed TMPDIR when
/tmp is a small tmpfs. The GDN fixture compares serial versus chunk T1–8 and tails
under the existing numerical gate, exact state/route checks, reset replays and
injected failures. It requires four named projection MUL_MAT nodes with actual
T-column shapes and checks successful projection submission counts separately
from completed chunk counts. The assembled CPU fixture enables the new option;
the fresh/resident CLI check exercises it at chunk size 8 and compares executed
logits using the unchanged manifest schema and numerical gates. Enabled JSONL and
timing receipts carry `gdn_batch_input_projections: true`.

For a minimal hardware check, build `portable_gdn_vulkan_test` with the retained
pinned dependency and Vulkan configuration. Run sequentially for each explicitly
verified device, inside the existing bounded resource/temperature guard:

```sh
timeout 600 ./portable_gdn_vulkan_test --device INDEX --expect-device DESCRIPTION
```

The fixture includes serial controls and diagnostic/production candidates, with
and without the new option. Require strict selected-device placement without
fallback, all four named T-column input projections, existing FFN coverage,
unchanged state/routes, and the existing numerical thresholds. Retain each
max_abs, relative_l2 and compared-element receipt. Each enabled Vulkan mode must
report 48 successful batched projection submissions and 180 columns across the
T1–8/tail/reset sequences. GPU injected failure checks remain zero; CPU fixtures
own fault injection. Stop on first admission, placement, state, route or numerical
failure. Do not infer assembled GPU model correctness or performance from this
component check.

## Adapter policy and bounded model comparison

Both `portable_quality_v1.py portable` and `portable_resident_v1.py` accept
`--gdn-batch-input-projections` and require a prompt chunk size. Enabled runs
record the boolean policy in their summaries and require the runtime timing/done
receipt to report the same policy. Disabled runs preserve the existing output
format and reject foreign enabled receipts. Suite references bind the fresh
summary, each saved command, and each enabled case's runtime timing and saved
chunk-work receipt; missing, mismatched, non-boolean or duplicate evidence stops
the comparison before resident execution.

After the component gate, a small model check can compare the same binary and
retained plans with chunk size 8 in both arms, adding the projection option only
to the candidate. Use `portable_quality_v1.py --gguf FIRST_SHARD portable` with
`--single-case --dump-executed-logits` and the existing guarded policy/cap
arguments. Compare the two `case-01/executed-logits` directories with
`executed_logits_v1.py --baseline-chunk-size 8 --candidate-chunk-size 8` and
unchanged numerical gates. Token rows and executed-logit manifests keep their
existing schemas; skipped prefix predictions remain explicitly unavailable.

For a subsequent short timing comparison, use `portable_resident_v1.py` with
`--tokens-file` containing the retained case 1 IDs, unchanged `--max-new`, the
same binary/plans and chunk size 8. Only the candidate adds the projection flag.
Each arm runs four requests and excludes the first. Capture is off. Check exact
rows/stops and available predictions against the corresponding fresh arm, and
retain the existing warm-zero expert-work checks and resource guards. These
commands describe a bounded follow-up, not evidence of hardware or model results.
