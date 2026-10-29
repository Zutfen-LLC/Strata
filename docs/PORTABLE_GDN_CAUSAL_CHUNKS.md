# Optional portable GDN causal chunks

`portable_model --prompt-chunk-size 2..8 --gdn-batch-input-projections
--gdn-production --gdn-resident-state --gdn-causal-chunk` enables an experimental
production path for non-PLE GDN layers. It also requires the existing strict
selected accelerator policy. The option is off by default. Fresh and resident
adapters forward it and bind the boolean policy in commands, summaries and runtime
receipts. Suite references require matching saved timing receipts as well.

Native prefill's GDN section (`src/prefill/prefill.cpp`) performs independent
T-column input projections before convolution, ordered recurrence and output.
This portable slice keeps those projection tensors in the same GGML graph as
T existing single-token convolution, recurrence and output expressions. Each
recurrence consumes the previous candidate state; each convolution consumes the
previous candidate history. Views select projection columns without host uploads.
It adds no new recurrence kernel or expert execution path.

The final graph state and history are separate from committed device inputs.
After one graph submission, the existing five bulk projection reads still check
finiteness, and one bulk residual read checks all T residuals before FFN. Projection
checks now happen after candidate attention computation, before provider execution
or any state commit. No data-dependent indexing is introduced. Only after batched
FFN succeeds are the final state and history copied once and synchronized once.
Any execution, finite-check, provider or commit failure poisons the layer until
reset. Completed output rows, routes and public token metadata publish together.

T1, final prompt/decode steps and PLE retain their serial path. Shorter T2–7
chunks use their own bounded shape. Only one chunk attention shape is retained;
its metadata, scheduler buffers, CPU work storage and host staging remain in the
existing module ledger. Existing caps remain enforced. Admission credits immutable
weights only when they are already allocated in the consumer backend's default
buffer type; at the pinned scheduler these sources need no scheduler copy. Other
conservative allocation bounds remain. Some configurations can still be refused.

The tail expression builder is shared with serial execution, preserving HC stream
addition order and the existing F32 nonlinear, convolution, recurrence and output
operations. No arithmetic reassociation or tolerance change is intended. Physical
Vulkan numerical equivalence and performance still need hardware verification.
This path does not batch GPU experts and makes no speedup claim.

## Focused verification

Run the existing `portable_gdn_test`, `portable_ple_test`,
`portable_model_chunk_test`, `portable_stdio_cli_test` and
`portable_timing_cli_test`, plus portable Python harness tests. Use disk-backed
TMPDIR for sparse assembled model fixtures. The new production component case
covers T1–8, tails, exact route IDs, reset replays, serial continuation and provider
and post-graph failures. CPU-only fault seams also reject the causal graph and
throw on its final committed-state write; failed chunks publish no output.
The assembled and CLI CPU regressions exercise their existing diagnostic policy;
they do not establish assembled production execution of the new graph.

For a focused hardware gate, use the retained pinned GGML dependency and known
working Release Vulkan configuration. Build `portable_gdn_vulkan_test` and record
source and binary hashes. With each selected device identity verified first, run
sequentially inside the existing finite resource, temperature and owned-cleanup
guard, stopping on the first failure without retries or cap changes:

```sh
timeout 600 ./portable_gdn_vulkan_test --device INDEX --expect-device DESCRIPTION
```

The fixture retains the serial and PR78 projection controls and adds the causal
production case. Require all non-metadata causal operations on the selected
Vulkan backend without fallback, four named T-column input projections, T actual
recurrence nodes per T>1 graph, the existing numerical gates and exact route/state
checks. Each mode compares 2,211,840 residual elements over 84 chunks and 216
attention tokens. Retain actual max_abs and relative_l2. The causal case must
report 48 successful submissions, 48 residual handoffs and 48 final state commits;
its projection receipts must show 48 submissions and 180 columns. GPU injected
failure checks remain zero; CPU fixtures own injection.

Runtime timing/done receipts expose actual cumulative-counter deltas only when
the option is enabled: `gdn_causal_submissions_delta`,
`gdn_causal_residual_handoffs_delta` and `gdn_causal_state_commits_delta`.
Submissions and handoffs can precede a failed FFN; commits count only successful
final copies and synchronization. Reset preserves these counters. These are host
API execution counts, not per-kernel timings. Optional telemetry is not a numerical
gate. Component success does not qualify assembled GPU model execution, executed
logits or performance; a separately authorized same-binary capture comparison
would be the next model gate.
