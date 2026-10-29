# Portable QSA device combine

`--qsa-device-combine` is experimental and off by default. It requires
`--qsa-production`, `--qsa-reuse-attention` and a strict selected accelerator.
Both the fresh quality adapter and resident adapter forward the flag. Fresh
references bind its boolean value in the summary, argv and saved runtime timing
receipt; resident requests must report that same policy. A missing receipt is
accepted only when the option is off.

The option appends the existing gate, output projection and HC residual combine
to the exact active attention graph. It consumes backend views of the projection
outputs and original residual. The projection input is retained as an output so
GGML cannot reuse it before the later graph consumes it. The view leaves do not
import the projection DAG. They must use the selected backend's default buffer
type. The standalone combine scheduler is not allocated in this mode.

Project execution, finite reads, CPU FP16 K/V conversion and overflow checks,
and causal append/pooling still happen in their original order. Attention,
gated attention and the combined residual are downloaded and checked before
FFN execution or public state publication. This is not a zero-download path.
The 24 separate single-head F32 attention products, indexer precision, cutoff
ties, spare/tail handling, routing and poison/reset rules remain unchanged.
There is one retained arena, primed at capacity and rebuilt at the exact active
shape. There is no padded attention or cache of graphs by sequence length.

The existing graph cap, conservative scheduler pricing, retained high water and
model admission ledger still apply. There is no extra permanent combine buffer
and no cap increase. The ledger can move allowance between actual scheduler
bytes and conservative host workspace; a smaller scheduler is not a promise of
lower total host admission. Check placement and admission on the selected backend.

With the option enabled, runtime timing and resident completion receipts include
`qsa_device_combine: true` and actual `qsa_attention_submissions_delta`,
`qsa_combine_submissions_delta`, and `qsa_device_handoffs_delta`. They count
successful graph computations, including computations in steps that later fail.
Reset preserves cumulative counters; request receipts subtract their starting
values. A successful N-token request visits 12 QSA layers per token: 12N attention
submissions, zero standalone combine submissions and 12N device handoffs. Chunk
FFN calls are separate and are not these attention counts.

## Checks

Build and run `portable_qsa_test`, `portable_qsa_bf16_test`, and
`portable_qsa_memory_test` on CPU. They cover chunk sizes 1–8 against the separate
graph path, exact active state/routes, BF16 widening, continuation, callback
failure and A/B/A reset. The selection fixture uses the same combine expression
builder with selected attention through 2056 cells, including the 2051 cutoff,
ties and incomplete tails, with unchanged numerical thresholds. It retains the
24-head F32 attention path. The small output projection in the cutoff fixture
isolates selection/combine semantics; full model-width projection is covered
by the layer fixture.

Run the portable Python tests, `portable_stdio_cli_test`,
`portable_timing_cli_test` and `portable_logits_guard_build_test`. Build the
Vulkan `portable_qsa_vulkan_test`, `portable_qsa_bf16_test` and `portable_model`
targets before requesting physical qualification. Compilation is not GPU
execution evidence.

On each verified device, run the existing QSA fixture with
`--device INDEX --expect-device DESCRIPTION`, then the BF16 fixture with
`--vulkan INDEX`. Preserve strict placement, existing numerical gates, state and
route checks, and the graph work counters. Keep a finite outer guard and the
existing continuous resource, thermal, sensor and owned-process cleanup
controls. Stop at the first failure; do not change caps or retry automatically.
CPU fixtures cover injected failures; do not infer GPU fault-injection coverage
from a successful physical run. Real-model numerical and performance
qualification is a separate later gate. No speedup is established here.
