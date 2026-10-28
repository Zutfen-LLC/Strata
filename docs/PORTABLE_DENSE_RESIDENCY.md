# Dense weight residency trial

Strata's native Qwen engine is the design contract: model-lifetime device
weights, session-lifetime recurrent state, reusable scratch, and explicit expert
ownership. A portable backend should follow that contract unless a specific
backend constraint or measured improvement justifies a difference. Backend
allocation APIs implement this ownership; they do not replace the model design.

`--dense-resident-weights` is a default-off trial that places immutable GDN,
QSA, FFN and output-head weights on their layer's selected accelerator. The
weights are uploaded once and retained by their existing layer owners. Without
an accelerator they remain on CPU. The existing `--gdn-resident-weights` flag
still affects only GDN attention weights; the new dense flag includes that
placement. Supplying both is equivalent to dense residency.

This slice changes weight storage only. Recurrent state, host expert boundaries,
scheduling, graph outputs, precision, routing and numerical acceptance limits
remain unchanged. QSA's BF16 indexer weights retain the exact F32 widening at
load time, using bounded host staging. The output head retains its Q8 encoding
and existing backend conversion-workspace policy. Graph splitting and fusion
can change when weight copies disappear, so hardware parity checks are required.

The existing model ledger admits actual aligned persistent device weights,
scheduler storage and workspace separately, before admitting resident experts.
Host and per-device caps, expert plans and free reserves remain explicit.
Preflight uses conservative existing weight/graph caps; admission refusal stops
rather than selecting smaller expert plans or enabling CPU fallback. Fallback
remains controlled by `--allow-cpu-fallback` and recorded in placement reports.
No speedup or full-model fit is implied by this option.

The resident adapter accepts the same flag and forwards it to its single model
child. Compare OFF and ON with the same binary, inputs and explicit expert plans.
Startup `actual_weight_bytes`, `actual_scheduler_bytes`, workspace bounds and
`admitted_bytes` must be inspected separately for each device.

The FFN, QSA, BF16 QSA and head Vulkan fixtures exercise both placements. GDN and
PLE fixtures exercise default, GDN-only and all-dense modes. Their existing
scalar/CPU references, placement checks and reset/rollback checks remain active.
CPU tests establish CPU behavior; they do not establish Vulkan hardware parity.
