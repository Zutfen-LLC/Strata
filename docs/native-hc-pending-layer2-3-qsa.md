# Experimental layer2 FFN to layer3 QSA pending read

`--native-hc-pending-layer2-3 RECEIPT_PREFIX` ports one ordinary native
pending boundary. It remains disabled by default. It requires serial T1 eager
execution, strict accelerator execution without fallback, resident GDN weights,
and the existing strict-FMA HC component build. Chunk and production modes refuse.

The specification is `Verifier::record_window()` in `src/core/verify.cpp`:
its `pre` lambda derives pending from the preceding layer and calls
`gr_read_group(0, pending, inj2_, inj_)`. `fused_gr_read_multi()` in
`src/kernels/cuda/fused_gr.cu` applies the previous FFN write with FMA before
normalization, injection and HC mixing. The Vulkan path reuses the qualified
pending/norm/mix components with their existing conversion, operation order and
normal-domain numerical contracts. It introduces no arithmetic kernel.

The actual layer2 GDN/FFN producer commits a value-owned original residual,
post-combine block output and injection tuple only after successful dispatch.
Layer3 consumes it before the existing QSA projections and indexer. Projection
and the attention residual both use the applied residual; the unmodified QSA
KV/indexer candidates commit only after the FFN succeeds. Failure and reset
retain their existing transaction semantics. Paid HC calls survive later
provider refusal; model counters distinguish layer2 productions and layer3
consumer calls. Enabling the earlier layer1→layer2 option together with this
option composes the two token-local carries.

Only layer3's four HC roles get a dedicated resident bank: existing Q8 roles
are replaced with native BF16 storage, norm remains F32. Other QSA weights keep
their previous placement. Aligned allocations, native component buffers and
bounded host storage are charged through the existing ledger. The model uses
its existing conservative layer cap before allocation and its measured ledger
for subsequent expert placement. No quota, reserve or selection rule changes.

Layer0's materialization before layer1 PLE and all remaining head/control
boundaries are unchanged. QSA projection, attention, pooling and indexer math
remain the existing portable implementation. This serial port does not
implement native captured-window scheduling or qualify QSA kernel equivalence.
It does not establish full HC, model-quality or performance parity.

`portable_qsa_native_hc_test` exercises the actual GDN producer and QSA consumer
with shared scratch, an independent FMA residual reference, four positions
including pooled-block completion, provider rollback/retry, reset, paid counters,
weight replacement and budget refusal. Its HC device implementation is a test
control. The `qsa` mode of `portable_gdn_native_hc_probe` uses real Vulkan HC
components and exports own-input arithmetic receipts while testing the same
boundary. Development HC weights are reused operands, not real layer3 role
identity. Real-model validation follows separately using layer3 roles directly
from the model and fresh native-policy planning with same-image control/on arms.
