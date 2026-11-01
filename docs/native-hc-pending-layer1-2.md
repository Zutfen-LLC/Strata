# Experimental ordinary layer1 to layer2 HC carry

`--native-hc-pending-layer1-2 RECEIPT_PREFIX` defers the actual layer1 FFN final
write and folds it into layer2's attention HC read. It is default-disabled and
restricted to serial eager strict accelerator execution with resident GDN
attention weights (or existing all-dense residency). Chunks, GDN production,
batched projections, legacy observer and alternate native HC modes refuse.
The existing native normal/pending/mix numerical contracts and feature checks
remain; no tolerance or shader arithmetic changes.

Native `verify.cpp` retains R, the combined FFN block and injection until the
next attention read, except at PLE/control/head boundaries. This slice covers
only the ordinary layer1 to layer2 boundary. Layer0 still materializes before
layer1 PLE. Layer1's PLE, attention and FFN pre/route/shared/combine computations
keep their current methods. In particular, carrying their actual output does
not qualify those upstream methods as native-equivalent. Layer2's subsequent
FFN and other boundaries keep their existing materialization.

The producer publishes a value-owned residual/block/injection tuple after its
existing successful FFN and GDN state transaction. The unused FFN post graph
is omitted. No weights or device scratch banks are duplicated. A conservative
1 MiB host envelope covers candidate, committed and caller tuple copies; one
fixed-shape tuple is 51,216 bytes. Layer2's existing pending HC owner retains
and charges its own input/output leaves and conversion/host envelope. Model
preflight and actual ledgers retain the original budgets, reserves and placement
policy. A fresh native plan must be derived for real model admission.

The model carry is local to one token forward, produced at layer1 and consumed
exactly once at layer2. Producer or consumer failure does not publish a rejected
layer candidate or overwrite the prior committed tuple. Reset clears tuple
lifetime, while paid counters survive. As before, a model forward can already
have committed earlier layers when a later layer fails: the session poisons and
requires reset/replay, rather than claiming a new whole-forward rollback.
`deferred_ffn_layer1_outputs_delta` counts completed tuple productions, not
write kernels. `native_hc_pending_layer2_calls_delta` counts actual successful
component calls, including work before a later provider refusal. Layer0 paid
counters remain restricted to layer0.

CPU tests run the actual two GDN coordinators with a mocked device HC backend,
including an independent FMA/stream-order reference, materialized-input control,
producer and consumer refusal/reset/retry, missing/invalid modes, no duplicate
weights or unused post graph, short-budget refusal and default CLI controls.
They are not GPU qualification. The existing probe's `boundary` mode runs a
small synthetic two-layer Vulkan check, reusing approved layer0 attention roles
as development data and synthetic remaining weights. It exports an identical-
input pending fixture for the unchanged numerical checker. It does not execute
actual layer1 PLE, load a full model or supply real layer2 role provenance.

The inspected retained STRL004 tuple is explicitly position0/layer0; the latest
model packet retains layer0 HC receipts, routes and heads. Those do not supply
a real layer1 FFN producer tuple or layer2 HC roles, and rounded data must not
be reconstructed into a missing tuple. Inspect any original retained native
trace before proposing a fresh capture for later real-boundary qualification.
No capture is required for the synthetic component gate.
