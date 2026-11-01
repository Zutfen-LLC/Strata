# Experimental layer-0 final FFN materialization

`--native-hc-ffn-write-layer0 RECEIPT_PREFIX` selects the existing qualified T1
native final-write component at the actual shared-plus-routed FFN block output.
It implies the preceding native attention/read and FFN pending-read options.
Explicit `--layer0-resident-weights` (or existing all-dense residency), strict
accelerator execution, serial eager execution and a receipt prefix are required.
Chunks, production/fallback, legacy observer and alternate HC modes refuse.
It is disabled by default and scoped to layer 0.

The retained applied FFN residual and its actual injection feed the write directly.
Only the final block and disjoint candidate/gate/exp leaves are added to the
existing HC owner. No weights or residual/injection banks are duplicated. The
unused GGML post context/graph/scheduler is omitted in this mode; the default
path retains it. Native scale-with-positive-zero-bias, sigmoid and final FMA
order are unchanged from the standalone component. Its existing normal-domain
contract still applies; there is no new residual tolerance or mode fallback.

The FFN publishes the materialized candidate only after its provider, combine,
component, finite checks, initial receipt and optional trace succeed. The
surrounding GDN commits its candidate recurrent/history state only after FFN
success, as before. Failed work publishes no route/residual/state. Reset keeps
paid component counters. `native_ffn_hc_write_layer0_calls_delta` reports
successful component work separately from pending read and actual GGML post
calls. A successful write followed by a later publication failure remains paid.

Model host preflight and actual ledger add a conservative 1 MiB envelope for final
preflight/candidate/receipt vectors. Actual owner tensor/component storage is
charged to the existing graph budget before the post-dense native planner.
Short budgets refuse. Caps, physical reserve, native quota/rank/exclusion,
placement/CPU misses, other layers and state policy are unchanged. Any physical
model validation must derive a common matched control/on plan after actual
construction; old slot counts are not new admission evidence.

## Native boundary and limits

Original native `verify.cpp` materializes the preceding FFN write before layer 1
PLE, while other non-PLE/control/head paths have their existing pending rules.
The portable model already hands a materialized layer 0 result to the next layer;
this option replaces that materialization's arithmetic without moving PLE or
state commits. It does not implement generic cross-layer pending carry, rewrite
attention/FFN scheduling or qualify later HC layers. PLE tables and predecessor
history retain the model's existing handling. No full-HC, model quality or
performance claim follows from component acceptance.

The existing `portable_gdn_native_hc_probe ... write` command exercises actual
GDN producer -> FFN pending read -> shared/experts -> final write, with retained
approved attention operands and source-generated synthetic FFN weights. It
exports both actual and same-backend materialized-control final-write operands,
checks reset/replay/rollback and paid counters, and repeats the disabled path.
Whole-state equality is source-asserted; whole state arrays are not exported.
This coordinator is a separate physical gate before a fresh matched-model run.

Local tests cover actual consumer identity, independent FMA/indexing, budget
refusal, no duplicate weights/unused post graph, creation/runtime refusal,
rollback/reset and counters, actual CLI/API preconditions, scoped host admission
and full CPU model default-output regression. CPU component stubs are transaction
controls, not GPU evidence. The bounded physical coordinator and fresh same-image matched-model diagnostic
now pass actual write participation, initial own-input numerical contracts and
within-arm reset/replay checks. Generated IDs agree across control/on, while
later routes and retained heads differ. This is limited layer-0 integration
evidence, not model numerical, quality or full-HC parity. Whole-state arrays
remain source asserted rather than exported; executable/runtime payloads remain
host bound. Generic inter-layer pending carry remains unfinished.
