# Experimental reusable serial HC family cycle

This default-disabled library slice completes the missing QSA attention→FFN
producer seam and permits the existing native HC components across GDN/QSA
layers of the supported geometry. It adds no arithmetic kernel. The bounded product capability is
`--native-hc-serial-cycle PREFIX` with explicit
`--hc-cycle-resident-weights`; earlier model flags keep their existing scopes.

The specification is `Verifier::record_window()` in `src/core/verify.cpp`:
`gr_read_group(0,pending,inj2_,inj_)` consumes a previous FFN tuple at attention
entry; after attention output projection, `gr_read_group(1,true,inj_,inj2_)`
consumes the actual `bo_`. The same pending/norm/injection/down/up/mix method is
used by both families. QSA now retains `projected` after `attn_output.weight`
as a graph output and copies it before releasing shared attention storage.
Its earlier `gated` tensor is not that producer. The unchanged projected value,
applied residual and attention injection feed the existing FFN HC component.
The FFN can commit a deferred tuple for the next attention or materialize a
native endpoint; those modes are mutually exclusive.

`native_hc_boundary.hpp` transcribes the native pre/post predicates: the first
layer has no prior tuple, layer1 PLE materializes before its HC read, prior
control coverage clears pending, current control/head/split-handoff boundaries
require materialization. Native HC still refuses PLE execution and unsupported
chunk/production/fallback modes. This is no implementation of control vectors,
transport, fused heads, native QSA kernels or captured-window scheduling.
Existing product PLE/control/head/split behavior is unchanged. The earlier
standalone layer0 incoming-tuple diagnostic remains available for its original
component tests; it is not a model boundary rule and cannot combine that
diagnostic with a native FFN read.

Layer-family options opt into the supported native geometry rather than
asserting a particular numerical method solely from an index. Incoming tuples
remain explicit; materialized same-backend controls can use the same native
read with no incoming tuple. Native FFN roles replace their Q8 storage in the
existing resident bank. Dedicated QSA attention HC storage, FFN/native buffers
and bounded host staging are charged by the existing ledger. No new residency
or placement default is selected. Full-model integration must explicitly admit
the selected scope before freezing its original native-policy expert plan.

A QSA outgoing tuple is allocated before the no-throw KV/indexer/public commit.
Provider/HC failure cannot publish it or change eager committed KV/indexer state;
reset clears tuples/state but preserves paid counters. GDN retains the same
candidate/commit behavior. The native components and intrinsic contracts are
unchanged. Same-backend controls and own-input component checks are evidence of
this integration, not native QSA or full-model parity.

`portable_hc_family_cycle_test` uses actual GDN2→QSA3→GDN4 coordinators, two shared
family scratch owners and an HC device test implementation. It independently
checks FMA order, actual post-projection output identity, complete-family state,
rollback/retry/reset, short allocation limits, setup-owner release and all native
boundary predicates. The Vulkan probe's `cycle` mode uses the actual components
and an independently materialized QSA FFN-write/GDN-read control. Four positions
cover QSA block completion. It exports three own-input pending reads and the
control's native write for unchanged numerical checks. Rebound approved HC data
are synthetic roles, not actual layer2/3/4 model role provenance.

## Bounded product cycle

The single capability composes GDN2 attention/FFN, QSA3 attention/FFN and GDN4
attention, reusing the family options and native boundary predicates. It implies
the existing entry boundaries, preserves explicit receipt prefixes, and adds
no independent algorithm per layer. Layer1 PLE and layer0 behavior remain
unchanged, and the scope hands off a materialized residual after the unchanged
GDN4 FFN. It does not broaden to all layers, control vectors, split execution,
heads or native window scheduling.

Explicit cycle residency selects only FFN2/3 and is independent of the numerical
capability so a matched control can admit the same bank policy. QSA attention
weights retain their previous placement. Native shared weights already use a
separate resident bank; the router is F32. Thus before alignment, generic
FFN norm/HC/router bytes are 12,290,560 in control and 18,472,960 with native
HC conversion. With the existing native-shared bank they total 17,518,080 and
23,700,480 bytes per selected FFN. These are source geometry calculations,
not observed VRAM or free-memory credit. Native conversion replaces each
old Q8 bank; buffers, alignment and host envelopes use the actual allocator
ledger. The next GDN attention replacement adds 6,182,400 weight bytes before
alignment. Existing caps/reserves and fresh post-dense native-policy planning
determine whether the complete configuration fits.

Actual paid per-request receipts are `native_hc_cycle_ffn_calls_delta` for the
two selected FFNs, `native_hc_cycle_outputs_delta` for QSA FFN production, and
`native_hc_cycle_reads_delta` for the next GDN read. Counts derive from the
executed calls, survive reset and include work before later failure. The
model's token-local immutable carry uses family-neutral producer access; it is
cleared only at its matching consumer and never crosses PLE/head boundaries.
Own-input checks and same-image model comparisons remain distinct from native
end-to-end or whole-state parity.
