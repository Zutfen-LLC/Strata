# Diagnostic pending tuple at the serial attention coordinator

This adds an explicit C++ diagnostic seam, not a model CLI flag or automatic
pending carry. The normal layer-0 attention entry has no previous native write;
it must never infer a pending tuple from the ordinary residual argument.
`PortableGdnOptions::native_hc_pending` requires the existing composed read,
eager layer 0/non-PLE T1, strict accelerator, resident attention roles and receipt
prefix. Existing no-pending and default-disabled behavior remain available.
Production/chunk modes and legacy Layer0Observer refuse: STRL003 does not bind
a pending block-output/injection tuple. Bounded PortableGdnTrace fields do.

`step_pending(R, previous_output, previous_injection, ...)` accepts immutable
caller operands. It uploads all three into stable, admitted owner storage,
forms a disjoint R-prime, then executes the existing native norm/injection and
mixing components. BOTH graph hidden/residual consumers and HC read consume
that same R-prime. Passing old R to the graph while HC reads R-prime is a bug.
The source uses exactly the components in [pending T1](native-hc-pending-t1.md);
it adds no arithmetic kernel, conversion or numerical tolerance.

Graph/state candidates stay private. Token metadata, public last_residual and
recurrent/convolution state publish only at the existing whole-layer commit
after FFN/provider success. Pending, norm and mix successful-work counters are
separate and cumulative across reset, including paid work before a later refusal.
Caller R/block-output/injection remain unchanged even on success. The native
up pass's in-place write is represented by an owned candidate here; this is a
transactional diagnostic bridge, not captured-graph scheduling parity or a new
production lifetime policy. FFN read/write arithmetic remains the old method.

There is no extra weight bank or retained host role copy. Stable original R,
prior block output/injection and gate/exp add 51,248 payload bytes before
alignment; the pending component's status/metadata and actual buffer allocation
are charged to the existing graph budget. A separate conservative 128 KiB
supplement to the composed host bound covers the less than 93 KiB pending
preflight peak or less than 51 KiB additional trace fields in disjoint phases.
The first successful attention graph can record the original tuple, R-prime and
HC outputs without overwriting previous receipts. No tuple is silently reused:
every diagnostic call must supply the complete fresh tuple explicitly.

Host tests run the real GDN/FFN coordinator and allocator with mocked device HC
arithmetic. They exercise private/shared scratch, host/resident diagnostic state,
partial component failures, provider rollback, reset/retry, next tuple uploads,
owner destruction, bounds, mode/schema refusals and unchanged disabled controls.
A separate sequential same-backend composed control is fed actual R-prime;
final residual/state/history must be exact. A deliberate old-R graph mutant
fails this control, while the corrected source passes. Mocks are not physical
arithmetic qualification.

`portable_gdn_native_hc_probe OPERANDS FRESH_OUTPUT EXPECT_DEVICE pending`
uses previously approved attention roles/embedding with deterministic synthetic
previous block output/injection and existing synthetic remaining GDN/FFN data.
It compares an actual pending coordinator against a sequential materialized
R-prime control, checks reset/provider rollback and disabled repeats, and exports
bounded tuple/component observations for unchanged own-input contracts.
Original embedding zero columns need not remain zero after a legitimate pending
write; original upload equality and exact own-gate FMA replace that inappropriate
invariant. No native GDN or all-layer/model equality follows.

Before any model use, the pending tuple's real half/layer producer and FFN role
identity must be established separately. The current attention-role extraction
cannot qualify FFN weights or native deferred carry across model halves. This
slice makes that missing semantic boundary explicit instead of inventing a model
method. Defaults remain disabled, and full_hc_parity/qualified remain false.
