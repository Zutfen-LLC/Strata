# Ordinary serial HC scope

`--native-hc-serial PREFIX --hc-serial-resident-weights` selects the existing
native-method HC components across ordinary layer boundaries. It requires the
qualified layer0 write option and its explicit residency, GDN attention residency,
strict eager T1 execution and primary dense topology. It is disabled by default.
`--hc-serial-resident-weights` alone changes residency without selecting arithmetic,
so a same-image control can have identical FFN banks and ownership.

The specification is `Verifier::record_window()` in `src/core/verify.cpp`: `pre`
reads the previous FFN pending tuple unless the preceding control operation or
layer1 PLE already materialized it; the post-attention read uses the actual output
projection; `post` leaves ordinary FFN writes pending and materializes the final
nonfused-head write. This port uses `native_hc_boundary.hpp` for those predicates.
Layer0 has no incoming write. Its existing write materializes before layer1 PLE.
Layer 1 retains its PLE graph. When the post-PLE native HC read is also selected,
its attention output and injection feed the existing native FFN pending read, and
the FFN weights are resident. Layers 2–47 use the shared GDN/QSA components, with
the final FFN write materialized before the head.

Native control-vector execution, pipeline/split transport and fused pending head
are not approximated: this scope only admits the existing primary dense serial
profile. `STRATA_FUSE_HEAD_GR=1` refuses explicitly. The portable interface does
not accept native control-vector or pipeline modes. This does not port native QSA
numerical kernels, window scheduling, native GDN recurrence on the PLE layer, or
final head HC arithmetic. The layer 1 handoff alone is not full-model parity.

The bounded `--native-hc-serial-cycle` profile remains available and exclusive with
the ordinary numerical scope. No cap, reserve, expert selection or helper policy
changes. Actual role banks replace the prior representation; existing aligned
allocation and scheduler/workspace ledgers plus conservative host bounds are
charged before original post-dense capacity planning. Planning refusal is a valid
result; there is no quota fallback or invented streaming mode.

For the target geometry, relative to the bounded cycle, additional persistent
role payload is 1,156,177,920 bytes before alignment: 32 GDN attention conversions
(197,836,800), 11 QSA HC banks (145,530,880) and 44 FFN banks (812,810,240).
Selecting the post-PLE native HC read additionally residents the layer 1 FFN bank;
the original estimate above excludes that bank. Shared FFN weights are already
resident and are not added again. Scheduler,
component workspace, alignment, host bounds and backend-private costs still apply;
these payload figures are not a fit claim. The unchanged planner must measure
post-dense room and derive the primary quota and excluded helper selection.

CPU tests cover all 48 predicates, matched residency, actual GDN/QSA shared-scratch
producer/consumer, final-layer write, candidate failure/reset/retry, exact local
FMA reference and unsupported-mode refusal. They mock HC device arithmetic and do
not prove hardware or full-model parity. Runtime counters expose per-layer actual
reads, FFN reads, deferred outputs and final writes. Model comparison retains exact
within-arm replay and own-input numerical contracts; cross-arm route/logit changes
remain diagnostics, with no invented final-logit tolerance or quality claim.
