# HC mix refusal diagnostics

This changes diagnostic status and failure-only readback, not HC arithmetic,
activation limits, acceptance or dispatch/synchronization order. It is active only
inside the already opt-in serial HC owner with its existing receipt prefix.

The same four-byte status allocation now contains three eight-bit fields: down
(stage 0), up (stage 1) and mixed reduction (stage 2). Within each field:

| Bit | Rejected condition |
|---|---|
| 1 | Nonfinite checked arithmetic |
| 2 | Subnormal checked arithmetic |
| 4 | Absolute preactivation exceeds the existing activation limit |
| 8 | Exponential is nonpositive or below minimum normal |
| 16 | Denominator exceeds the existing bound |
| 32 | Nonzero numerator is below the existing derived quotient threshold |
| 64 | Nonzero numerator produced a zero quotient |

The refusal still occurs for exactly the union of the old conditions. Several
bits/stages may be set; propagation is not proof of multiple independent causes.
This does not identify the first dot-product lane or instruction. The captured
actual intermediates allow the next same-input source/native comparison.

Only the typed domain refusal, raised after all three dispatches and existing
synchronization complete, can create `OWNER/refusal/`. The metadata identifies the
actual norm role, layer, owner, explicit logical position (null for callers that
provide none), norm attempt, successful mix count and raw/per-stage status. It
marks the values unaccepted. The current normalized input and down/up/mixed
intermediates total 177,920 bytes; one readback vector needs at most 40,960 bytes.
Both fit the existing conservative HC host allowance. No GPU tensor or bank is
added. Generic/preflight/dispatch failures cannot expose stale outputs.

Files are exclusive, preserve first-success receipts and prior failed artifacts,
and cannot turn the rejected candidate into a committed result. Diagnostic I/O
failure is reported separately while the original numerical refusal is rethrown.
Position is passed by the actual GDN/QSA owner through the FFN producer boundary,
rather than inferred from paid counters or reset history.

The PR112 position 2/layer 38 stop remains a stop. A single on-arm reproduction with
the original input/profile, retained frozen plan and fresh actual admission is
needed because no rejected input survived that run. No control repetition, domain
correction, clipping, tolerance change or broad comparison is part of this slice.
