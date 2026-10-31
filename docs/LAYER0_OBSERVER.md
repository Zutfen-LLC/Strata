# First-token layer-0 observer

This opt-in diagnostic records existing values at position 0, layer 0. It is for
finding the first native/portable difference through MoE combine. It does not
establish numerical, quality or performance parity and changes no kernel,
route, mask, state commit or default model policy.

Set both variables only for the selected child process:

```sh
STRATA_LAYER0_OBSERVER=/new/private/trace.bin
STRATA_LAYER0_BINDING=64_lowercase_hex_SHA256_of_reviewed_input_and_ownership_receipt
```

The binding is an operator-provided identity receipt, not an automatic proof of
weight provenance. Before comparing, verify the same source model/role versions,
logical weight formats, token, fresh state, primary/helper assignments and
recorded runtime settings. Retain source/binary/argument/environment identities
with the raw traces. Do not set the same binding on unrelated artifacts.
Actual layer-0 primary/helper membership and token are additionally recorded and
must agree. Physical cache slot numbers and allocation addresses are not
numerical operands.

Use a fresh one-token request, not a continuation of an earlier session. The
native reference requires CUDA, the existing native pack and methods, serial
serving, an actual first window of T1, no layer split/pipeline and at most one
local expert helper. Preserve configured `--spec` capacity: native packs require
at least two, while their first one-token window is T1. Only T1 graphs receive
snapshot nodes. `--native` is the supported method preset; it resolves dense
shards and enables the other native components, rather than merely aliasing
`--native-dense-gguf`. Preserve that preset and its resolved role provenance.
Local `--remote-expert-opt` remains selected when present. It is distinct from
remote CPU serving. The observer refuses remote CPU mode,
missing primary residency evidence and programmatic dependent launch, whose
kernel-only graph dependencies cannot be preserved by inserting copy nodes.
The portable observer requires the existing serial eager GDN policy; it refuses
prompt chunks and GDN production mode rather than changing those settings.
The observer is never rearmed by reset. Later calls cannot append to the trace.

## Values and availability

The fixed version-3 schema describes H2560, HC4, 512 experts, K10 and the existing
GDN state/history layout. It records embedding/HC residual, incoming GDN state
and history, the FFN pending residual, normalized MoE input, router scores,
ordered IDs/weights, gated shared output, original-order unweighted expert rows when available,
combined output and injection coefficients. The pending residual, combined
output and injection are separate operands of the native pending update.

Version 3 additionally requires the attention HC mixed input (2560 F32 values)
and attention HC injection (4 F32 values), before GDN projections/recurrence.
Native snapshots the existing fused attention HC read outputs on its original
stream. Portable retains and reads the corresponding existing graph outputs.
Only the opt-in graph keeps these intermediate outputs alive; its scheduler
storage remains part of the ordinary graph admission ledger. No attention output
projection or residual is reconstructed. These fields precede the FFN pending
residual in comparison order. A difference there locates divergence before GDN;
equal HC fields do not establish equality of the later GDN or residual arithmetic.
Version-1/2 traces retain their original scope: the new HC fields are reported
unavailable, never inferred, and the pending residual remains a composite
attention-path boundary for those comparisons.

Local helper reduction is recorded explicitly in version 2. In that mode native
helpers weight/reduce their rows before returning partials, and the main device
adds those partials after its local combine. `Verifier` has no guaranteed complete
unweighted row matrix, so that field is unavailable. It is not filled with zeros
or reconstructed. Portable still combines complete unweighted rows in original
route order. These are different full-FFN floating-point orders even though the
component kernels follow native methods. The observer changes neither path.

The final combined outputs remain comparable semantic outputs. Without matching
complete inputs, the report cannot attribute a difference locally to combine.
It explicitly reports incomplete row coverage and unavailable input boundaries.
Version-1 traces remain readable under their original required-unreduced-row
contract; they are never reinterpreted as reduced-mode evidence.

Native does not materialize the post-FFN residual at that boundary. That field
is explicitly unavailable. The observer does not reconstruct it or extend into
layer 1. Portable may retain its existing materialized value, which the paired
comparator deliberately does not compare to a different native object.

The trace uses logical F32 `[head][value][key]` state order. Native physical
`[key][head][value]` is byte-permuted only in the downloaded diagnostic image;
GGML already uses the trace order. No floating-point operation or live state
changes. The mapping follows `gdn_step_commit_body` in `verify_kernels.cu` and
the pinned GGML gated-delta state indexing; golden index and bijection tests
cover it.
Convolution history uses `[channel][oldest..newest]`. Committed state is available
on native only when the original T1 window self-commits it. Snapshot copies follow
those kernels, and host observation follows the window's existing completion.
Otherwise state-after fields remain unavailable: no commit or synchronization is
added. Portable observes its already completed eager commit. Unavailable fields
are distinct from zero-filled values or missing/invalid required evidence.

## Bounds and lifetime

Outputs are exclusive-create, owner-readable/writable files. Existing evidence
is never replaced. The writer admits only fixed field shapes, rejects duplicate
or missing required records, caps the whole trace at 8 MiB and seals completion.
A setup/execution/write failure can leave a partial file; the reader rejects it.

Native reserves one 8 MiB device snapshot before expert capacity is sized, so
native free-memory sizing sees that allocation. Reservation refuses insufficient
room while retaining the existing 512 MiB free reserve. Snapshot addresses stay
stable across the unchanged captured window. Download happens only after the
window's existing completion, one field at a time, with an 8 MiB host envelope.
No new stream/event synchronization or numerical kernel is inserted.

Portable charges an additional 8 MiB host bound to its existing admission ledger.
It uses existing host operands plus explicit diagnostic reads, with no observer
GPU buffer or new GPU allocation. The disabled path allocates no observer,
performs no observer downloads, creates no files and adds no graph nodes. Only
optional-owner checks remain in the host coordinator. Native kernels and Vulkan
shader sources are unchanged.

## Exact comparison

```sh
python3 tools/compare_layer0_observer.py native.bin portable.bin new-report.json
```

The reader validates bounded little-endian encoding, complete seal, exact shapes,
finite float fields, unique valid route IDs and disjoint primary/helper ownership.
Different bindings, tokens or ownership stop comparison. Router weights and
expert rows are not numerically compared when their ID association differs.
The report identifies the first difference in documented dependency order and
labels later differences as potentially propagated. Parallel branches do not
become a claimed serial causal chain. A local-boundary candidate requires exact
matching observed inputs and incoming state; it is not causal proof.

Exit 0 means all comparable captured boundaries are exact for this case. Exit 1
means a captured numerical boundary differs. Exit 2 means invalid/missing input,
identity mismatch or output failure. The report always retains `qualified=false`;
missing optional state or reduced-helper rows is not a parity success for those
boundaries. A matching combined output does not establish equivalent reduction
order or complete-input attribution. Existing full-logit screens
and their numerical limits remain unchanged.
