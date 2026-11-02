# Experimental native FP16 T1 QSA attention in the model

This default-disabled adapter replaces only attention chunk/merge and the separate
output gate in ordinary serial QSA layers. Its specification is
`Verifier::record_window()` → `qsa_decode_attn_batch()` →
`native_qsa_gate_apply()`, using the default SM8 FP16 T1 launchers from
`src/kernels/cuda/qsa_decode_attn.cu` and `native_qsa.cu`.

`--qsa-attention-banks PREFIX` allocates the same detached candidate bank in
control and on. `--native-qsa-attention-t1` selects the native method. Both require
native normalization/RoPE, native indexer and native scores/IDs, strict selected
Vulkan execution, reusable eager scratch and context at most 256. Production,
chunked, alternate native attention dispatch and rotated/quantized KV modes
refuse. These are backend plumbing constraints, not alternative algorithms.

After the actual project/normalization submissions complete, the adapter copies
the real normalized query, raw projected Q/gate, ascending selected IDs and
current FP16 KV candidate into the native paged layout. Integer half bits are
preserved. Pages have four cells, two KV heads and 256 channels. All previous
accepted KV and the current uncommitted cell are included; no synthetic model
operands or repeated gate are used. The actual native gated result is an input
leaf of the existing output projection. Its output and subsequent HC/FFN method
remain unchanged.

The owner publishes no persistent state. Existing FFN-success candidate/commit
semantics remain responsible for KV and indexer state; numerical/attention or
FFN failure leaves state, token count and last residual unchanged. Reset uses
those same owners. Stage counters include completed chunk, merge and gate work,
including stages paid before a later refusal.

Admission charges actual aligned 13-tensor storage and three status/workspace
owners against the existing graph budget. The bank is about 1.87 MiB at context
256 before backend alignment and owner metadata. Eight MiB per QSA layer is a
conservative host allowance for transport/readback, first-call serialization,
metadata and input validation; existing weights are neither duplicated nor moved.
The legacy scheduler reservation is retained in both arms. Planning uses the
materialized control with matched banks, then freezes its native placement for
both arms; the on arm must independently admit that placement within unchanged
caps and physical reserve.

CPU tests exercise the real graph's output projection input, token/head/page
mapping, explicit ID/query dependencies, actual stage counters, budget refusal,
attention/FFN noncommit, retry and reset. Only the Vulkan primitive is mocked in
those tests; they are not hardware numerical evidence. PR132's eight physical
component cases already passed the unchanged own-input contracts. The model
handoff runs fresh planning and same-image control/on, two reset-separated
25-token prompt/GEN3 requests each. Initial layer3 receipts check own primitives,
source-order partials/output, exact query/ID/gate and WO links. Within-arm raw
heads/routes/state digests must replay exactly. Cross-arm differences are
reported without a new final-logit tolerance.

Bounded on-only model validation passes as reported below. `qualified=false` remains. This does not port output
projection quantization/MMVQ, quantized/rotated KV, competitive model selection
above capacity 2051, native accepted-prefix/window scheduling or full model parity.

The initial bounded run at `4ca2d22d` built 31 focused tests and passed planning
and two control requests, but on refused allocation before READY/GEN. A real
GGML CPU allocator regression reproduces the same admission error for the native
combine graph: the independent residual output is an unconsumed leaf, omitted
by the node/source-only estimator. The correction supplies the phase's actual
forward roots and charges only previously unaccounted leaf storage, using each
backend's allocation size/alignment. Actual buffers, graph arithmetic, placement
rules and caps are unchanged. Materialized control/native gated allocation in
the isolated regression is 90,112/75,776 bytes both before and after correction;
only the latter's admission previously failed. Reused control evidence is
historical source-compatible evidence, not a newly executed same-image pair.

Hermes reports the corrected bounded continuation passes at `0079b358`: fresh
admission with retained native-selected plans, one on-model load and two
reset-separated fixed 25-prompt-token GEN3 requests, both `[18,24,16]`. Each
ordinary QSA layer runs 28 chunk, merge and gate calls per request. Layer3
positions 0..7 pass own-input numerical contracts and exact query/ID/gate-to-WO
links; 72 attention and 566 preceding modules validate. Within-arm state and
routes replay exactly. The historical source-compatible control is reused;
this is not a new same-image comparison. All owned processes exit/reap, no
GPU owners remain, and three cleanup observations restore baseline. The 922
resource samples are discrete; sampled owned swap is zero.

The first sealing attempt exceeded the evidence-copy bound. Offline completion
passes without a workload rerun or cap increase: copied evidence is 75.31 MiB,
sealed evidence 75.60 MiB, with 2,701 manifest entries, 3,076 authenticated
historical control/planning references and five large current-artifact references.
All originals and partial outputs remain unchanged. Earlier allocation and
runner STOPs remain recorded. These are operator-reported results, not a claim
of independently rerun hardware validation. Full native numerical/state/window,
quality and performance parity remain unproven; defaults remain disabled.
