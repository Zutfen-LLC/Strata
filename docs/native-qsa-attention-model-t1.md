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

Model validation is pending. `qualified=false` remains. This does not port output
projection quantization/MMVQ, quantized/rotated KV, competitive model selection
above capacity 2051, native accepted-prefix/window scheduling or full model parity.
