# Experimental model layer-0 FFN pending read

`--native-hc-ffn-read-layer0 RECEIPT_PREFIX` connects the existing actual GDN
projection and attention injection to the layer-0 FFN pending read. It implies
the existing native attention norm/injection and mixed read. It does not create
a diagnostic prior tuple, change native arithmetic or enable deferred final FFN
write/next-layer carry. The option is off by default.

The caller must explicitly request `--layer0-resident-weights` or the existing
full `--dense-resident-weights`. The new narrow residency scope applies only to
layer 0, including its FFN, and can also be used by a matched control. It does
not change the attention-only meaning of `--gdn-resident-weights`, other layers,
the head, dense topology or expert placement. Serial eager strict accelerator
execution, no legacy observer, supported environment modes and a receipt prefix
are required. Existing non-PLE, shape, capability and ABI guards still apply.

Model preflight charges the attention and FFN conservative host envelopes before
construction. The FFN uses its existing owner and replaces the three Q8 HC banks
with original compat-BF16 conversion, without duplicate resident banks. Actual
aligned weights, scratch, graphs and backend-private allowances remain in the
existing complete ledger. Caps and physical reserve are unchanged. Native expert
capacity and ownership must be derived after dense/head allocation with the
existing planner; a historical quota or the weight delta alone is not admission.

Both FFN HC and the final residual consumer use the existing disjoint candidate.
Original producer lifetime, recurrent/history/token commit, rollback and reset
semantics are unchanged. `native_ffn_hc_layer0_calls_delta` reports successful
actual component calls in a request, including paid work before a later provider
failure. A failed component is not credited. Reset does not erase paid calls.

Focused tests exercise real model option preconditions, explicit residency scope,
actual model-to-layer configuration and GDN producer/consumer with mocked device
HC arithmetic, independent source-order materialized control, replacement ledger,
short-room refusal, paid counters, rollback and disabled CPU model logits.
Standalone real-role component evidence does not establish actual-model FFN or
full-HC parity. A new model image still requires its own strict backend, complete
planner admission/freeze and matched model validation. Initial release is
CPU preparation and READY/QUIT planning only; GEN/pair needs a separate plan
review. No default change, quality tolerance or performance claim follows.
