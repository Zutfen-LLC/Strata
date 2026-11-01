# Standalone T1 native pending-write/read composition

This is a component experiment, with no model option, default change, scheduler
rewrite or product state commit. It extends the separately selected HC method
only through an explicit new API; the original no-pending API still refuses
pending writes. Fixed geometry is H2560/HC4/LR320/T1, FP32 activations, raw BF16
projection banks and epsilon 1e-6. Existing alternative HC modes refuse.

The native reference is the existing `fused_gr_read_multi` selected staged
variant 3 with `FusedGrArgs::apply=true`, in `src/kernels/cuda/fused_gr.cu`.
Its pending update is:

```text
scaled[c] = previous_injection[c] / 4
exponential[c] = __expf(-scaled[c])
gate[c] = 2 * (1 / (1 + exponential[c]))
Rprime[c,d] = fma(previous_block_output[d], gate[c], R[c,d])
```

Rprime is formed before the original square/reduction/normalization order.
The actual native up pass also materializes that same update into its owned R
copy. The test-only gate probe observes the unchanged expression in the same
native translation unit; the production launcher and kernels are unmodified.
The standalone native executable restores its original R before each repeated
call and preserves prior block-output/injection. Local exact FMA checks against
the actual launcher output also test whether the probe represents its gate.

Vulkan computes Rprime into a disjoint candidate, then invokes the already
existing norm/injection and down/up/mixed components. This is backend plumbing
for a standalone component, not a claim to reproduce native deferred graph
scheduling. Original R, block output and previous injection remain immutable;
the caller must not publish a candidate after any failure. There is no commit
API here. Model lifetime integration is a separate gate. All input/output spans
are contiguous, same-device and disjoint; unsupported shape, type, overlap,
feature/strict-FMA or metadata/storage limits refuse. The test program admits
at most 16 MiB of device tensor storage; small component owners use existing
bounded metadata/status allocation. No resident model weights or roles are read.

## Numerical contract: native-hc-pending-t1-v1

This composes existing contracts without a new fitted cross-backend epsilon:

- Require normal-or-zero finite inputs and accepted intermediates. Reject
  subnormal/nonfinite results, overflow, quotient underflow and unknown modes.
  The host preflight is conservative and is not a CPU fallback. Device status
  is checked before norm/mixing consumers run.
- Reuse the specification-derived normal sigmoid domain from
  [mix v2](native-hc-mix-component.md), applied to previous_injection/4, not to
  observed fixture extrema. This bounds exp away from under/overflow and keeps
  1+exp in the existing guaranteed division range. Zero injection and natural
  positive saturation keep the original operations; no clamp/shortcut is added.
- Check each backend's observed exp against its existing intrinsic allowance.
  Check observed gate/2 against the existing division interval for that SAME
  observed exp and exact rounded denominator. Scaling by 2 is exact in the
  accepted normal domain. The native observation is separate from actual
  normalization; exact Rprime checks verify its consistency with the launcher.
- Require every Rprime word to equal the correctly rounded FMA of original
  block output, observed gate and original R. Separate multiply/add is not
  accepted. Cancellation uses this exact local test, not relative error, so
  near-zero cancellation cannot hide behind a broad epsilon. Exact zero is
  valid; subnormal cancellation refuses. There is no assertion that both
  backends' Rprime words match when their permitted sigmoid values differ.
- Apply [norm v1](native-hc-numerical-contract.md) to each backend's ACTUAL
  Rprime, with exact weighted-input/reduction/downstream association checks
  and the existing own-input RS interval. Then apply mix v2 to its ACTUAL
  normalized values. Do not substitute one backend's operands for the other's.
  Native mix exposes low/mixed only; its existing bounded unobserved-gate
  propagation limit remains, rather than inventing captured preactivations.

`tools/native_hc_pending_numeric.py` checks this contract. Cross-backend raw-bit
comparisons remain diagnostics. This proves bounded own-input method behavior;
it is not universal intrinsic identity, FFN role identity, full HC, model quality
or performance acceptance. qualified and full_hc_parity remain false.

## Validation and bounded next gate

Host controls pin the independent dyadic FMA result
(1+2^-23)*(1-2^-23)-1 = -2^-46, distinguish separate multiply/add, and check
stream/column indexing, exact zero cancellation, original tuple preservation,
mode/shape/nonfinite/underflow refusal and unchanged no-pending rejection.
The Python checker rejects identically erroneous outputs on both backends.
Independent C++/Python generators produce byte-identical public synthetic
zero/cancel/sparse fixtures. Effective shader tests use the existing pinned
GGML helper, require FmaKHR and protected basic operations, test helper variants
and refuse a modified FMA. These are CPU/build checks, not physical evidence.

The next physical gate is these tiny synthetic cases through actual CUDA then
Vulkan, with immutable fixture/binary/runtime binding and the existing reviewed
policy/OwnedSwap runner. CUDA compilation is an operator prerequisite when the
local compiler is unavailable. Preserve source/cache/binaries before advancing
and regenerate existing CMake without changing cached options for new targets.
No model load or real FFN extraction is needed. A successful gate can justify
reviewing a later pending tuple lifetime/FFN read slice; it cannot silently
widen the current serial layer0 model flag.
