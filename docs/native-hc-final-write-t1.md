# Experimental T1 native final FFN write component

This standalone component materializes a residual candidate with the selected
native `native_gr_post` method. It is not wired into the portable model. Existing
model defaults, pending-read shader and command-line flags are unchanged.

The fixed shape is four streams of 2560 F32 values, one 2560-value block output
and four injection values. Stream-major indexing shares the block column across
streams. The selected native source is `src/kernels/cuda/native_gr_postops.cu`:

```
scaled = FMA(0.25, injection[c], +0)
sigma = 1 / RN(1 + exp(-scaled))
weight = FMA(2, sigma, +0)
candidate[c,d] = FMA(block[d], weight, original[c,d])
```

Both explicit scale operations retain positive-zero bias and the final residual
operation rounds once. The Vulkan write variant reuses pending component buffer
validation, strict-FMA feature checks, device ownership, arithmetic status and
bounded allocation. The pending-read variant keeps its existing operations.
The factory is explicit; callers use the existing run/destroy/bytes interface.
All input/output spans must be disjoint. Producer storage is immutable until the
caller accepts a candidate after consumers succeed. No in-place materialization
or cross-layer pending lifetime is introduced here.

The synthetic coordinator executable admits at most 128 KiB of tensor storage,
64 KiB of component owner storage and 64 KiB of tensor metadata. Domain-preflight
host vectors are bounded below 1 MiB. It checks short-budget and alias refusals,
input/canary preservation, rejected-consumer noncommit, explicit accepted host
materialization and reset/replay. Those tests are not an actual model transaction.

## Numerical contract

Normal-or-zero finite F32 inputs and every explicit arithmetic result are required;
subnormals, nonfinite values, overflow and out-of-domain activations refuse.
The activation domain and exp/div allowances are the existing derived normal-domain
v2 policy in `native_hc_mix_numeric.py` and `native_hc_mix_limits.h`, not limits
fitted to these fixtures. The selected native CUDA file uses `--use_fast_math`;
its actual launcher remains the required hardware oracle. No CPU exp value is
substituted for a backend value.

The checker bounds the positive weight by propagating the existing exp interval
through rounded denominator addition, the existing quotient interval and exact
power-of-two FMA scaling. Endpoint propagation is valid because this positive
normal sigmoid composition is monotone. Vulkan also exports its actual exp and
must satisfy the own-exp quotient association. Each final word must equal the
independent rational/RNE FMA using that backend's actual weight, with no relative
or absolute residual epsilon. This handles cancellation and near zero without a
relative-error exception. A material wrong residual is rejected even when both
backends return identical wrong words. Cross-backend bit differences remain
reported diagnostics; component acceptance is not full-model qualification.

The CUDA test invokes the existing `native_gr_post` launcher without changing its
implementation. A second identical-injection invocation with zero residual/unit
block exposes the actual positive weight exactly through FMA(1,weight,+0).
Uniformity is checked across all columns. Original inputs remain immutable.
No native exp intermediate is claimed. Native build/execution must be reported
separately when the local CUDA toolchain is unavailable.

## Existing real tuple and next integration boundary

`verify.cpp` already snapshots `pending_residual` after the FFN HC read, actual
`combined_output` after local/helper combine, and actual FFN `inject`, on their
existing CUDA stream before reuse. These three STRL004 fields are exactly the
final-write tuple. `tools/extract_native_hc_write.py` recovers only these fields
from a sealed, expected-hash/binding original trace. It does not infer a block
output from HC mixed, a GDN output or a rounded final residual. No observer
extension, new roles or model invocation is needed when that trace is retained.
The extracted fixture has 51216 operand bytes plus its eight-byte version header.

Product integration remains separate: preserve the real producer lifetime,
complete ledger, disjoint candidate and commit/rollback/reset semantics. The
original native preset materializes before layer-1 PLE; it must not carry a tuple
through PLE just to avoid a write. Other deferred/control/head boundaries require
their original semantics. This component does not qualify those paths, arbitrary
shapes, T8, full HC, model quality or performance.
