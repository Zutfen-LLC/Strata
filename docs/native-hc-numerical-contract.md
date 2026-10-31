# Standalone HC numerical acceptance, version 1

`tools/compare_native_hc_numeric.py` implements `native-hc-normal-v1`. The exact
comparator stays unchanged and its complete result is embedded as a diagnostic.
Numerical acceptance is an explicit separate decision, never a relabeled exact
pass. Neither result establishes full-model parity, quality or performance.

The fixed method is staged native BF16 fused-GR, H2560/HC4, epsilon 1e-6, T1..8,
no pending writes, original F32 residual/norm and raw BF16 injection. Native
variant 3 and Vulkan guard/finite/repeat receipts are required. Input fixture
bytes, shapes and method tags must match exactly. Pinning binaries, source,
logical features and effective modules remains an operator requirement; a JSON
receipt alone is not independent proof of their identity. Existing mode refusals,
source-order adversarial tests and effective shader tests remain mandatory.

## Error allowance from the operation contract

The validator independently computes source-order square and reduction results
with integer dyadics and exact round-to-nearest-even FP32 primitives. It follows
the compiled first pair `round(y*y)` then x/z/w FMAs, logical warp XOR order and
ordered eight-warp finish. It does not use the runtime C++ host oracle or a GPU.

For each stream let S be that FP32 sum and q=S/2560 the exact rational quotient.
Native division must round q correctly. Vulkan may return a normal representable
m with `|m-q| <= 2.5*max(ulp(m),ulp(RN(q)))`, following GLSL division accuracy.
The checker enumerates the small representable neighborhood, including binade
boundaries. Zero S requires exactly zero mean. Epsilon is the exact encoded F32
constant; `m+epsilon` rounds once. No epsilon fitting or reassociation is allowed.

For every admitted denominator d, compute a correctly rounded direct `1/sqrt(d)`
reference. The validator uses 100-digit Decimal and exact distances to neighboring
F32 values, not the differently rounded expression `1/float(sqrt(d))`.
An actual RS must be within 2.5 times the larger local ULP of that rounded center
and actual value. GLSL's 2-ULP error against the mathematical result may require
at most half an additional ULP when expressed around a rounded reference center.
NVIDIA already defines its quoted 2 ULP against correctly rounded RNE output;
the extra 0.5 ULP is conservative v1 policy slack for NVIDIA, not required
recentering of its quoted accuracy. The existing v1 radius is unchanged.
It applies independently to both backends; Vulkan can
match any denominator admitted by its division interval. There is no fitted
per-fixture or per-device offset and no broad relative epsilon.

NVIDIA lists rsqrtf at 2 maximum observed ULP, explicitly not an exhaustive
accuracy guarantee. GLSL specifies inversesqrt at 2 ULP and division at 2.5 ULP in
its stated range. These motivate the acceptance policy; they do not prove it for
all hardware or inputs. Every tested native and Vulkan result must satisfy the
policy. Strict FMA/RTE controls do not imply identical intrinsic approximations.
Sources: [CUDA floating-point computation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html),
[GLSL precision](https://github.khronos.org/Vulkan-Site/glsl/latest/chapters/variables.html).

This admits the explicitly bounded uncertainty in Vulkan pre-rsqrt division;
it does not claim that captured final RS alone isolates the hardware divide and
inverse-square-root instructions. That unresolved attribution is recorded in
the contract rather than charged to an unexplained final-output epsilon.

## Exact downstream checks and cancellation

For each backend, use its actual captured RS. Compute `(R*w)*RS` with two exact
RNE FP32 multiplications and require every normalized word to match, including
signed zero. Compute injection using the original BF16 values, eight ordered
FMAs per chunk, 40 chunk additions per lane and five XOR additions. Require every
injection word to match. A normalization mismatch makes local dot attribution
unavailable; it cannot be blamed on downstream arithmetic with unequal inputs.
Even a one-bit local association/dot error fails. Thus a large forward-error
bound cannot hide a wrong reduction or numerically tiny erroneous output.

An additional absolute propagation check reports the cross-backend consequence.
For row weights w and normalized arrays a,b, define

```
B = sum_j |w_j| |a_j-b_j| + gamma_53 (sum_j |w_j a_j| + sum_j |w_j b_j|)
gamma_53 = 53u/(1-53u), u=2^-24
```

Each contribution traverses at most eight FMA rounding steps, 40 lane additions
and five warp additions: 53 steps. The standard product-of-rounding-factors
bound yields gamma_53 for normal RNE operations. Sums and the comparison with B
use exact rational arithmetic. The first term propagates the actual accepted
normalization difference; the second bounds both ordered dot roundings.
Require absolute injection difference <= B. This bound holds with cancellation;
no relative error requirement divides by a near-zero reference, no arbitrary
absolute floor is added, and an all-zero sum gives B=0. Exact local dot checks
are still required even if an incorrect output would fall inside B.

## Supported validation domain and refusal

All supplied inputs, parameters, RS and captured outputs must be finite normal
F32 or signed zero (BF16 values are interpreted exactly). All independent basic
operation intermediates must stay normal or zero, with no overflow or underflow;
the nonzero reference division must remain normal. The validator refuses a case
outside that domain even if a runtime silently flushed a tiny result. This is a
conservative qualification subset, not a claim of general denormal handling.
Existing component mode/nonfinite/subnormal refusals are preserved. No runtime
kernel, data conversion, arithmetic, default or synchronization is changed by
this offline validator. A failure is retained; changing criteria requires a new
contract version, not overwriting old exact or numerical reports.

```sh
python3 tools/compare_native_hc_numeric.py /native /vulkan /fresh/numerical.json
```

Exit 0 accepts these captured component boundaries, 1 records a numerical check
failure, and 2 rejects invalid/missing/out-of-domain evidence. `qualified=false`
is always retained: the tool validates a capture; it does not certify every
input, device, execution mode or full model. Exact diagnostic failures stay
visible. Independently pinned vectors from a standalone C++ std::fma oracle
check dense three-token indexing, square/reduction and dot order; that oracle
does not call the Python helpers. Algebraically pinned interval/binade edges
and RNE/fused-cancellation primitives are separate controls. Helper-generated
mock captures test acceptance/rejection predicates; those captures alone do not
independently verify their generating helpers. Predicate tests cover signed zeros,
deterministic operands, accepted intrinsic variation and rejection of
material RS errors, one-bit local changes, malformed bindings and near-zero
escape attempts.

## Path to opt-in model integration and behavioral validation

First qualify this unchanged component on retained real operands, prescribed
synthetics and a small fresh held-out case with the pinned native launcher and
Vulkan stack. Require all structural/identity/lifecycle checks as well as this
contract. Keep exact diagnostics. No performance claim follows from a tiny test.

An experimental layer-0-only serial-T1 seam is now implemented in
`PortableGdnLayer::Impl`; see [native-hc-serial-t1.md](native-hc-serial-t1.md).
Its retained owner leaves are populated after actual residual upload and before
graph consumers, never during graph construction. Physical seam qualification
and model behavior remain separate pending gates.
Charge converted BF16 storage, graph/scratch and component ownership once;
preserve source rank/placement, caps, observer liveness and state commits.
Do not silently enable chunk, pending-write or production-state paths. Default
execution must remain bit-identical with the option disabled. Reject unsupported
configurations before allocation/compute. The experimental model flag remains disabled by default.

Treat HC down/up BF16 conversion, ordered dots, SiLU, sigmoid/gating, mixed
summation/mean and pending writes as a separate native-method slice. Current
portable mixed still uses its original Q8_0 GGML path; the standalone native
oracle discards mixed from its zero dummy down/up matrix. Injection acceptance
cannot certify full HC.

After integration, validate matched model/token/ownership/state: first compare
the initial component boundaries under this contract, then one bounded request
with finite logits, valid state progression/reset, stable admission/counters and
clean teardown. Require default-disabled equivalence separately. Preserve
full logits, top-token margins, route/output changes and exact differences;
do not derive a final-logit tolerance by reusing the primitive ULP allowance.
Use the repository's fixed quality/behavior protocol against a concurrent native
baseline and unchanged sampling/prompts, reporting each verdict and uncertainty.
A behavioral regression blocks promotion even when components pass; a matching
short request alone does not establish quality. Benchmark/performance work and
other execution modes remain separately scoped. No default enablement or merge
is implied by component acceptance.
