# Standalone T1 native HC down/up and mixing

This component is not called by PortableModelSession and adds no model flag or
default. It consumes an identical already normalized F32 operand and raw BF16
down/up rows, H2560/HC4/LR320/T1, with no pending write. Original compat-BF16
conversion, eight-FMA chunks, per-lane chunk addition, XOR offsets 16..1 and
ordered stream summation stay unchanged. Native probes call existing plain and
staged kernels; they do not implement another numerical method.

The old bounded synthetic gate passed at `df488f25812a9a5cac94eabe6630bfa7bcae75b0`.
Both authorized real roles then matched original packer conversion exactly, but
the identical retained real operand exceeded the first qualification domain.
This version defines `native-hc-mix-normal-v2`; its wider-domain hardware gate is
pending. It does not relabel old results or qualify full HC/model behavior.

## Primary precision contract

The [Vulkan SPIR-V environment](https://github.khronos.org/Vulkan-Site/spec/latest/appendices/spirvenv.html#precision-and-operation-of-spir-v-instructions)
defines correctly rounded basic add/multiply and OpFmaKHR, a 2.5-ULP division
allowance for normal denominators through 2^126, and the GLSL.std.450 exp
allowance `(3+2*abs(z))` ULP. Strict FMA and RTE32 must actually be enabled;
NoContraction protects explicit basic operations. Ordinary GLSL.std.450 Fma is
not substituted for OpFmaKHR.

The [CUDA intrinsic table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
reports `2+floor(abs(1.173*z))` ULP for native `__expf`, against correctly rounded
RNE. NVIDIA explicitly describes observed maxima, not exhaustive guarantees.
Every captured result must pass the policy; there is no universal cross-vendor
intrinsic identity claim. These allowances are unchanged, not fitted to data.
Vulkan exp is checked against the mathematical result; native exp uncertainty
is centered on RNE as documented. Division uses the existing conservative
2.5-ULP policy for both backends.

## Derived common normal domain

Let `u=2^-23`, `m=2^-126`, and `q0=m+4*2^-149`. For a proposed symmetric argument
limit A, `r=3+2*A` bounds both exp policies. Local ULP, possible candidate/reference
binade differences, and the native RNE reference's half-ULP displacement give

```
alpha = (r*u*(1+u/2)+u/2)/(1-r*u)
exp_lower = exp(-A)*(1-alpha)
den_upper = (exp(A)*(1+alpha)+1)*(1+u/2)
```

For an admitted finite exp result y and exact t, use
`|y-t|/t <= r*u*(1+|y-t|/t+u/2)+u/2`. Subnormal candidate ULPs are bounded by
`m*u` while t>=m, so the argument also excludes them rather than assuming them
away. The extra half ULP here is conservative domain-proof headroom; it does not
increase the pointwise acceptance radii.

Choose the largest positive F32 A for which `exp_lower>=m` and
`den_upper*q0<=1`. The latter also keeps the denominator below 2^126 and the
smallest sigmoid above normal underflow with division headroom. The independent
100-digit derivation yields **87.33651733398438**, bits **0x42aeac4c**; its next
F32 fails the inequalities. The shader/header constant is tested against that
derivation. No measured fixture extrema enter it. This proves properties of
admitted error intervals, not that every native device obeys an empirical table.

Exp must be positive normal, the rounded denominator must stay in [1,2^126], and
for nonzero numerator a the runtime/checker require
`abs(a)>=RN(q0*den)`. Four minimum-normal ULPs cover threshold multiplication
rounding plus the 2.5-ULP division allowance with strict margin. This is a
sufficient safe quotient predicate, not a maximal classification of all native
inputs. It refuses tiny SiLU results instead of accepting a possible FTZ zero.
Zero numerators preserve their exact sign. Division by rounded denominator one
must preserve its numerator exactly as a local identity.

Positive sigmoid/SiLU saturation occurs only when the original rounded
`1+exp(-v)` becomes one. The algorithm still executes exp/add/divide; there is no
clamp, exp shortcut, input rescaling or forced negative-side zero. Arguments
outside the derived normal domain, overflow, underflow/subnormal cases and
nonfinite inputs/results refuse. Native behavior beyond this subset is not
redefined. Supporting those cases would be a separate semantic qualification.

## Exact checks, cancellation and provenance

The integer-dyadic validator independently requires exact source-order scaled
down preactivations, up dots against each backend's actual low, and Vulkan mixed
output against its actual gates. It checks each captured exp and the activation
computed from that *same* exp, within the separate division policy. Exp interval
enumeration is radius-derived and complete across binades; the old fixed search
window could truncate otherwise valid candidates and is not reused.

Basic projection/product/addition underflow remains unsupported. Exact zeros
and cancellation are preserved; subnormal arithmetic cannot be silently replaced
by zero to satisfy the checker. Shader status checks are not a proof against all
unobserved FTZ intermediates: independent exact local checks remain mandatory.
Propagation uses signed endpoint products and ordered additions, not a relative
final-output epsilon. Native unobserved gates remain interval attribution from
actual native low; they are not claimed as captured exact intermediates.

Synthetic goldens cover lane/chunk/row/stream order, binades, next-domain values,
normal/subnormal quotient edges, signed zero, saturation and common erroneous
outputs. Approved retained-input preflight checks all down rows and bounds up
projections from both backends' admitted low intervals before dispatch. Its
private operands never enter public tests. Actual own-input/own-gate checks,
feature receipts, canaries, repeats and lifecycle cleanup still decide the tiny
hardware gate. No real-role hardware acceptance, model integration, quality,
performance, pending-write, FFN HC, T8 or HIP qualification follows from source
proofs. Defaults remain disabled and PRs stay draft.
