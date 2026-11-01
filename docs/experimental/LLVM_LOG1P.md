# Standalone LLVM libc positive log1p candidate

PR118's Metal-expression STOP is preserved. This separate candidate adapts LLVM
libc commit `4497566c7ca6b1d958fe27678f541308b948f6c2`, accurate non-FMA/RNE configuration.
See third_party/llvm-libc-log1p/NOTICE.md, original headers and full Apache-2.0
WITH LLVM-exception license. No coefficients, cutoff or output tolerance are fitted.

The outer native Strata method remains `gdn_ab_multi`: reduce alpha, add dt,
select strictly v>20 ? v : log1pf(__expf(v)), then multiply ssm_a. This primitive
probe is not integrated there. It retains the earlier signed-zero or positive
normal <=2^29 domain and unchanged one-adjacent-F32-step accuracy gate. Other
values/modes refuse. LLVM source claims correct rounding; the Vulkan translation
must additionally match the original non-FMA source algorithm exactly on tests.
CPU proof and upstream guarantees are not a guarantee for a new Vulkan compiler.

Binary64 is required, but FP64 FMA is not. The pinned GGML factory already passes
supported core physical features into device creation. This change records its
actual shaderFloat64 creation field after successful creation without requesting
new features. The candidate refuses unless physical support, logical enablement,
RTE64 and signed-zero/Inf/NaN preservation are present. It adds those FP64 execution
modes, retains existing FP32 controls, and protects every basic operation against
contraction. It uses no approximate log, reciprocal, division or FMA instruction.
There is no double subnormal arithmetic in the bounded domain; no FP64 denorm
policy is introduced. Unsupported capability is a stop, not a fallback.

Original accurate range reduction, C_LOG compensation, tables, polynomial
associations and positive in-domain hard-case returns are preserved. Tables and
coefficients retain exact binary64 words. Word-pair exponent extraction avoids
shaderInt64. RNE-only helper returns omit errno/floating exception flags; other
rounding modes and negative/special branches are explicitly outside this probe.

The independent host oracle compiles original upstream algorithm bodies with
include-only adaptation and minimal non-FMA/RNE support shims. A separate compiled
copy of the GLSL adapter matches it on 100,000 deterministic inputs plus hard cases.
Independent Decimal checks cover three original fixtures and a held-out fixture
with branch/table boundaries and upstream hard cases, rejecting incorrect results.
The GPU probe exports final F32, intermediate F64 bits and branch tags, requiring
exact local source-order control and the unchanged primary numerical gate.

Tensor and metadata bounds are each 64 KiB. Input/canary, alias/refusal/retry,
repeat/reset and effective-module checks reuse existing components. Native02 then
Vulkan05 use the established pressure, owned-process, resource and cleanup guards.
Fixtures are generated locally; no model, capture or role extraction is needed.
`qualified=false` and no model/a-b/default enablement remain. Hardware and logical
capability evidence are required separately from local CPU/build/SPIR-V checks.
