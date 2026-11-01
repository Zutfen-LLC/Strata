# Experimental native T1 GDN recurrence

This ports `Verifier::record_window()` → `gdn_step_norm_multi()` in
`src/kernels/cuda/verify_kernels.cu`, not a replacement recurrence. The bounded zero/sparse/dense component comparison passed the actual unchanged
CUDA launcher and Vulkan own-primitive state/output controls, reset/repeat and
module checks. The selected native delta contraction matched. Dense cross-backend
bits differ; this is not universal intrinsic or full-model parity.

`--native-gdn-t1 PREFIX` adds a default-disabled model path for the 35 ordinary
GDN layers; layer1's PLE branch stays explicitly unchanged. It requires eager T1,
primary dense topology, strict accelerator execution and explicit resident state
and attention weights. Earlier HC choices remain independent and must be matched
between comparison arms. No default or native placement policy changes.

The initial supported shape is S128/HK16/HV48, one token, keep one, no fused Q8
output. State is native `[row,head,column]`, address `(row*48+head)*128+column`.
Convolution, q/k L2 normalization, a/b projections, input/output matrix arithmetic
and their quantization remain the existing portable operations; this does not
qualify them as native. The component consumes identical post-convolution q/k/v,
decay-log gate, beta, output-gate preactivation z, gamma and initial state.

The shader uses native 128-column/four-row-group geometry, sequential 32-row FMA
chains, left-associated four-group reductions, state update, q projection,
`rsqrt(128)` scaling, XOR lane reduction and four-warp sum, epsilon, output RMS
scale, gamma and sigmoid in native order. The delta's `v-g*kv` is represented by
an explicit FMA, following CUDA's normal contraction. This compiled detail is an
explicit qualification question: the reference checks actual native next state
and y against an independent control and refuses on a mismatch. It must not be
resolved by loosening a tolerance or changing the native build. The reference
uses original T1/keep1/non-QFUSE launcher selection. It does not replace the native
kernel; separate diagnostic kernels obtain the same intrinsic functions for
source-order checking. CUDA/HIP/Metal/SYCL variants are not interchangeable claims.

The Vulkan API reads immutable state and writes a separate candidate. The layer
seam runs its existing prefix into persistent post-convolution leaves, completes
the component, then runs the existing output projection/HC/FFN consumer. Only the
existing successful layer transaction copies candidate state/history into committed
storage. Reset, failed component and failed FFN retain their existing semantics.
Prefix trace tensors are graph outputs, and component leaves survive both graphs.
Eager T1, resident state and attention weights, strict accelerator/no CPU
fallback and a receipt path are required. PLE, chunk/batch,
production-policy and speculative accepted-prefix modes explicitly refuse. This
bounded serial seam is not native window scheduling or accepted-prefix commit.

The bank has one candidate state plus small h/g/b/z/y/diagnostic/history leaves;
immutable state and gamma reuse existing owners. Aligned actual allocation and
pipeline metadata are charged against the existing graph budget before scheduling.
Both prefix/suffix scheduler storage/workspace are counted, and a 4 MiB additional
host bound covers the largest one-at-a-time 3 MiB input preflight and metadata.
No capacity/reserve or placement policy changes. Eleven descriptors express the
same data within GGML's twelve-descriptor limit; four diagnostics share a bounded
auxiliary allocation. A 512-invocation workgroup, shared storage and descriptor
limits are checked before dispatch. Existing strict FMA and logical FP32 feature
checks remain mandatory.

## Numerical contract v2: native exponent tails

Inputs and recurrence/basic intermediates/results must be finite normal or zero;
subnormal recurrence arithmetic still refuses. Decay-log inputs may be any finite
normal/zero nonpositive value. Native `gdn_step_norm_multi` evaluates `__expf(gate)`
directly, without a minimum-log clamp. Its actual exponential must be normal or
zero and satisfy the original pointwise exp bound, allowing permitted underflow/
FTZ zero only where that bound reaches the subnormal range. Preserved subnormal
exponential values refuse before commit; this does not introduce an FTZ mode or
replace the intrinsic. For example, the retained model refusal at
`-138.10023498535156` has real exp approximately `1.0564e-60`, far below half the
least F32 subnormal: rounded zero is legitimate, not a source input error. The
original normal-only preflight rejected this valid native class before dispatch.

The only other exponential in this numerical family is output sigmoid
`1 / (1 + __expf(-z))`. It reuses the already qualified HC up-sigmoid tail contract:
normal domain at the derived limit, any positive finite tail only with its actual
rounded denominator exactly one and sigmoid exactly one, and negative z at most
-89 only with actual +Inf exponential/denominator and exact positive-zero sigmoid.
The mathematical overflow boundary lies below 89; the integer ceiling leaves the
narrow transition/subnormal-quotient class explicitly unsupported. Positive-tail
exp diagnostics may be subnormal/zero under the primary exp/FTZ allowances because
the unchanged denominator addition must absorb them. No such subnormal feeds the
recurrence. Negative sigmoid transition inputs between -89 and the derived normal
limit refuse explicitly. No clipping, saturation assignment, exp shortcut, changed
native build flags, new denorm execution mode or final-output epsilon is added.
The original exp/add/divide and all recurrence FMAs execute in their original order.

The independent host control uses exact F32 FMA/basic-operation association with
contraction disabled outside explicit `std::fma`. State, output reduction and final
y must match bitwise using that backend's actual exp/rsqrt/sigmoid primitives.
A separate primary-spec exp/division validator checks those primitives, including
the permitted underflow and exact natural sigmoid tail identities.
For the positive normal norm argument, reciprocal square root reuses the existing
2.5-local-ULP v1 bound about its independently computed RNE result, including binade
edges. For CUDA, 0.5 ULP is conservative policy slack beyond its documented 2 ULP
RNE-relative accuracy, not required recentering. This is a local intrinsic bound,
not a tolerance on state or y. Zero/near-zero/cancellation results still undergo
exact local checks; no relative error is divided by a near-zero output. The mean
uses division by the exact power of two 128. Its rounded normal result plus the
unchanged `1e-6f` epsilon is checked as part of the source-order norm argument.

References: [Vulkan floating point rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html),
[CUDA intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html),
and [CUDA mathematical function accuracy](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
Cross-backend state/y differences are diagnostic. Accepting each backend's bounded
intrinsics and exact local order does not establish equal model state, full
numerical parity, quality or performance. The first tiny qualification must report
actual native compiler-control success separately from this source transcription.

The synthetic cases are zero, sparse pinned hand-calculated values and dense
signed/cancellation-sensitive values. The host regression also has independently
pinned dyadic dense indexing vectors across row-group/head/column boundaries and
rejects the same wrong output on both backends. Device probes repeat from the
same initial state. Vulkan also checks input/canary preservation and invalid-input
noncommit. No new weights, model capture or full-model execution is needed.

## Shared model integration

The model reuses the existing shared scheduler slot1 for the prefix and slot0 for
the suffix. No fourth scheduler or state authority is introduced. Persistent
component leaves and native `[row,head,column]` committed state survive leases;
all graph pointers are cleared when the lease releases its metadata. The prefix
and suffix allocations/workspace, aligned banks and actual resident state are
admitted through the original ledger and planner. Raw bank payload over 35 layers
is 120,162,700 bytes before alignment/metadata, without copying existing state or
gamma weights. Both comparison arms explicitly use the same state residency;
previous expert quotas are not a fit promise.

Only layer0's first actual call retains a bounded own-input fixture and candidate
outputs. Its committed-state receipt is read after the existing successful commit;
diagnostic I/O cannot retroactively reject that committed transaction. Other
layers retain modules/counters, not bulk inputs. Any component refusal preserves
its current immutable inputs, position and an explicit unavailable-output marker.
Request-boundary hashes of the eager host mirrors diagnose state progression and
reset replay, not cross-backend state equivalence. The offline checker replays the
first actual component against its own primitives and compares candidate/actual
committed bytes. No separate native capture or new raw weight extraction is needed.

The next model comparison keeps HC fixed, uses one fresh native-planner load and
same-image control/on arms with identical ownership. Admission must succeed under
unchanged caps/reserve; otherwise stop with its actual reason. Own-input component
contracts, exact within-arm reset/replay, finite head/routes and actual paid work
are distinct from cross-arm logit/token diagnostics. Accepted-prefix commit,
production state/window scheduling and full-model numerical/quality/performance
parity remain unclaimed.
