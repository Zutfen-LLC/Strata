# Experimental native T1 GDN recurrence

This ports `Verifier::record_window()` → `gdn_step_norm_multi()` in
`src/kernels/cuda/verify_kernels.cu`, not a replacement recurrence. It is an
unqualified component and a default-disabled `PortableGdnOptions` seam, with no
model CLI switch. Physical comparison against the unchanged CUDA launcher is
required before extending its use.

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
Private eager T1, resident state and attention weights, strict accelerator/no CPU
fallback and a receipt path are required. Shared scratch, PLE, chunk/batch,
production-policy and speculative accepted-prefix modes explicitly refuse. This
bounded private seam is not native window scheduling or accepted-prefix commit.

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

## Numerical contract v1

Inputs/basic intermediates/results must be finite normal or zero; subnormals
refuse. Decay-log gate is in `[-87.33651733398438,0]`; z has magnitude at most
`87.33651733398438`. This reuses the pre-existing normal-domain exp/division
contract, not the HC up-sigmoid tail extension. Unsupported domain results refuse
without committing. There is no clipping, alternate sigmoid or output epsilon.

The independent host control uses exact F32 FMA/basic-operation association with
contraction disabled outside explicit `std::fma`. State, output reduction and final
y must match bitwise using that backend's actual exp/rsqrt/sigmoid primitives.
A separate unchanged primary-spec exp/division validator checks those primitives.
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
