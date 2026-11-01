# Native FP16 T1 QSA attention component

This standalone opt-in component follows `Verifier::record_window()` →
`qsa_decode_attn_batch()` in `src/kernels/cuda/qsa_decode_attn.cu`, then
`native_qsa_gate_apply()` in `native_qsa.cu`. It does not change model execution,
placement, defaults or state authority. Default FP16 KV is explicit; INT8,
rotated/Q4/hybrid, split/Daily/lane-cell/alternative merge variants and native
window scheduling remain unsupported rather than replaced by another method.

The supported geometry is native 24 query heads, two KV heads, dimension 256,
page size four, one text query and cells <=256. Pools retain exact FP16
`[page][KV head][page cell][dimension]` layout. Integer expansion preserves finite
FP16 subnormals. Queries are F32; selected ascending IDs and the native step
remain separate immutable inputs. Negative-one pages are masked without reading
their storage. No resolver, streaming or missing-page recovery is invented.

The chunk shader preserves native 256-thread/eight-warp work, 64-cell chunks,
one rounded multiply plus seven FMAs per eight-dimensional dot partial,
32-lane XOR reduction and scale 1/16. Native per-head maxima, paired exponent
sum/XOR and ascending-cell value FMAs produce the original partial layout.
Capacity 2051 means 33 chunks, including empty capacity chunks. Empty chunks write
only max=-FLT_MAX and sum=0, preserving unused accumulators and 64 scratch floats.
The merge preserves global-max and chunk chronology, explicit exp rescaling and
FMAs, skips sentinel chunks and performs native RNE division with the existing
integer `native_f32_div_rn.glsl` adapter. QSA gate layout and exp/add/divide/
multiply association follow the separate native gate adapter.

Compilation precision is significant. `qsa_decode_attn.cu` uses explicit
`__expf` and ordinary precise division in the default CMake path. The separate
`native_qsa.cu` adapter is compiled with `--use_fast_math`. The probe retains both
unchanged launchers and checks those distinct operations. The source-order
control and existing native exp/sigmoid intrinsic contract are mandatory; there
is no final-output epsilon or coefficient fitting. The existing normal/zero
basic-arithmetic domain, exp-underflow and natural sigmoid zero/one tails are
reused. Nonfinite/subnormal arithmetic and the unqualified sigmoid transition
class refuse explicitly. No clipping or saturation replacement is added.

All inputs are disjoint from detached candidate scratch, diagnostics, attention
and gated output. Preflight invalid geometry/addressing/aliasing dispatches no
work; a failed numerical stage stops before later stages. Completed stages are
counted, candidates never commit persistent state, and repeat/reset/invalid/
retry and canaries are checked. Runtime status encodes chunk/merge/gate stage.
The Vulkan port only substitutes buffer/submission and logical-warp shared
shuffle plumbing, requiring 256 workgroup invocations, 16640 bytes shared storage
and nine storage bindings. Vulkan tensor admission is 2 MiB; the CUDA probe also bounds its primitive-control
allocations within a separate 4 MiB device cap. Three pipeline/status owners
have 192 KiB metadata budget. Existing physical/lifecycle safeguards remain.

The independent C++ control has pinned single-cell, FP16-subnormal, 64/65-cell,
all-masked and eight-term cancellation vectors, plus dense/permuted/missing-page
and natural chunk/merge exp-underflow cases. Separate Python checks enforce the frozen intrinsic bounds, paired sums,
merge chronology/RNE division and own-sigmoid product, rejecting common erroneous
backend values. Effective modules require strict FMA/protected basics and exact
helper transformations; the actual producer output CLI is tested before GPU use.

The native probe calls the original attention and gate kernels. Its partials,
attention and gated output are actual original-kernel results. Per-logit/exp
control diagnostics are independently derived from the source-order input and
separate CUDA primitive calls, **not direct snapshots inside the native chunk**.
The one-input gate control also uses the unchanged native gate launcher. Reports
keep this distinction; derived diagnostics cannot establish a native intermediate
first-divergence claim. No model weights or captured requests are needed.

Local build/host checks are development evidence only. Native CUDA compilation
and native02 then Vulkan05 physical comparison are required next. No full native
QSA, accepted-prefix/window, quality or performance parity is claimed, and no
model flag/default is enabled by this component.
