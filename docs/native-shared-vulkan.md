# Native shared expert on Vulkan

This opt-in component observes the native shared-expert method. The portable
FFN/model can select it explicitly; all defaults remain off. Build
with `STRATA_VULKAN_NATIVE_STAGES=ON`,
`STRATA_VULKAN_NATIVE_Q8_STRICT_FMA=ON`, `GGML_VULKAN=ON` and
`STRATA_BUILD_TESTS=ON`. Run `native_shared_input_test --write NEW_FILE H F T`
to create a synthetic input. The fixture carries ordinary contiguous Q8_0
projection bytes, raw BF16 scalar weights, and original FP32 activations.
There is no model download.

The executables accept the same arguments:

```
native_shared_component --device INDEX --expect-device DESCRIPTION --fixture FILE --output NEW_DIR
native_shared_reference --device INDEX --expect-device DESCRIPTION --fixture FILE --output NEW_DIR
python3 tools/compare_native_shared.py --fixture FILE --native CUDA_DIR --vulkan VULKAN_DIR
```

`native_shared_reference` requires a CUDA build. Both runners refuse an existing
output directory and verify the device description. A failed comparison or
nonfinite output retains the diagnostic files. Operational fleet paths and
physical-test evidence belong in the private handoff repository.

## Native binding and arithmetic

The scalar weight is `ffn_gate_inp_shexp.weight`. Native `layout.cpp` declares
`WeightKind::Bf16InF32`; `layer.cpp::moe_shared` requires that kind and consumes
BF16 storage. `weights.cpp::Conv::Bf16High` takes the high half of an exactly
promoted BF16 container. Rows marked `raw16` copy two-byte values unchanged.
This is recovery of BF16 storage, not a policy for rounding arbitrary F32.
The fixture stores raw BF16. `native_shared_promoted_bf16` accepts only finite
F32 images with a zero low half, then performs that exact native recovery.
Portable FFN's accepted F32 scalar tensor is not automatically eligible. Model
integration must establish provenance/binding instead of silently truncating it.

Source correspondence, inspected at the PR's parent `02cef471`:

| Boundary | Native source | Vulkan implementation |
| --- | --- | --- |
| Original input to Q8_1 | `native_mmvq.cu::native_quantize_q8_1_kernel`, `q8_1_finite.hpp` | Existing `strata_vk_native_stages_create_quant_rn_observation`, mode 0; exact division for scale and code ratio, half RNE and finite clamp |
| Dense gate/up/down | `native_mmvq.cu::native_small_mmvq_kernel<Q80Block,8>`, `SmallTraits<Q80Block,8>`, `native_mmvq_multi_kernel` | `native_shared.comp`, mode 0; 128 logical threads, `kbx=tid/4`, `iqs=2*(tid%4)`, stride 32 blocks; two packed signed dots, rounded scale product then strict FMA; warp partials added 1,2,3 before XOR offsets 16,8,4,2,1 |
| SwiGLU | `shared_expert.cu::native_swiglu_kernel`, `native_mmvq.cu::native_swiglu_quantize_q8_1_kernel` | Existing stage mode 1, then RN quantizer mode 0; materializes the unrounded F32 boundary for capture |
| Scalar dot | `native_bf16.cu::bf16_f32_mmvf_kernel`, `_multi_kernel`, `mmvf_block_size` | `native_shared.comp`, mode 1; original F32 inputs, two ordered strict FMAs per BF16 pair, identical block-size search and two XOR reductions with zero-padded warp partials |
| Sigmoid and output scale | `shared_expert.cu::launch_sigmoid_scale_rows` | `native_shared.comp`, mode 2; FP32 sigmoid then separate output multiplication |

The dense projection is not the grouped-expert `row_dot`. The latter reduces
one warp per row. Dense MMVQ combines four warps before its final XOR tree.
Small-K row packing and multi-token weight reuse affect placement, not the
per-row accumulation translated here. The standalone component uses one
workgroup per output row/token. Native CUDA supports additional quantized
projection types; this component explicitly supports Q8_0 only and does not
restrict or modify CUDA's other formats.

The CUDA runner calls the existing quantizer, dense MMVQ and BF16 MMVF
launchers. Two diagnostic unary kernels capture SwiGLU and sigmoid/scale using
the native expressions. It independently invokes `shared_expert_multi` with
native BF16 enabled and checks that its final output exactly equals the
captured sequence. No copied dense CUDA kernel serves as an oracle.

## Limits and safety

T is 1 through 8. Both dimensions are multiples of 32 in 32 through 8192,
covering existing shared-expert dimensions, including 2560 by 640. The bound
also keeps indexing and allocation arithmetic within range. This is a bounded
component, not support for arbitrary GGUF layouts, mixed projection formats,
legacy BF16 activation rounding or deferred gate application.

The fixture validates finite activation/weight values before allocating device
tensors. It caps data at 256 MiB; the Vulkan runner checks aligned allocation
size before allocation and actual size afterward. Component metadata has a
caller-specified budget (64 KiB by default), checked before and after allocation.
Pipeline creation reuses strict feature admission, final effective-module
validation and exception-safe ownership. Callers serialize synchronous calls
and keep the backend alive until the component is destroyed. Each kernel call
checks tensor types, exact spans, alignment, device identity, non-overlap and
dispatch limits before submission. No per-call device allocation is introduced.

Every fixture view has prefix/suffix sentinels. Inputs and the unused dummy
span must remain unchanged. Captures are `qx`, `gate`, `up`, `h`, `qh`, `down`,
`raw`, `sigmoid`, and `out`, plus the five original inputs. All `.bin` files
contain little-endian words of the active view, without sentinels; `report.json`
records the full readback guard check. Q8_1 is nine words per 32 elements;
BF16 scalars are two values per word.

The comparator requires identical captured inputs and valid receipts. It checks
input quantization's consumed scale/codes, gate/up on identical quantized input,
the raw scalar dot, requantization on identical hidden input, down on identical
consumed hidden codes, and final multiplication on identical inputs. It reports
unused Q8_1 sum-half differences separately. Exp and CUDA fast division can
differ across vendors, even with identical arguments. SwiGLU, sigmoid and
propagated differences are observations; comparison success always retains
`qualified=false`. A retained H2560/F640/T8 CUDA/Vulkan pair has identical
quantized inputs, gate/up/down projections and raw scalar dots. Nonlinear and
final output differences are at most one ULP on that fixture; final scaling
replays exactly from each captured sigmoid and the identical down projection.
There are no exp/reciprocal intermediates for individual-instruction attribution.
There is no throughput or broad-quality claim.


## Portable FFN and model opt-in

`PortableFfnOptions::native_shared` and `PortableModelOptions::native_shared`
require a strict accelerator, no CPU fallback and a nonempty private
`native_shared_receipt_prefix`. The CLI is `--native-shared PREFIX`. It requires
the strict-FMA native-stage build and the component's T1..8/aligned dimensions.
All three shared projections must be Q8_0. The scalar accepts raw BF16 (GGUF
30) or an exact promoted BF16 value in F32 (GGUF 0), with the finite checks above.
An arbitrary F32 scalar is refused before device weight allocation. No policy
for rounding it is introduced. Router and combine opt-ins are independent.

In this mode the pre graph produces the same live mixed activations, router
logits and inject values; its generic shared projections and sigmoid are omitted.
The native component reads that mixed tensor directly. Its finite gated result
is passed unchanged to the existing routed/shared combine boundary. The default
branch still builds and executes its original generic shared graph.

The four immutable shared weights have one device allocation shared by the T1
runner and its active chunk runner; they are not duplicated as generic weights.
Their actual aligned bytes enter the existing weight cap/device-weight ledger.
Each execution owns quantized activations, intermediates and component metadata,
charged to the remaining aggregate graph cap before scheduler admission. Both
components check remaining metadata budgets before allocation. Host accounting
includes the extra immutable metadata, conversion bound, per-execution owners
and readback. Combined ledgers count the immutable owner once and both scratch
sets. Existing serialized-backend, chunk lease and failure/poison rules apply.

Every owner reserves a fresh `PREFIX.layer-L.t-T.owner-N` directory atomically.
It stores `dense-scalar-effective.spv` and `quant-effective.spv`. Existing
receipts survive collisions and replanning. This is a pair of shader receipts
per owner; runtime activations and model data are not written there.

`PortableRoute::native_shared` is set only after component execution and finite
result validation. FFN `native_shared_calls/rows` publish only after its post
graph succeeds. The model stages actual provider-boundary tags, verifies the
requested mode and publishes `PortableModelWork` totals only after a complete
forward including any requested head. Reset retains completion totals; failed
chunks and poisoned forwards gain no completed-forward credit.

Only when enabled, stop/done/timing JSON includes `native_shared_method=true`,
`shared_numerical_parity=false` and `native_shared_calls_delta/rows_delta`.
The placement report includes `NATIVE_Q8_SHARED_METHOD`. For serial T1,
completion is 48 calls/rows per consumed token, including prompt and generated
tokens. A chunk credits one call and T rows per layer. Default counters stay
zero and default CLI fields stay absent.

Focused host tests use the actual FFN and native owner orchestration with mocked
kernel submission: T1/T2/T8, router/combine coexistence, live mixed input,
BF16 binding and unsupported types, direct shared-result consumption, failure
rollback (including post-graph nonfinite output), counter staging, budgets,
weight sharing, chunk leases, receipt collisions and owner cleanup. These
checks establish coordinator contracts, not new CUDA/Vulkan numerical evidence.
A real-model OFF/ON stdio smoke is still operator work; no new fleet run is
part of this integration. Stdio has no full-logit dump in that recipe.
