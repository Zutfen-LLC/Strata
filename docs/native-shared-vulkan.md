# Standalone native shared expert on Vulkan

This opt-in component observes the native shared-expert method before model
integration. It does not change the portable FFN or any model default. Build
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
legacy BF16 activation rounding, deferred gate application or model integration.

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
`qualified=false`. A physical pair is still required before model integration.
There is no throughput or broad-quality claim.
