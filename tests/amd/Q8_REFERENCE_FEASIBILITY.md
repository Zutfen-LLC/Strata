# Vulkan Q8 CPU-reference feasibility screen

Date: 2026-10-26. Strata base: `e7715a002aef16620609a1587cf86f47da3dddc0`.

The direct FP32 arithmetic port has a source-level FMA-contract concern on the
reported Mesa RADV 25.0.7 / gfx900 stack. Actual installed-driver behavior has
not been tested. Keep executor integration and model traces paused pending the
tiny direct discriminator below. This is **unconfirmed static analysis of the installed
driver's expected behavior**, supported by local compiler and CPU probes;
there was no GPU execution in this investigation.

## Required contract

[`gpu_cpu_reference.cu`](../../src/worker/gpu_cpu_reference.cu) uses signed-byte
transforms, saturating integer pairs, eight accumulators, sequential Q8 blocks,
an explicit round-to-nearest scale product, a single-rounding FP32 FMA for each
accumulator, and a fixed add tree. The native host quantizers and SwiGLU can
remain unchanged. The blocking operation is the accumulator FMA.

## Vulkan and driver findings

Vulkan's legacy `GLSL.std.450.Fma` does not require a fused operation.
`VK_KHR_shader_fma` / `OpFmaKHR` supplies the correctly rounded contract;
ordinary float controls alone do not strengthen legacy Fma to that contract.
See the [official extension description](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_fma.html)
and [Vulkan SPIR-V precision rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation).

The [reported host inventory](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793033992390069)
identifies RADV `25.0.7-2+deb13u1`, Vulkan 1.4.305, and Vega 10 V340 devices.
This is a prior Hermes measurement, not a fresh device query by this task.

The following source was read from the official Mesa `mesa-25.0.7` tag:

- [RADV shader options](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-25.0.7/src/amd/vulkan/radv_shader.c):
  `lower_ffma32` is enabled below GFX10.3, including gfx900, for compute as well
  as other stages. RADV runs the algebraic lowering pass.
- [NIR algebraic lowering](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-25.0.7/src/compiler/nir/nir_opt_algebraic.py):
  the `lower_ffma32` rule replaces FP32 FMA with multiply followed by add. It
  lacks the inexact-only marker; the file explicitly distinguishes mandatory
  backend lowering from transformations restricted by precision. Later FMA
  fusion matches inexact adds, so it cannot serve as a guarantee for an
  explicitly protected multiply/add sequence either.
- [RADV device capabilities](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-25.0.7/src/amd/vulkan/radv_physical_device.c):
  FP32 RTE is available, and FP32 denormal preservation is advertised when
  using ACO. The extension table has no `KHR_shader_fma` entry. This is an
  upstream source observation, not verification of Debian patches or the
  live device's advertised extensions.

Vega hardware FMA availability does not resolve that lowering policy. The
[ACO selector](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-25.0.7/src/amd/compiler/aco_instruction_selection.cpp)
can emit `v_fma_f32` for surviving NIR FMA operations, but that is downstream
of the RADV/NIR lowering above. Neither a successful GLSL compile nor seeing
Fma in SPIR-V proves that this hardware instruction survives.

## Local checks actually performed

The probe source, exact cases and AVX2 witness source are supplied in this
repository under `fma_screen/`. Recorded local checks:

- `glslc` 2026.4 compiled `precise float f = fma(a,b,c)` for Vulkan 1.2 at
  `-O0`; SPIR-V validation passed. Disassembly has legacy `OpExtInst Fma`
  without a `NoContraction` decoration. Explicit separate multiply/add have
  `NoContraction`. No shader was dispatched.
- The AVX2/FMA two-block Q8 witness produced fused bits `3b401000` and
  separate-operation bits `3b400000`. Both blocks use valid signed-byte
  dot inputs and binary16 scales. Block 0 contributes -1027 using scales
  1/1 and dot -1027; block 1 uses scales `1 + 2^-10` on both sides and dot
  1025. Other lanes are zero. This demonstrates that a second rounding step
  matters even within this Q8 arithmetic domain. It is not a real-expert
  parity measurement.

## Decision and remaining evidence

Pause the small direct native-FMA port at this feasibility gate pending a
direct witness on the installed driver. This is not yet a demonstrated
platform blocker. The contract concern is requiring CUDA `__fmaf_rn` semantics
through legacy Vulkan Fma on the reported RADV gfx900 lowering path. This is not a claim that exact GPU
arithmetic is mathematically impossible. GPU integer/software emulation or a
different qualified driver contract could be investigated separately; neither
was implemented, and no driver change is authorized here.

Run the tiny direct screen below first, recording the actual driver and
capabilities. Inspect final ISA only if the measured result needs explanation.
If a supported exact path is established, require byte-exact gate, up, hidden,
packed boundaries, and down/output parity on the one retained real expert
before any model or trace rerun. No fleet or Hermes dispatch was made.


## Cheapest installed-driver discriminator (proposed, not executed)

Reuse the already compiled module built from `fma_screen/fma_probe.comp`
(SHA256 `e8eaa3fda0cb484f5ed3d9b37d07b18c3af53254dee84f697a288a9fdbba92a2`).
Its source is supplied in this PR; the README shows how to retain disassembly.
It has local size 1 and one storage buffer, set 0 binding 0, with five FP32
words: `[a,b,c,legacy_fma_result,separate_result]`, offsets 0/4/8/12/16.
It requests neither explicit RTE nor FloatControls2. This first run tests the
unchanged module and installed defaults. It does not establish a controlled
rounding mode contract for a later projection shader.

Use one confirmed physical V340 die on 05, preferably the device reported at
PCI BDF `0000:07:00.0`; resolve its UUID from the wrapper's `--list` output,
then select it explicitly with `--device-uuid`. Do not assume enumeration
index 0. Check vendor/device, driver name/version,
device UUID and PCI properties. Refuse software implementations and ambiguous
device selection. Record the actual selected device and shader hash.

The standalone wrapper is now supplied in `fma_screen/fma_screen.cpp`; see
[`fma_screen/README.md`](fma_screen/README.md) for exact standalone commands.
It needs only the installed Vulkan headers/loader, one compute queue, one
20-byte host-visible coherent storage buffer, descriptor set, pipeline,
command buffer and fence. No ggml or Strata build.
For each of the four cases in `fma_screen/fma_cases.json`,
copy the three exact input uint32 bit patterns, initialize output words to
sentinels, dispatch `(1,1,1)`, issue compute-write to host-read visibility,
wait for the fence, and print all five words as eight-digit hexadecimal.
Wait before reusing the same buffer. The supplied wrapper refuses execution
if coherent memory is unavailable. Bound the whole helper externally, preserve
stdout/stderr, and stop on Vulkan failure.

| Case | a bits | b bits | c bits | Fused RNE | Separate RNE |
| --- | --- | --- | --- | --- | --- |
| Q8 positive | 3f804008 | 44802000 | c4806000 | 3b401000 | 3b400000 |
| Q8 negative | 3f804008 | c4802000 | 44806000 | bb401000 | bb400000 |
| Cancellation | 3f800001 | 3f7ffffe | bf800000 | a8800000 | 00000000 |
| Control | 3f800000 | 40000000 | 40400000 | 40a00000 | 40a00000 |

These expected values were computed locally with `libm.fmaf` and explicit
FP32 rounding after separate operations; the Q8 positive case was also
checked with actual AVX2/FMA intrinsics. The cancellation case is an extra
arithmetic discriminator, not a claimed Q8 block construction.

If legacy Fma returns fused bits for all three discriminators and the
separate path returns its expected bits, the installed stack passes this
small screen; do not treat that as real-expert or universal exactness.
If legacy Fma returns separate bits, the suspected failure is directly
observed. Any other result, unchanged sentinel, or failed control requires
investigating harness/device/rounding behavior before drawing conclusions.
No model loading, full build, installs, driver settings, or environment
compiler overrides are needed. Do not expand into an ISA campaign initially.

During the same harmless property query, record `shaderFloatControls2`,
`shaderRoundingModeRTEFloat32`, denormal/signed-zero properties, and whether
`VK_KHR_shader_fma` plus its `shaderFmaFloat32` feature are actually available.
Feature availability alone does not modify an unchanged shader.

FloatControls2 controls optimization permissions, including GLSL extended
instructions; it does not replace legacy Fma's permitted accuracy with a
correctly rounded FMA guarantee. RoundingModeRTE fixes rounding direction,
but separately rounded multiply/add still differ from a single-rounding FMA.
If a follow-up is necessary, a single small module variant with supported
explicit controls can test their actual effect without a shader campaign.
A supported `OpFmaKHR` path would materially change the contract conclusion.
See [FloatControls2 revision 1, promoted to Vulkan 1.4](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_float_controls2.html),
[SPV_KHR_float_controls2 revision 10, 2024-04-03](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_float_controls2.html),
and the [Vulkan individual-operation and rounding definitions](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation).
