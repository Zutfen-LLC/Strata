# Standalone native-method HC normalization and injection

This component is built only inside the existing opt-in Vulkan native strict-FMA
observation configuration. Run `native_hc_component` explicitly. It is not called
by a portable model, router, FFN, GDN or output head, and changes no default policy.
It covers normalization and injection only. Down/up, gates/mixed, pending writes
and model integration are outside this slice.

## Selected method and bounds

The reference is the actual Verifier's BF16 fused-GR read, not the older separate
`native_gr_norm` path. Geometry is H2560, HC4, epsilon 1e-6, and independent T1..8
rows. The component refuses other geometry, Q8/REQ8 representations, pending-write
fixtures and malformed spans. Both tools require the recorded native mode keys
STRATA_HC_Q8, STRATA_HC_Q8_INJECT, STRATA_HC_REQ8, STRATA_GR_V3 and STRATA_HC_SPLIT
to be unset/empty. A conflicting explicit configuration refuses; do not strip
flags or substitute a different mode to make a test pass. Injection weights are a raw BF16 `[4][10240]`
matrix packed into I32 storage; norm weights and residuals remain original F32.
The standalone normal-or-zero arithmetic domain excludes nonfinite/subnormal
inputs and host-control intermediates. Overflow/subnormal device observations
also refuse the call. It does not provide a general subnormal arithmetic mode.

`native_hc_q8_bf16` follows `iq_pack.py --compat-bf16`: dequantize each Q8_0
F16-scale/signed-code value to F32, then round BF16 nearest-even. It does not
requantize activations or round norm weights. Raw BF16 stays raw. Nonfinite or
conversion-overflow cases refuse. Host nearest-neighbor tests independently
check BF16 results, including signed zeros and even/odd ties; no private weights
are embedded in tests.

The shader uses one 256-thread workgroup per row for normalization. Each thread
visits the same four-value columns as `fused_gr.cu:253–291`, computes square
contributions, sums its own stream contributions, then performs explicit logical
32-lane XOR 16/8/4/2/1 reductions and an ordered eight-warp finish. It computes
`(R * norm_weight) * reciprocal_rms`, preserving that association. Injection uses
eight ordered strict FMAs per chunk, per-lane chunk accumulation and the same
logical warp XOR finish from `fused_gr.cu:59–83,162–166`. No arithmetic depends on
Vulkan hardware subgroup width. FP32 additions/multiplications/division carry
NoContraction; contracted operations use OpFmaKHR explicitly.

Normalization square lowering follows the compiled native sm86 norm: rounded
`y*y`, then strict FMAs for x, z and w in order. The first pair is significant;
rounding x first can change normalization bits. A synthetic adversarial host
regression and compiled-SPIR-V operand tracing guard that order. The reciprocal uses GLSL InverseSqrt. These are explicit, inspectable
choices requiring comparison with the compiled selected CUDA launcher on each
qualified configuration: source expressions alone do not establish cross-device
rsqrt bits. The inspected contraction does not qualify other compiler/architecture
lowerings.
There is no fitted constant, tolerance, device-specific correction or claim that
GLSL InverseSqrt is equivalent to CUDA rsqrtf. Reports retain `qualified=false`
and `cuda_rsqrt_equivalence_established=false`, including when an isolated case
matches. The shared observation helper leaves FP32 denorm mode unspecified;
host/domain checks cannot prove that a device never silently flushed a tiny
result. Subnormal-sensitive equivalence remains unsupported and unqualified.

Tensor views must be contiguous, word aligned, disjoint and on the same Vulkan
device. Descriptor biases are explicit. The bridge reuses existing pin-checked
component ownership, queried/logically enabled strict-FMA and integer-dot stack
gates, float-control checks, final effective-SPIR-V validation and synchronous
submission. It does not require buffer-device addresses. Device grid/shared
limits, metadata budget, all spans and status are checked. A bounded host domain
preflight reads the real input/parameters; it never supplies GPU output or acts
as a CPU fallback. Host temporaries stay below 1 MiB at T8. Tensor allocation in the
standalone tool is capped at 16 MiB; pipeline/driver allocations are separate.

## Independent reference and exact comparison

`native_hc_reference` is a CUDA-only test target. It calls the existing
`fused_gr_check` and `fused_gr_read_multi`, with recorded HC mode variables unset,
original residual/norm/BF16 injection and no pending write. A shared zero dummy
matrix satisfies unrelated down/up arguments; their mixed result is discarded.
It captures the launcher's actual reciprocal RMS, normalized vector and
injection without implementing a new numerical reference kernel. The selected
checked native variant is recorded. It does not force a different native mode.
CUDA compilation/execution must be performed where its toolchain is available.

The public fixture format carries fixed geometry/method tags, F32 input/norm and
raw BF16 injection. Both native and Vulkan tools preserve the exact fixture in
their fresh output directories. Vulkan checks nonzero-offset views/canaries,
immutable inputs and two repeated executions. It emits `rs.f32le`,
`normalized.f32le`, `inject.f32le` and its effective shader. The comparator checks
identical fixture bytes, actual native-launcher/Vulkan guard receipts and exact
raw words in dependency order: reciprocal RMS, normalization, injection.

```sh
native_hc_component --synthetic /fresh/synthetic.bin
native_hc_reference --device 0 --expect-device DEVICE --fixture /fresh/synthetic.bin --output /fresh/native
native_hc_component --device 0 --expect-device DEVICE --fixture /fresh/synthetic.bin --output /fresh/vulkan
python3 tools/compare_native_hc.py /fresh/native /fresh/vulkan /fresh/comparison.json
```

Generic `--from-q8 INPUT_F32 NORM_F32 INJECT_Q8 FRESH_FIXTURE` prepares native BF16
parameters using the checked conversion. It reads at most T8 residuals and the
fixed two roles; preparation initializes no Vulkan device. Private retained
operands belong only in private validation artifacts.

Exit 0 from the comparator means these captured component boundaries are exact;
exit 1 locates the first raw-word difference, and exit 2 rejects invalid evidence.
A reciprocal or normalized-input difference prohibits attribution locally to the
injection dot. An isolated injection match does not qualify normalization across
other inputs, other modes, mixed output, a full layer/model, quality or speed.

Host coverage includes independent conversion ties, constant-stream/basis-row
controls, T/stream indexing, zero/cancellation/adjacent-value/dynamic-range
fixtures, unsupported arithmetic refusals, actual dispatch validation with
mocked submission, strict arithmetic disassembly and exact pinned-helper
transformations/mutation refusals. None substitutes for the independent CUDA and
Vulkan comparison.

Numerical acceptance is now specified separately in
[native-hc-numerical-contract.md](native-hc-numerical-contract.md). The original
exact comparator remains diagnostic and unchanged. The numerical validator
bounds only division/rsqrt differences and requires exact local normalized/dot
operations from each backend's actual RS; it does not integrate the component
or turn component results into full-model acceptance.
