# Locating a chunk snapshot mismatch

The component fixture keeps its existing `chunk committed snapshot differs`
assertion. Failure-only `[chunk-snapshot]` lines identify the mode, projection and
causal flags, width, replay, position, T and comparison phase. Named GDN fields
report differing-element counts, first differing index and bits, finite-pair
max_abs/relative_l2 and nonfinite counts. These metrics are descriptive; they do
not add a tolerance or change pass/fail. Signed-zero differences still fail.

The GDN Vulkan fixture exercises these modes in order, with a serial control on
the same selected Vulkan device:

| Order | Policy | Batched input projections | Causal graph |
| --- | --- | --- | --- |
| 1 | diagnostic | off | off |
| 2 | production | off | off |
| 3 | diagnostic | on | off |
| 4 | production | on | off |
| 5 | production | on | on |

A diagnostic GDN snapshot contains token count, last output, two integer token
history entries, recurrent state, convolution history and any PLE convolution
history. A production snapshot exposes only token count, last output and token
history; it does not inspect device recurrent/convolution state. Empty fields
are retained in the diagnostic layout. QSA keeps its existing opaque history
payload and comparisons; named state diagnostics here are specific to GDN.

| Phase | Exact fields checked | What it distinguishes |
| --- | --- | --- |
| `provider-publication` | tokens, history, last output against the pre-chunk snapshot | Premature publication during provider execution |
| `after-chunk-vs-serial` | tokens and history; last output excluded | Attention state/history equality with the same-device serial control |
| `reset-A-replay` | tokens, history, last output against the first A run | Reset/replay determinism at the same chunk width |
| `serial-continuation` | tokens, history, last output against serial | State continuation and exact final output equality after a T1 step |
| `failure-provider-publication` / `failed-chunk-publication` | tokens, history, last output against pre-chunk | Failed execution must not publish a completed result |

Output residuals and route activations/weights also have separate existing
numerical checks; route IDs remain exact. A passing numerical output check is
not permission to weaken snapshot equality. In particular, aggregate residual
metrics from an earlier completed mode say nothing about the field that stopped
a later mode. A last-output difference at continuation is distinct from a token
or inspected-state difference. Neither alone establishes the implementation's
cause; retain the failure and inspect the indicated operation boundary next.

## One-device diagnostic gate

Build `portable_gdn_vulkan_test` at the diagnostic commit using the retained
pinned dependency and known-working Release Vulkan configuration. Record source
and binary hashes. Verify device 0's identity, then run once inside the existing
finite resource/temperature/sensor guard with owned-process cleanup:

```sh
timeout 600 ./portable_gdn_vulkan_test --device 0 --expect-device DESCRIPTION
```

Retain stdout and stderr together. Stop on the first failure without retries,
cap/tolerance changes, subsequent device runs or model execution. Return the
`begin` markers, complete `[chunk-snapshot]` mismatch block, original assertion
and nearest preceding component summaries. A pass still requires every original
numerical, placement, route, state, submission and reset gate; this addition
cannot convert a failure to a pass.

Local `portable_chunk_snapshot_test` deliberately changes each GDN snapshot field,
a one-ULP output, signed zero, nonfinite output and output length. It verifies the
original rejection message and that `last=false` retains its original meaning.
It tests the diagnostic seam, not physical device numerical behavior.

## Pinned projection dispatch investigation

At GGML pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`, inspect
`ggml/src/ggml-vulkan/ggml-vulkan.cpp`:

- `ggml_vk_mul_mat` selects the vector implementation for contiguous Q8_0 weights
  and N=1..8 with unbatched higher dimensions. The maximum column count is eight
  in `ggml-vulkan-types.h`. These GDN matrices fit the storage-buffer range, so
  the large-matrix split exception does not apply under the standard limit.
- `ggml_vk_should_use_mmvq` returns true for N>1 before vendor-specific T1 rules.
  Modern NVIDIA Q8_0 T1 returns false (also false for K<=4096). AMD GCN's Q8_0 T1
  rule differs. This is a shape/capability policy, not a vendor-wide failure.
- `ggml_vk_mul_mat_vec_q_f16` additionally requires accelerated integer-dot
  capability, contiguous F32 activations and alignment before using MMVQ. With
  that path it quantizes activations to Q8_1. Otherwise it selects the
  F32-activation vector pipeline. The selector's name alone does not establish
  FP16 accumulation.
- The vector selector does not read accumulator or source precision annotations.
  `ggml_prec_set_acc` writes op parameter 0 and `ggml_prec_set_src` writes parameter
  3 for source 1; neither is consulted there. A F32 annotation alone therefore
  cannot prohibit this vector-path activation quantization at the pin.
- Cooperative-matrix/FP16-accumulator selection belongs to the matrix path, not
  this vector branch. The Q8_1 shader uses packed integer dot products and F32
  accumulation. `quantize_q8_1.comp` rounds activations after block scaling by
  max_abs/127; its scales are stored in the Q8_1 representation. This is not a
  source basis for attributing this branch to TF32.

This establishes a possible serial/batch numerical-contract change when the
capability and default-policy conditions hold. It does not identify the first
changed tensor in an actual device execution or prove that all divergence is
explained by that change. No exact snapshot gate is relaxed.

For a smaller diagnostic than the stateful fixture, build the same
`portable_gdn_vulkan_test` and use the existing guard for one invocation:

```sh
timeout 600 ./portable_gdn_vulkan_test --device 0 --expect-device DESCRIPTION --projection-probe
```

This mode runs only synthetic HC/input-projection graphs for the first two
fixture rows, sharing one immutable weight bank per control. It compares T2
against two T1 calls at norm, HC down/low/up, gated, mixed, injection, qkv, gate,
alpha and beta stages. Each stage is protected for readback. It reports the first
changed stage plus exact differences, finite-pair metrics and nonfinite counts.
The three controls are default layout/precision, F32 accumulator+source
annotations, and a T2-only strided F32 layout made with public pad/view operations.
At the pin the strided input excludes MMVQ and is packed back to F32 for the
F32-activation kernel; it is a diagnostic candidate, not a runtime correction.
Output protection can affect fusion/liveness, so this probe does not qualify the
original stateful graph or its performance. It sets no environment override.

Retain initialization/capability lines, any existing MMVQ environment override
receipt and all `[gdn-projection-probe]` lines. Exit zero means the diagnostic
completed, explicitly `qualification_evaluated=0`; it is not numerical acceptance.
If default first diverges at HC down while norm matches, F32 annotations leave
that divergence, and the strided control restores equality, that supports the
activation-quantization explanation. Any other result needs investigation at the
reported earlier stage. Stop after the probe; no retry, model run or broad switch.

Until a serial-equivalent batching choice is qualified, the smallest existing
fallback is to omit both `--gdn-batch-input-projections` and `--gdn-causal-chunk`
while retaining `--prompt-chunk-size`. This keeps serial GDN attention and the
other layer/FFN chunk improvements. It does not require a device blacklist or a
global backend override.
