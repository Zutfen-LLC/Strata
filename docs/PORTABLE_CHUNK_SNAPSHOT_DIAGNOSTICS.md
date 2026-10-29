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
