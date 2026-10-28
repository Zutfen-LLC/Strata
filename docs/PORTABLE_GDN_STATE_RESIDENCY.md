# GDN session state residency trial

`--gdn-resident-state` retains each GDN layer's recurrent and convolution
history on its selected backend between tokens. Configured PLE convolution
history shares that layer's state allocation. The option defaults off and is
independent of `--dense-resident-weights` and `--gdn-resident-weights`. The
resident adapter forwards it only when requested. Without a requested available
accelerator, the existing CPU path remains active.

The blueprint is Strata's native `src/core/session.cpp`: `session_bytes`,
`session_init` and `gdn_point_at` reserve and select one state slice per GDN
layer. The portable layer now owns a persistent backend input buffer through
its existing backend/context abstractions. Its existing scheduler owns separate
candidate outputs. No kernel arithmetic, precision or expert scheduling changes.

Every candidate state, history and attention residual is still read back and
checked for finite values before FFN. Trace reads and public host getters stay
active. Only after attention validation, FFN success and host allocation finish
do synchronous backend tensor copies commit candidates to the persistent inputs.
Host mirrors, token history, count and residual then commit together. Successful
steps no longer upload those host mirrors as the next token's state inputs.

Host candidate state/history vectors are pre-sized in the existing layer owner
and reused regardless of the device-state flag. Complete reads and finite
checks still run in their original order. Only a successful post-FFN commit
swaps those private banks with the public mirrors; they never alias. Reset
clears the public state while retaining private capacity. A rejected candidate
is fully overwritten and validated before any later successful commit.

The host ledger already prices committed and candidate copies, so its state
budget and the device ledger do not change. Retained host RSS can increase:
one candidate bank holds 3,268,608 bytes per GDN layer, plus 368,640 bytes for
configured PLE history. At 36 GDN layers and one PLE layer, candidate payload
capacity totals 118,038,528 bytes (about 112.57 MiB). This is a geometry-derived
capacity, not a measured RSS increase. Validate actual host headroom; no cap or
reserve is raised. Readback volume and synchronization remain unchanged.

Callback rejection, exceptions and nonfinite results before commit leave the
committed backend inputs and public mirrors intact. A recoverable exception
during device commit poisons the layer: further steps refuse until reset.
Reset clears and synchronizes the persistent buffer before clearing public state.
An unrecoverable backend error retains the backend's existing fatal-error
behavior; the copy API does not return a recoverable device status. Model-level
partial-token failures retain the existing session poison/reset rules.

Before graph scheduling, the layer checks the aligned persistent allocation
against `PortableGdnOptions::resident_state_budget` (4 MiB by default), then
checks its actual buffer size. Existing scheduler buffers already price the
candidate storage. `device_state` and startup `actual_state_bytes` report the
persistent allocation separately from weights and schedulers, and add it to
the same per-device admission total. The host bound also includes the additional
metadata context and the copy API's bounded host staging fallback. Model
preflight reserves the state budget before constructing each affected layer.
Caps, reserves and expert plans are unchanged; refusal stops admission.

The pinned Vulkan backend implements same-device buffer copies directly. An
explicitly permitted CPU fallback may require host/device copies; the trial
does not change fallback policy or promise that every placement avoids uploads.
The backend, full candidate readbacks, finite checks and host mirrors remain
part of the cost. The existing attention/exchange timer ends before FFN; the
post-FFN state commit is included in total GDN layer time, outside that inner
scope. Do not infer eliminated PCIe traffic or a speedup from the flag alone.

CPU tests cover selected-backend sequence/trace parity, failed candidates and
retry, reset replay, absent-backend behavior and aligned budget refusal. PLE
checks include token history, EOS and rollback. Admission tests cover persistent
state sums, per-device separation and transactional cap refusal. GDN and PLE
Vulkan fixtures exercise state OFF/ON with default, GDN-only and dense weight
placement, retaining their independent references and placement checks.
Compilation and CPU execution do not qualify Vulkan hardware or model parity.
