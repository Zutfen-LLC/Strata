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

## Candidate readback diagnostics

The existing `--expert-timings` option adds these cumulative host-call scopes to
`PortableGdnWork`, emitted as per-request deltas by the resident stdio CLI:

| Fields (each has `_ms_delta` and `_calls_delta`) | Scope |
| --- | --- |
| `gdn_candidate_readback` | Existing tensor-get calls for candidate recurrent state, convolution history, attention residual, and configured PLE history |
| `gdn_candidate_finite` | The existing finite scan immediately after each of those reads |
| `gdn_state_commit` | Existing post-FFN device candidate copies and synchronization; one scope per resident GDN layer |

Readback and finite scopes are disjoint and nested in attention/exchange.
Trace-only reads, tensor-size checks and host buffer allocation are excluded.
The commit scope is outside attention/exchange and FFN, inside total GDN layer
time. It does not include the later host mirror swaps. A readback interval may
include an existing backend wait; it is not a GPU event or pure PCIe bandwidth
measurement. No synchronization, transfer, scan or numerical operation is added
or removed. With timing disabled, no new clock reads or counter increments occur.

For each complete consumed model row, expect 109 candidate reads and 109 scans:
three per each of 36 GDN layers, plus one configured layer-1 PLE history read.
With state residency active on all GDN backends, expect 36 commit scopes; without
resident state expect zero commit scopes and zero commit time. Failed steps can
retain completed or attempted scopes and must not be interpreted as complete
rows. Trace capture does not increase these candidate counters. Reset preserves
caller-owned cumulative totals.

The resident validation adapter accepts older binaries that omit the entire new
diagnostic group. When present, all six fields must be valid and complete. It
checks candidate counts, readback-plus-scan containment in attention surrounding
time, and commit containment in the outer GDN residual. Enclosing and nested
timers must never be added together.

### Bounded comparison recipe

Build the reviewed diagnostic source using the retained dependencies and build
settings. Before model execution, run the existing GDN, PLE, model, stdio and
timing tests, plus the existing GDN/PLE Vulkan fixtures on each selected physical
device. This diagnostic requires a rebuilt binary; it does not change placements
or model data. The exact binary SHA256 is obtained after that build.

Use the existing frozen case-1 input and old 480-expert plans. The adapter command
below sends four identical GENs in one resident child, excludes the first from
warm timing means, then sends QUIT. All paths and hashes are retained operator
inputs; do not copy private inputs or output artifacts into a public PR.

```sh
python -B tools/validation-harness/portable_resident_v1.py \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA256" \
  --tokens-file "$CASE1_TOKENS" --max-new 32 \
  --plan "$PLAN0" --plan "$PLAN1" --expert-tensor-mib 2404,2404 \
  --dense-resident-weights --gdn-resident-state --expert-timings \
  --model-root "$MODEL_ROOT" --manifest "$MANIFEST" \
  --devices 0,1 --split 32 --device-cap-mib 7680 --out "$NEW_OUTPUT_ROOT"
```

Keep context 256, threads 4, CPU workers 12, host cap 32768 MiB, workspace
128 MiB and free reserve 512 MiB per die (the adapter fixes these). Keep the
qualified temperature/sensor stops, finite 1800-second clock guard, outer bound
1000 seconds, and adapter load/request/child bounds 180/180/960 seconds, with
verified owned cleanup. Do not add GENs, retries, warmups or resize plans/caps.

Compare every recorded row and stop against the retained same-backend baseline;
retain individual requests and report warm means for the three new scopes and
the existing enclosing phases. Confirm 109 read/scan and 36 commit calls per
consumed row. A timing change alone is not an output or numerical-parity result.
