# Offline CPU/CUDA correctness discriminator

Issue #18 now prioritizes correctness and model capacity. A slower distribution can
still be useful if the model cannot otherwise fit. The prior live performance result
and failed frozen numerical gates remain unchanged. No tolerance is relaxed here.

This diagnostic reuses the retained CPU capture, trusted B assignment plans, one
verified expert blob, native CPU functions, native CUDA functions and the existing
GPU executor. It does not load a model or contact a resident or fleet worker.

## Inputs and bounds

- The existing validated Q8_0 capture: at most 96 MiB; sampled records have T <= 2,
  K <= 10, H = 2560 and FF = 640.
- The original B GPU assignment plans. Supply all four plans to find the earliest
  active captured entry in their union. The output records which plan matched.
- Exactly one selected expert blob, 5,222,400 bytes, checked against that plan's
  content digest. Copy this file from the retained cache; do not open that cache
  through GpuCache or copy full model shards.
- A new scratch directory. Existing paths are refused. The diagnostic retains a
  private one-expert cache there. It does not evict or delete existing assets.

The selection receipt includes record, original entry, layer, token, expert,
original plan placement digest, asset digest and exact retained cache basename.
Plan validation checks the model and geometry against the capture. The trusted B
provenance of the supplied plans must be checked by the operator; filenames alone
do not prove that provenance. Record the capture and plan SHA-256 checksums.

## Build and run on the qualified CUDA host

Update the existing checkout to this diagnostic commit without overwriting other
work. Reuse its existing CUDA/ggml configuration and build directory. No dependency
installation or fresh model setup is needed.

```sh
build=/home/hermes/strata-cuda-rows-01539e99-20261026T035048Z/build
diag=/home/hermes/strata-cpu-cuda-offline-diagnostic
cmake --build "$build" --target gpu_capture_replay_check gpu_capture_replay -j 2
ctest --test-dir "$build" -R '^gpu_capture_replay_check$' --output-on-failure
"$build/gpu_capture_replay" --select "$diag/cpu-capture.bin" "$diag"/B-*.plan
```

Before the last command, stage only the retained capture and original B plans in
`diag`. After selection, stage the printed cache basename from the matching B
worker's retained cache as `diag/selected.blob`. Do not claim it belonged to B
until the original plan and selection receipt have been checked. Then run once:

```sh
"$build/gpu_capture_replay" --run "$diag/cpu-capture.bin" \
  "$diag/selected.blob" "$diag/run-1" "$diag"/B-*.plan
```

The selected expert executes once through each of the direct CUDA and real GPU
executor paths, with the same captured F32 inputs and verified weights. All active
occurrences of that expert in the record retain their original token association.
Device 0 follows the operator's existing CUDA_VISIBLE_DEVICES selection. The GPU
executor requires a 256 MiB runtime reserve and allocates only its bounded normal
workspace plus this one expert. The direct CUDA allocations are released before
the executor allocates its workspace.

## Interpretation

`PACKED input_token_*` compares native CPU Q8_0 scale/codes with actual device
Q8_1 scale/codes, ignoring Q8_1's unused sum for the Q8_0 dot. No host transcription
stands in for CUDA. `CPU_vs_direct_CUDA_unweighted` compares before routing weights
or reduction. If it differs, the earlier outer worker-sum regrouping defect cannot
be the sole source of that row difference.

Only on a row mismatch, the tool reports gate/up and hidden packed differences.
It also feeds the actual CUDA input scale/codes into CPU gate/up dots. Remaining
gate/up differences in this control exclude input quantization as their cause.
The fused kernel may not materialize hidden F32, so the tool does not read its
unwritten scratch. Hidden packed differences alone do not distinguish SwiGLU
evaluation from hidden quantization.

`direct_CUDA_vs_GPU_executor_unweighted` must be bitwise equal, including original
entry indices. A mismatch points to executor metadata/scheduling or another
integration difference; it is separate from CPU/CUDA parity. This tests the GPU
executor directly, not remote host architecture or a fleet transport.

Finally a temporary loopback fixture returns frozen rows through the real GPU
client/group and invokes the actual mixed reducer. Selected GPU rows are real;
other entries use surrogate rows from the same expert's native CPU evaluation.
Both sides receive the identical dense matrix. `fixed_rows_mixed_vs_unsplit` must
be bitwise equal. This isolates assembly/reduction and does not replay B's full
routing or reproduce captured full partials/logits. The fixture loads no GPU or
resident CPU arena. It opens only a temporary loopback listener.

Exit 0 means executor rows and fixed-row reduction matched; it does **not** mean
CPU/CUDA or model parity. CPU/CUDA discrepancies are printed even on exit 0.
Exit 1 means direct CUDA versus executor row mismatch. Exit 2 means a refused
input, failed invariant, CUDA error or mixed-reducer mismatch. Preserve stdout,
stderr and the exit status. Do not turn these diagnostic metrics into new model
acceptance tolerances.

## Local validation

Passed: CPU-only Release build of `gpu_capture_replay_check` using existing local
ggml sources. Its synthetic loopback CTest passed. It checks selection, one-asset
plan decoding, blob digest refusal, repeated-expert T2 indices, overlap refusal
and the real mixed reducer with cancellation-sensitive fixed rows.
The final CTest passed 1/1 in 0.27 s; mixed versus unsplit reported bitwise equality,
maximum absolute difference 0 and relative L2 0.

An initial fixture omitted the required READY GPU weight-accounting counters;
that fixture was corrected before the passing run. Sandboxed loopback socket
creation was unavailable; the permitted loopback test ran outside that sandbox.
The successful fixture's shutdown also printed `failures=1`. The service increments
this counter on receive failure, including connection closure. This does not
establish the reason for the separate fleet counters or their relation to logits.

Not run locally: CUDA compilation or execution, retained real capture selection,
and real CPU/CUDA parity. No CUDA toolkit is installed on this machine. The PR
remains draft pending the targeted CUDA host build/replay. No product arithmetic,
default, tolerance, resident, network access policy or model support is changed.

## Supplied first real-entry result at e275891d

The parent supplied a completed Hermes replay on 03, not a private-artifact audit
by this coding task. Record 0, entry 2, layer 0, token 0, expert 77 intersected the
actual B peer 03b plan. Its 5,222,400-byte blob passed the assignment digest check.
The actual replay took 0.615 s; the synthetic CTest also passed.

Both input tokens had zero scale/code differences across 80 blocks each. The
selected entry's hidden activation had zero differences across 20 blocks. The
CPU/direct-CUDA row difference was max absolute 1.49011612e-08, relative L2
1.05761476e-07. Gate/up max absolute differences were 2.98023224e-07 and
2.38418579e-07, relative L2 1.87733338e-07 and 2.02152427e-07. Feeding actual CUDA
input scale/codes to CPU dots left those gate/up discrepancies unchanged.
Direct CUDA/executor rows and the fixed-row mixed/unsplit reduction were exact.
The latter still used surrogate rows for other experts.

Thus the crafted rounding case was not triggered in this real entry. Tiny
backend arithmetic differences exist before reduction, but this observation does
not explain the live 0.0754367 logit relative L2 failure. It does not establish
parity for other experts or routing. No model, resident or access change occurred
in the supplied replay.

## Bounded survey of the retained real samples

The same executable now supports `--inventory` and `--survey`. Inventory reuses
`capture_row_samples`: first/last records for T1/T2 at layers 0, 12, 24 and 47,
up to the existing 16 samples and 225 active entries. It does not replay every
record in the larger capture. It validates disjoint B plans, reports every
original entry/peer association and deduplicates blobs by layer/expert.

Stage the original four B plans and capture as before. Use a flat read-only
`diag/blobs` directory containing the required cache basenames. Existing local
cache files can be referenced there; the tool only reads these staged inputs,
never opens them through GpuCache or changes their original caches.

First build just the CPU-capable inventory target and inspect the exact counts:

```sh
cmake --build "$build" --target gpu_capture_replay_check -j 2
"$build/gpu_capture_replay_check" --inventory "$diag/cpu-capture.bin" \
  "$diag/blobs" "$diag"/B-*.plan
```

Inventory exit 3 means missing files. Its `NEEDED plan=N cache_key=... bytes=...`
lines are the exact staging list; `plan=N` is the zero-based supplied plan order.
Stage only these verified selected assets from their corresponding B caches,
then repeat inventory to require zero missing files. Do not transfer full model
shards or all B assets. Inventory checks file type and size; replay verifies each
asset's digest before loading it.

Then build and run the actual CUDA survey once:

```sh
cmake --build "$build" --target gpu_capture_replay -j 2
"$build/gpu_capture_replay" --survey "$diag/cpu-capture.bin" \
  "$diag/blobs" "$diag/survey-1" "$diag"/B-*.plan
```

Survey refuses missing files before CUDA work. It loads one unique expert at a
time and reuses that expert across its sampled records. Direct CUDA allocations
are released before the executor prepares that same expert. The private scratch
cache rotates only files the diagnostic itself created and retains at most one
5,222,400-byte blob. Staged blobs are preserved.

Hard bounds are 225 entries/unique blobs and 300 seconds, checked between bounded
expert operations. The absolute staging upper bound is 1,175,040,000 bytes;
inventory provides the actual smaller count. The two retained row matrices take
at most 6,553,600 bytes. The existing capture reader is capped at 96 MiB; its
transient input/decoded copies, metadata and one-expert buffers keep this a small
diagnostic rather than a resident arena. The supplied one-entry 0.615 s elapsed
time is not a calibrated survey prediction. Even multiplying it by the 225-entry
upper bound gives about 138 s; the 300 s stop is a conservative guard, not a
throughput claim. The target host is the verified i3-10100F/16 GB PC with two
RTX 3060 cards, not a large-RAM server.

Survey emits per-expert/record contexts and stage summaries for input quantization,
gate/up, hidden quantization, down rows and direct CUDA/executor parity. A down
control feeds actual CUDA hidden scale/codes to CPU down dots. Summary fields
identify earliest differences and maximum absolute/relative discrepancies;
quantizer observation counts may repeat the same token input across experts.
Fused kernels may leave gate/up scratch unavailable; those stages are explicitly
reported as unavailable rather than interpreted as real values. Hidden F32 is
still not assumed to be materialized.

For each record, only its real B-intersection rows participate. The tool compares
native CPU and CUDA subset sums with the captured routing weights, then requires
exact assembly/reduction for all-CUDA and fixed CPU/CUDA row mixtures. All rows
in these controls are real computations; no surrogate rows are used. The mixed
control assigns experts by fixed layer/expert parity solely to exercise both
sources; it does not claim this is B's original CPU placement. Non-B entries are
excluded and counted. Captured full partials and model logits are not reproduced.

The survey opens no listeners or worker connections. Its reducer is the same
internal pure function called by GpuWorkerGroup: the old assembly/check/reduction
body was extracted without changing its arithmetic. Existing group fixtures
continue to test the socket/session wrapper. Local CPU-only Release builds and
both focused CTests passed (2/2 in 8.38 s) after this extraction. Expanded actual
CUDA compilation/execution remains pending on 03; no numerical fix is proposed.

## Two-record grouped-batch control and CPU comparison

`--grouped` is fixed to original records 0 and 828 (layers 0 and 12). It uses
their real B-peer expert sets and compares the production executor's grouped
outputs to direct standalone CUDA rows, restored to the original entry indices.
For each nonempty peer batch it runs full batch, one real expert, then full batch
again on the same prepared executor and scratch. It stops on the first mismatch
or after those two records. There are at most eight prepared batches and 24
executor calls, with a 60-second guard between bounded operations.

The batch control must hold its small simultaneous expert set: at most 20 experts
(104,448,000 bytes), rather than the survey's single expert. Host staging still
reads one 5,222,400-byte blob at a time. No complete B plan, full model or resident
arena is loaded. Its private 128 MiB cache rotates only owned scratch assets.
It opens no listeners or worker connections.

An opt-in executor method copies the actual submitted expert/asset-slot, group
start, destination, token and original-entry arrays to the diagnostic. Ordinary
inference does not request a snapshot. No kernel arithmetic or row weighting is
changed. The log includes the original B plan digest/slots, subset resident slots,
actual restored indices, device capability, runtime/driver/compiler versions and
relevant environment settings. A mismatch prints the original record/entry/token,
channel, phase and both F32 values and raw bits. This tests bounded subset plans
on 03; it does not claim to exercise all slots in the original large fleet plans.

The parent's identity audit confirmed the live A0/B engine and CPU worker hashes,
request/model assets and first-logit alignment. Live network workers had no
STRATA overrides, so KQ splitting was off. The prior survey receipt did not retain
its entire inherited STRATA environment. Use the clean environment below for this
control, explicitly setting KQ splitting off; do not expand the flag matrix.

After updating the qualified 03 source to this commit, reuse the existing build:

```sh
cmake --build "$build" --target gpu_capture_replay gpu_capture_replay_check -j 2
sha256sum "$build/gpu_capture_replay" "$diag/cpu-capture.bin" "$diag"/B-*.plan
env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  CUDA_VISIBLE_DEVICES=0 STRATA_KQ_SPLIT=0 \
  "$build/gpu_capture_replay" --grouped "$diag/cpu-capture.bin" \
  "$diag/blobs" "$diag/grouped-1" "$diag"/B-*.plan
```

Exit 0 means all requested comparisons matched exactly. Exit 1 is a row-bit
mismatch; exit 2 is an input, CUDA or metadata invariant failure. Preserve stdout,
stderr, exit status, the checked source pin and printed binary/asset hashes.
Do not automatically run another request or model trace.

`--cpu-control` executes only native CPU functions for record 0/entry 2/expert 77
and record 828/entry 13/expert 390. It verifies each blob and exports six files per
entry: native input bytes, gate/up F32, hidden F32, native hidden bytes and down
F32. Both entries total exactly 42,640 bytes. Output identities and GGML CPU feature
flags are printed. It does not use CUDA, a socket or a resident.

Run this on 03 with the CPU target, then on 02 using its existing qualified CPU
worker build configuration and GGML source/options. Do not copy the 03 executable
to 02 as a substitute for checking the 02 build. The supplied audit reports both
as GNU 14.2 Release, GGML 3cf03257f, native AVX2/FMA/F16C and no AVX512/VNNI;
different library hashes alone do not establish different arithmetic.

```sh
env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  "$build/gpu_capture_replay_check" --cpu-control "$diag/cpu-capture.bin" \
  "$diag/blobs" "$diag/cpu-control-03" "$diag"/B-*.plan
```

For 02, substitute its qualified worker build, retained capture path, staged two
verified blobs and original B plans, and a new output directory. Only those two
blobs (10,444,800 bytes) are needed. Compare the twelve identically named files
with `diff -rq` after staging the small outputs together. The first differing
stage separates quantization, gate/up, hidden processing or down arithmetic.

Local validation: CPU-only Release target builds and both focused CTests passed
(2/2 in 8.32 s), including metadata mapping/refusal, multi-expert plan encoding and
the 21,320-byte one-entry CPU export bound. Actual CUDA compilation and grouped
execution remain pending on 03; the PR stays draft.
