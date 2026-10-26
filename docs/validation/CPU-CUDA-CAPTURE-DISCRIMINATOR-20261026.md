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
