# Q8 CPU-equivalent CUDA diagnostic control

The supplied first-boundary replay passed. Historical/raw-input identity and A0
reconstruction matched; replacing only 13 rows reproduced all 5,120 recorded B
union values exactly. Identical-row reducer controls passed. This establishes
expert-row arithmetic as the source of the initial discrepancy, rather than the
old outer reduction-order bug. It does not establish which individual arithmetic
instruction accounts for every later model difference.

## Operations and scope

Source inspection of the qualified Q8 AVX2/non-VNNI baseline shows:

| Operation | Native CPU | Current CUDA Q8 path |
| --- | --- | --- |
| Integer partial | Four codes per accumulator lane, signed-byte transform and saturating pair sums | Eight codes per partial using two dp4a calls |
| Float accumulation | Eight lanes, sequential 32-value blocks, explicit FMA per block | Warp-strided partials with a different float addition order |
| Final reduction | Fixed eight-lane horizontal-add tree | 32-lane XOR butterfly |
| SwiGLU | CPU `std::exp(float)`, division and multiplication | CUDA `__expf`, division and multiplication in the CUDA compilation environment |
| Hidden packing | Native CPU Q8_0 quantizer | CUDA Q8_1 quantizer |

The exact CPU/CUDA instruction sequences after compiler contraction have not been
audited in assembly. Nevertheless, the grouping and accumulation trees differ in
source. Existing controls established equal input scale/code fields, so changing
`roundf` alone cannot explain or repair the observed first-boundary discrepancy.
Nonlinear and hidden-quantization differences remain possible even after dot
products match. Earlier stage controls found gate/up differences and, on another
retained boundary, a hidden code difference.

`STRATA_CUDA_Q8_CPU_REFERENCE=1` selects a **slow hybrid diagnostic** only in the
network CUDA executor, restricted to plain Q8_0/Q8_0, H2560/FF640. It is not a
CUDA-only implementation or a production qualification:

- Gate, up and down matrix products still execute on CUDA with GPU-resident expert
  weights. The new kernel follows the CPU's four-code groups, saturating pair
  operations, eight accumulators, sequential block FMAs and horizontal tree.
  Explicit round-to-nearest intrinsics and separate precise compilation prevent
  fast-math/FTZ/contraction from changing that specified sequence.
- The actual raw input is packed with the native CPU quantizer. Gate/up results
  return to the GPU worker's host for the same CPU SwiGLU expression and native
  hidden quantization, then only the packed hidden rows return to CUDA for down.
- Expert weights are not retained on the host or recomputed on the CPU. Existing
  VRAM workspace is reused. Additional host scratch is bounded by 80 entries and
  is under 0.5 MiB. Synchronization and host/device transfers may make this slow;
  no throughput claim is made.
- Without the option, the original quantizer/grouped-kernel call sequence remains
  in place. Unsupported reference geometries are refused. Grouped-kernel metadata
  snapshots are refused in this different diagnostic path rather than mislabeled.

Host math-library/quantizer and CPU ISA/build identity are part of this reference
contract. This control targets the qualified baseline, not universal bit equality
across CPU libraries, GPU architectures, quantizations or backends. No AMD code
or global precision/default change is included.

## Finite offline gate on 03

The parent coordinates execution independently of the Q6 benchmark. Use existing
source/build settings, the same accepted first-boundary packet and staged 13 blobs.
Preserve old binaries and outputs. Build only the changed CUDA diagnostic targets
and the two small synthetic checks:

```sh
cmake --build "$CUDA_BUILD" --target gpu_capture_replay \
  q8_cpu_reference_host_test q8_cpu_reference_cuda_test -j 2
ctest --test-dir "$CUDA_BUILD" \
  -R '^q8_cpu_reference_(host|cuda)_test$' --output-on-failure
```

The CUDA CTest explicitly sets the diagnostic option and tests repeated expert
entries with two tokens against native CPU rows. A skip is not a pass.
Then run the actual first-boundary control once, on the same qualified CUDA device
and with the original B plans in order 01/03a/03b/04:

```sh
umask 077
env STRATA_CUDA_Q8_CPU_REFERENCE=1 STRATA_KQ_SPLIT=0 \
  timeout --signal=TERM --kill-after=5s 180s \
  "$CUDA_BUILD/gpu_capture_replay" --boundary \
  "$HISTORICAL_CAPTURE" "$FIRST_BOUNDARY_PACKET" "$STAGED_BLOBS" "$NEW_REFERENCE_OUTPUT" \
  "${B_PLANS[@]}" >"$NEW_REFERENCE_LOG" 2>&1
```

All earlier raw-identity, accepted SRA1, A0 reconstruction and identical-row reducer
gates remain mandatory. In this explicitly labeled `CPU_REFERENCE_CONTROL` mode,
**every one of the 13 CUDA rows must match the retained CPU row bit for bit**.
Their union must match recorded A0 bit for bit. The tool separately reports
`recorded_B_bits`; inequality with the old divergent B is expected in a successful
CPU-equivalent control. `target_bits=1 cpu_reference=1` is the explicit new
*diagnostic target*, not a change to B's original frozen gate. With the option
absent, the existing B-reconstruction target is unchanged.

On any row/control/union mismatch, stop and report the first entry/channel/stage.
Do not start a model trace, change tolerances or silently switch to a CPU-only
expert executor. On success, return the result to the parent; **only a separate
parent-dispatched first-request model trace** can determine whether this removes
amplification. No resident, model GEN, tunnel, service or fleet operation is
performed by these offline commands.

## Recommendation and quality contract

First use this control to answer the arithmetic question cheaply. If the offline
rows match, run the separately authorized model trace and retain the current
strict logit/KL gate. If that passes too, decide whether the slower hybrid path is
useful as an explicit capacity/correctness mode before investing in optimization.
A later all-CUDA CPU-equivalent SwiGLU would need a pinned exp/division/quantization
implementation and its own exact tests; ordinary CUDA `expf` is not assumed to
match the CPU math library. That is a larger next step, not hidden in this patch.

A broader quality contract could eventually allow small backend numeric variation
when task quality and distributional checks support it, but that is a separate,
explicit user decision. Matching a fixed CPU operation graph is achievable for a
qualified configuration; universal fast cross-backend bit identity is a much wider
requirement. This patch neither relaxes the user's current gate nor substitutes
one passing prompt for quality acceptance. Unified memory on AMD would change
placement/data movement; it alone would not make mixed CPU/GPU arithmetic equal.

## Validation actually performed

Local Release-optimized host test passed: the proposed eight-lane gate/up/down dot
transcription matched the linked ggml CPU dot bit for bit, and host SwiGLU/hidden
packing matched native CPU execution. Shape refusals passed. The executor and
CUDA-test host translation units passed CUDA-header syntax checks. The existing
first-boundary host check and exporter tests remain passing.

Not run locally: the new CUDA kernel compilation, CUDA synthetic test, real
13-row CPU-reference control, default-path hardware regression or model trace.
There is no NVCC/device here. A clang CUDA parse attempt stopped at the incomplete
local SDK's missing `nv/target`; no kernel compile success is asserted. A missing
`<cmath>` in the host test was corrected and the test rerun successfully. Keep
this PR a draft until the targeted CUDA build and tests execute.
