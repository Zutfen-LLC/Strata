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

## Supplied CUDA result and the single model-trace delta

The parent supplied PASS at `9e7cf13e`: configure/build succeeded; host and actual
CUDA CTests both ran and passed without skips. One real first-boundary execution
matched all 13 retained CPU rows and all 5,120 A0 union values exactly, with all
identity/reducer controls passing. `recorded_B_bits=0` is expected for the old B;
`target_bits=1 cpu_reference=1` establishes this control's intended match. No CPU
expert recomputation or rerun was reported. This is supplied hardware evidence,
not independent local execution. It does not yet establish model-trace parity.

The next run is **B-reference only**, comparing with the already verified A0 trace.
No new A0 request, warmup, complete answer or throughput campaign is needed.
Per-action temporary-access approval is a separate prerequisite; this document
does not create or authorize access. Parent dispatches after that approval.

The option is read by `CudaExecutor::prepare` in the remote worker. It changes no
wire capability, cache key, plan identity or coordinator API. Therefore retain the
verified `8f3920d6` trace engine and existing `7a6aa106` harness helpers; no engine
rebuild is necessary. Build only the worker from the qualified `9e7cf13e` source
and existing CUDA configuration:

```sh
cmake --build "$CUDA_BUILD" --target strata-cuda-expert-worker -j 2
sha256sum "$CUDA_BUILD/strata-cuda-expert-worker"
readelf -d "$CUDA_BUILD/strata-cuda-expert-worker"
```

Preserve prior ELFs. Stage the new worker under a new name on the three target
hosts and use it for all four configured endpoints (01, 03a, 03b, 04). Preserve
its hash, original device selection, port, plan, cache root/budget, reserve,
setup/deadline settings and all existing backend/environment settings. Add only
this environment binding to **each remote GPU worker's launch environment**:

```python
gpu_worker_argv[0] = new_staged_worker_elf
gpu_worker_env = dict(original_gpu_worker_env, STRATA_CUDA_Q8_CPU_REFERENCE='1')
```

This belongs in the retained owned remote GPU launcher, not merely coordinator
`cfg['env']` or the local CPU resident's environment. It is an environment option,
not a new worker command-line flag. Do not start a worker outside the existing
owned lifecycle.

### Native CPU and runtime dependencies

The new worker does require native CPU quantization and host math code. The
build links it through `strata_gpu_compute -> strata_gpu_protocol ->
strata_gpu_assets -> strata_expert_worker -> strata_kernels_cpu -> ggml-cpu/base`.
These project libraries are STATIC, and native CMake forces `BUILD_SHARED_LIBS=OFF`,
`GGML_STATIC=ON`, `GGML_BACKEND_DL=OFF`, and `GGML_OPENMP=OFF`. No separately deployed
`libggml` or Strata CPU `.so`, model files, CPU expert arena or CPU worker service
is needed for this local activation work.

This does **not** make the executable fully static. Preserve actual CUDA/runtime
library bindings and check `readelf`/`ldd` against each staged ELF before its owned
launch; there must be no missing libraries or GLIBC/GLIBCXX/CXXABI version errors.
The source links `CUDA::cudart`; actual runtime linkage is determined by the build,
not inferred from its filename. Keep the existing `LD_LIBRARY_PATH` where needed.
Use retained compatibility receipts to confirm AVX2/FMA/F16C host support and that
the CUDA build includes compatible device code/PTX for each target GPU. The
executed 03 test qualifies its tested configuration, not every target by inference.
Do not install libraries, change build ISA/precision flags or broaden validation
when a required compatibility binding is missing; report that specific blocker.

CPU `std::exp`/quantization still execute on each remote host, so its math runtime
is part of the arithmetic contract. Loader/ISA compatibility alone does not prove
bitwise equivalence across hosts. The model trace below is the exact test of this
four-peer configuration; retain the first differing stage if it fails.

### B-reference config and acceptance

Copy the retained successful B trace config into a **fresh** run root. Keep the
same current complete CPU resident and its saved identity, unchanged CPU worker
binary/manifest, both local GPU placements, all four B plans/caches/endpoints and
policies. There should be no CPU weight loading or network GPU weight transfer.
Change only the external GPU worker ELF/environment above and these output labels:

```python
cfg['run_root'] = new_absolute_run_root
arm['name'] = 'Bref'
arm['restart'] = False
arm['requests'] = [0]
arm.pop('reference', None)
cfg['assets']['A0_logits'] = retained_A0_first_logits_absolute_path
arm['first_logits_reference'] = 'A0_logits'  # Existing harness exact-byte gate; never old B.
cfg['arms'] = [arm]
cfg['capture_first_logits'] = True
cfg['env'].pop('STRATA_DUMP_FIRST_LOGITS', None)
cfg['env']['STRATA_FIRST_REQUEST_TRACE'] = str(Path(cfg['run_root']) / 'Bref.trace')
```

Keep the original GEN256 line with all 38 IDs, prefill64/short-read64, spec2,
context1024, local placement and original finite setup/RPC/request limits. Do not
preemptively enlarge the trace's 64 MiB bound or loosen a deadline. Before GEN,
require all four worker logs to contain this actual reference-prepare marker, in
addition to their normal successful READY/cache/placement receipts:

```text
strata CUDA diagnostic: Q8 CPU AVX2 dot order + host SwiGLU/quantization; slow reference, not production qualification
```

The generic worker-listening line still says `CUDA/q8_1 approximate`; it is not
proof of the selected arithmetic mode. Existing A0/B entry-row readiness markers
stay unchanged. Use the normal bundle/run commands once, with no new runner:

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$BREF_CONFIG"
python3 -B "$BREF_RUN_ROOT/run.py" --run-root "$BREF_RUN_ROOT"
```

Require one `DONE 0 ... trace`, no ERR, one sealed trace and `.0` first-logit row,
then normal QUIT/reaping and the same owned-worker/access cleanup. The existing
harness compares the first-logit file exactly to the staged A0 asset after request
completion/reaping; no new acceptance path is needed. On mismatch, preserve the
sealed trace/failure receipt for the comparison below and stop further runs.
Preserve the current CPU resident and report its actual identity rather than an
old PID.

Compare with retained A0 using the existing trace reader:

```sh
"$PYTHON_WITH_NUMPY" "$SRC/tools/compare_first_request_trace.py" "$A0_TRACE" "$BREF_RUN_ROOT/Bref.trace" \
  --a0-logits "$A0_LOGITS" --b-logits "$BREF_RUN_ROOT/Bref-first-logits.0" \
  --output "$NEW_COMPARISON_JSON"
```

Desired reference result: no retained numerical/routing differences and a
bit-identical A0 first-logit row. Ownership differs by design. Also evaluate the
existing **unchanged** fixed numerical gate. Its CLI expects three requests;
for this one pair invoke its existing functions directly, without duplicating
one observation into three artificial requests:

```sh
"$PYTHON_WITH_NUMPY" - "$SRC" "$A0_LOGITS" "$BREF_RUN_ROOT/Bref-first-logits.0" "$NEW_GATE_JSON" <<'PY'
import importlib.util, json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / 'tools/validation-harness/compare_logits.py'
spec = importlib.util.spec_from_file_location('fixed_gate', path)
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
a, b = map(pathlib.Path, sys.argv[2:4])
r = gate.compare(gate.read(a), gate.read(b))
r['bitexact'] = a.read_bytes() == b.read_bytes()
report = dict(limits=gate.LIMITS, one_request=r)
with open(sys.argv[4], 'x') as f: json.dump(report, f, indent=2)
print(json.dumps(report))
raise SystemExit(0 if r['numeric_pass'] and r['bitexact'] else 1)
PY
```

A bit-identical match gives zero logit error/KL/probability difference. The frozen
limits remain relative L2 1e-4, probability maxabs 1e-3, KL 1e-5 and the existing
absolute-logit envelope. A numeric PASS without bit identity must be reported as
such, not as established CPU equivalence. Any failure ends this single run and
returns the earliest difference; no automatic retry, tuning or extra request.
