# First-request trace for issue 18

This is instrumentation, not a numerical fix. The starting revision is
`d1a7506a0cdb18fb445483c29f1bcfd04fce8a80` on draft PR 10. The original supplied
checkout was clean at `982ea405`; work uses a separate checkout. No fleet connection,
model load, worker start, access change or old offline control was performed here.

## Capture and stop

Set `STRATA_FIRST_REQUEST_TRACE` to a new absolute file. Also set the existing
`STRATA_DUMP_FIRST_LOGITS` prefix. Both arms use the same candidate engine. The
trace is disabled when its environment variable is absent or empty.

The original GEN length, prompt processing, verify window size, graph grouping,
expert arithmetic and ordered reducer stay in place. The trace stops the first
request after the existing first-logit calculation and successful dump, before
commit, token emission, further drafts or continuation windows. Normal request
cleanup emits `DONE 0 ... trace ...`; the existing driver can then send QUIT and
reap the engine. There is no process exit from a callback. A second request is not
traced. Use one request per fresh engine, with no warmup or resume.

The serial remote CPU path already reads the prompt through verify windows. No
prefill path is changed. Batch, vision, layer splits, pipelined windows, the old
CPU capture and the legacy network CUDA mode are refused with tracing enabled.
The supported experiment is the existing CPU group and ordered CPU/network CUDA
union. Ordinary launches keep their current settings.

Each record identifies its window, position, layer and token slice. Retained data:

- Original prompt IDs, original GEN length, prefill setting, spec setting, context,
  model geometry and every window's actual IDs.
- Every token's F32 MoE input, full router scores, selected IDs and weights.
- Every token's post-MoE output after the local/helper/network merge.
- The actual weighted CPU/network union passed to the existing accumulation, with
  zeros when that slice has no network contribution. No diagnostic reduction is added.
- Dispatch ownership for every entry and each active CUDA peer's original-index
  mask. Kinds are -2 skipped, -1 local CPU, 0 primary VRAM, 1 PCIe, 2 helper,
  3 network CPU, 4 network CUDA. Peer records follow configured endpoint order.
- All residual channels and injection coefficients for the **original final prompt
  token**, at each layer's FFN read. This is the residual before the pending MoE
  write. The engine defers that write to the next layer. Do not interpret it as
  a second, newly reduced post-MoE tensor.
- The first-logit row using the existing `merged ? T-1 : 0` convention.

GPU snapshots are D2D copies recorded on the producing graph stream, before
scratch reuse. The host reads them only after the existing graph completion and
copy-stream waits. The ownership and network union records are copied after all
workers finish. File-write failure is latched until the graph drains; it cannot
abandon a GPU doorbell wait midway through a layer.

Limits: at most 64 original prompt tokens, 32 windows, 64 MiB on disk and 64 MiB of
additional device snapshots. Host snapshot scratch holds one layer, bounded by
the same 64 MiB ceiling, plus one vocabulary row and existing worker storage.
Oversize geometry, a missing prompt window, allocation failure or write failure
ends with ERR rather than a successful seal. Existing files are never replaced;
POSIX trace permissions are 0600. The offline reader caps each input at 64 MiB.
Keep raw activations and prompt data private.

## Hermes invocation changes

The parent must first resolve its pending access approval. This document does not
authorize creating routes. Use the retained config-driven bundle/run lifecycle,
resident identity checks and cleanup. Keep the Q6 acquisition and unrelated
services untouched. Do not make diagnostic STATUS calls during an engine session.

Use the launch/config/request assets under the retained
`strata-fanout-live-982ea405/A0` and `B-recovery` roots as bindings. Do not edit
those evidence roots. Make two fresh diagnostic configs/run roots and change only:

| Binding | A0 trace | B trace |
| --- | --- | --- |
| Engine asset | Same newly built candidate | Same newly built candidate |
| Request selection | `[0]`, original plain GEN256 line | Same original line |
| Reference token-array gate | Absent, because this diagnostic emits zero tokens | Absent |
| `capture_first_logits` | `true` | `true` |
| `env.STRATA_FIRST_REQUEST_TRACE` | New absolute `A0.trace` | New absolute `B.trace` |
| `STRATA_DUMP_FIRST_LOGITS` | Existing driver supplies unique A0 prefix | Existing driver supplies unique B prefix |
| All other launch/environment/asset bindings | Original A0 | Original B |

Keep all 38 prompt IDs, `--prefill 64`, `--spec 2`, `--max-context 1024`, the
original GEN256 binding, both local GPU placements, CPU resident, manifests,
worker binaries, policies and B's four CUDA peers unchanged. Do not set GEN1,
`--max-new 1`, disable speculative scheduling, add a warmup or switch prompt
paths. There are no new flags on workers and no new wire capability.

Exact added environment binding at the engine invocation is:

```sh
STRATA_FIRST_REQUEST_TRACE="$TRACE_ABSOLUTE_NEW_FILE" \
STRATA_DUMP_FIRST_LOGITS="$FIRST_LOGITS_ABSOLUTE_NEW_PREFIX" \
"$CANDIDATE_ENGINE" "${ORIGINAL_ARM_ARGV[@]}"
```

Use that environment through the existing driver, not a separate unmanaged engine
launch. Run its normal commands once for each diagnostic config:

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$TRACE_CONFIG"
python3 -B "$TRACE_RUN_ROOT/run.py" --run-root "$TRACE_RUN_ROOT"
```

Require one DONE with finish `trace`, zero emitted tokens, one first-logit file
with ordinal `.0`, and no ERR. Then QUIT/reap and check the existing owned idle
receipt before the next arm. These timings are instrumented and have no throughput
meaning. Do not retry a failed arm or start another experiment automatically.

After both arms finish, run exactly one comparison:

```sh
python3 tools/compare_first_request_trace.py "$A0_TRACE" "$B_TRACE" \
  --a0-logits "$A0_FIRST_LOGITS_PREFIX.0" \
  --b-logits "$B_FIRST_LOGITS_PREFIX.0" \
  --output "$RESULT_ROOT/first-request-comparison.json"
```

The reader rejects incomplete traces, changed prompt IDs/window positions and shapes, missing
layers/slices, changed request settings and a trace logit row that differs from
its separate first-logit artifact. The report's `request_binding` starts with
`[256, 64, 2, 1024]` for this experiment. Ownership changes are expected; they do
not themselves count as numerical divergence. Changed speculative token rows are
reported explicitly; values on those rows no longer compare equal token inputs.
The original prompt rows must still match their recorded prompt IDs.

Output gives the earliest retained bit difference, first routing difference,
max-absolute and relative L2 error at every retained boundary, and maxima per
stage with window/layer/token coordinates. Exit 0 means the comparison completed,
**not** that numerical correctness passed. Exit 2 refuses an invalid comparison.
Keep the existing strict final-logit gate unchanged and report its result too.
Compare A0 against its retained exact first-logit reference before interpreting B.

A small early difference is not proof of the final error's cause. Follow growth
through inputs, network union, post-MoE output and router changes. If an already
large difference enters the first retained boundary, or if a differing union
needs individual expert attribution, stop and report that limit. This patch does
not retain all expert rows or internal attention/GDN state; a second selected
boundary would need separate justification. Do not infer that the old regrouping
bug returned merely from a nonzero error.

## Validation of this patch

Passed locally on Linux x86-64, GCC 16.2.1, Python 3.14.8:

```sh
python -m unittest discover -s tests/core -p test_first_request_trace.py -v
cmake -S . -B /tmp/strata-first-request-trace-build \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/strata-first-request-trace-build --target first_request_trace_test -j 2
ctest --test-dir /tmp/strata-first-request-trace-build -R first_request_trace --output-on-failure
```

Three Python tests passed. They execute the real writer with tracing disabled and
enabled, check exclusive creation, a vocabulary over 64k, the bounded failure
latch, no early seal before the final prompt window and no second-request trace.
Synthetic trace comparisons check growth, routing change, unchanged data,
misaligned IDs and truncation. Source placement checks establish the diagnostic
break precedes commit and the post-MoE snapshot follows the remote merge. The
Release CMake disabled-path test passed. These are not model correctness tests.

All three modified engine translation units passed a host syntax check with CUDA
runtime headers, native experts and network CUDA worker enabled:

```sh
g++ -std=c++20 -fsyntax-only -mavx2 -mfma '-DSTRATA_VERSION="0.1.45.3"' \
  -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_NETWORK_CUDA_WORKER=1 -Iinclude \
  -I/tmp/cuda-hdrs/nvidia/cuda_runtime/include \
  -I/tmp/cuda-hdrs/nvidia/cuda_nvcc/include \
  src/core/expert_source.cpp src/core/verify.cpp src/program/generate.cpp
```

The model-free `first_request_trace_cuda_test` compiled and linked against the
CUDA runtime. Execution returned skip code 77: no CUDA device is available here.
It shares the production layout/copy helper and checks T=1,3,4, two layers,
individual token slices, scratch overwrite after every snapshot, untouched padded
rows and two graph replays. On Hermes's existing configured CUDA build:

```sh
cmake --build "$CUDA_BUILD" --target strata first_request_trace_cuda_test -j "$BUILD_JOBS"
ctest --test-dir "$CUDA_BUILD" -R '^first_request_trace_cuda_test$' --output-on-failure
```

Failed: no remaining test failure. The first standalone syntax command used
C++17 and omitted the normal version definition, so generate.cpp rejected that
command; the corrected C++20 command above passed.

Not run: complete CUDA/HIP backend builds, GPU graph-test execution, the actual
A0/B request, or default-path numerical byte comparison. There is no CUDA or HIP
compiler/device in this desktop environment. SYCL has its own unchanged source
copies. Keep PR 10 as a draft until applicable compilation and the focused CUDA
checks run. Default-path preservation is supported by guarded allocation/I/O and
unchanged arithmetic/call order in source, not by an unperformed hardware run.
