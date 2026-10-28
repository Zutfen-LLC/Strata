> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# validation host coarse expert-provider/executor timing — one bounded run

Pending parent review/operator dispatch. This new runtime slice starts exactly at
PR50 source `a4e1d9f5ad2b807ef304d2c84607978bd41f3877`, retaining its head
admission change. The final handoff pins the new candidate SHA and complete
source archive. Do not change the separate pinned head-admission physical job,
PR49's measured source or the NVIDIA45513fd8 checkpoint.

The [PR49 receipt](portable-pool-phase-amd-results-v1.md) places about1.55s of
warm request time inside native CPU pool calls, versus about46.6s of prompt/decode
wall. The new explicit --expert-timings flag adds only two kinds of host-clock
scope: whole expert-provider callback and per-device executor infer_rows call.
No new GPU synchronization, per-op profiler, kernel/precision/scheduling/placement
change is introduced. Read the exact containment in
[PORTABLE_MODEL](../docs/PORTABLE_MODEL.md#coarse-expert-provider-and-executor-boundaries).
No performance or causal attribution is claimed before the measurement.

Local validation: fresh Vulkan-linked Release build; seven focused serial CTests
passed in80.89s, including PR50 memory admission, deterministic timer enabled/
disabled/unwind capture, provider reset/call/containment checks, executor attempts,
real stdio deltas and unchanged one-shot/logit controls. All54 portable Python
tests passed in0.908s, including flag forwarding and invalid GPU timing arrays.
The CLI translation unit compiled without Vulkan. Python compilation, recipe
bash syntax/local links and git diff checks passed. No fleet or new physical
boundary-timing run was performed by this task.

## Fresh exact-source build

Parent exports `git archive <EXPECTED_SHA>` plus source.commit and SHA256 as in
[the resident recipe](portable-resident-hermes-v1.md). operator stages them under a
fresh `${PRIVATE_ARTIFACT_1FAA4401}<SHA>` root, preserving all
prior binaries/source/receipts. Reuse the successful retained05 CMake, pinned
GGML and Vulkan prefix/include/loader settings. No downloads or installs.

```sh
set -euo pipefail
umask 077
: "${ROOT:?fresh secondary-disk portable-expert-boundary-SHA root}"
: "${EXPECTED_SHA:?reviewed boundary-timing candidate from final handoff}"
: "${CMAKE:?retained executable}" "${LLAMA_SRC:?retained pinned GGML}"
: "${PYTHON05:?existing qualified Python}"
cd "$ROOT"
test "$(cat source.commit)" = "$EXPECTED_SHA"
sha256sum -c source.tar.sha256
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
mkdir source build checks
tar -xf source.tar -C source
"$CMAKE" -S "$ROOT/source" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE_MODEL=ON \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON -DSTRATA_PORTABLE_QSA=ON \
  -DSTRATA_BUILD_TESTS=ON -DGGML_VULKAN=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" -DCMAKE_PREFIX_PATH="${MODEL_CMAKE_PREFIX05:-}" \
  -DCMAKE_CXX_FLAGS="${MODEL_CXX_FLAGS05:-}" > checks/configure.log 2>&1
"$CMAKE" --build build --parallel 2 --target portable_model portable_stdio_test \
  portable_stdio_cli_driver portable_model_test portable_cpu_experts_test \
  portable_local_experts_test portable_timing_cli_test > checks/build.log 2>&1
"$(dirname "$CMAKE")/ctest" --test-dir build --output-on-failure \
  -R '^portable_(stdio|stdio_cli|model|cpu_experts|local_experts|timing_cli)_test$' \
  > checks/ctest.log 2>&1
"$PYTHON05" -B -m unittest discover -s source/tools/validation-harness \
  -p 'test_portable*.py' > checks/harness-tests.log 2>&1
sha256sum build/portable_model > checks/binary.sha256
PRIOR=${PRIVATE_ARTIFACT_561FB951}
# Revalidate accepted tokens against the unchanged frozen suite and local05 GGUF.
PYTHONPATH="$ROOT/source/tools/validation-harness" "$PYTHON05" -B - "$PRIOR/tokens.json" <<'PY'
import json,sys
from pathlib import Path
import portable_quality_v1 as P
tok=P.Q.tokenizer(P.S.MODEL_ROOT/'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf')
prompts,binding=P.frozen(tok,Path('${PRIVATE_ARTIFACT_021ADBD1}'))
assert json.loads(Path(sys.argv[1]).read_text())==prompts[0]
assert P.Q.MAX_NEW==32
print(json.dumps(binding,sort_keys=True))
PY
printf '%s  %s\n' \
  74514c6c5329d2cd518ee8b13015621bc3ba67d3815e3a15753e823dffa8257f ${PRIVATE_ARTIFACT_07F8412C} \
  064002c422b53f3508b9a771cafe8b8604ca5786865542c61db3a15a094fa061 ${PRIVATE_ARTIFACT_730CE699} \
  | sha256sum -c -
sha256sum "$PRIOR/tokens.json" ${PRIVATE_ARTIFACT_07F8412C} \
  ${PRIVATE_ARTIFACT_730CE699} > checks/inputs.sha256
```

Stop if any build/check/identity fails. Run heavy model fixtures serially outside
the clock window. Retain the frozen-binding verifier output in the parent/operator
preflight receipt.

## Exactly one process; one excluded warmup plus three warm requests

After parent review, operator wraps this whole supervisor/model lifetime in the
existing verified finite high-clock/temperature guard, unchanged. Keep the
30-minute outer guard, temperature and sensor-failure STOP, active-clock samples
on both dies, owned cleanup and auto restoration. No clocks/voltage/power/driver
changes beyond that existing guarded profile; no new guard framework.

```sh
set -euo pipefail
cd "$ROOT"
sha256sum -c checks/inputs.sha256
BINARY_SHA=$(cut -d' ' -f1 checks/binary.sha256)
timeout --signal=TERM --kill-after=15s 1000s "$PYTHON05" -B \
  source/tools/validation-harness/portable_resident_v1.py --expert-timings \
  --binary "$ROOT/build/portable_model" --binary-sha256 "$BINARY_SHA" \
  --model-root ${PRIVATE_ARTIFACT_9AA3D2B0} \
  --manifest ${PRIVATE_ARTIFACT_32CC92CE} \
  --tokens-file ${PRIVATE_ARTIFACT_35AD3B32} \
  --max-new 32 \
  --plan ${PRIVATE_ARTIFACT_07F8412C} \
  --plan ${PRIVATE_ARTIFACT_730CE699} \
  --devices 0,1 --pci 0000:07:00.0,0000:0b:00.0 --split 32 \
  --device-cap-mib 7680 --out "$ROOT/run" > checks/supervisor.log 2>&1
```

Same retained480 plans, tensor2404MiB/die, workspace128MiB/die, free reserve512MiB,
totalcap7680MiB/die, split32/16, CPUworkers12/threads4. No row checks, retuning,
native rerun, model copy or retries. Existing host/cgroup working-set admission,
8GiB reserve, no new swap-out/child swap, per-die growth/free/cap guards and bounded
logs remain active for the whole process. Bounds remain load180s, request180s,
process960s, normal exit15s, forced owned cleanup at most10s; outer timeout1000s.

Require the previous lifecycle controls: four complete identical same-backend
token/greedy rows, positive per-die entry deltas each request, warm zero CPU
verification/copy/digest/invalidation work, exit0/child_reaped and no owned process.
Every request additionally retains scalar expert_provider_ms_delta and
expert_provider_calls_delta plus ordered per-device gpu_infer_rows_ms_delta and
gpu_infer_rows_calls_delta arrays. Existing CPU pool fields remain present.
Require provider_calls = consumed token rows *48 and positive per-device timing/
call counts alongside the existing per-device GPU-entry deltas. The adapter
validates finite/nonnegative timing arrays and retains all fields.
All must be finite/nonnegative. Report these per request alongside existing
prompt/decode/selection/reset/count fields, never sum enclosing provider with GPU/pool intervals, or enclosing pool
elapsed with its phases. Whole-request pool counters are not split into prompt/decode counters.
Do not label forward wall time outside pool/provider elapsed as dense compute; it includes
routing/preparation, GPU experts, dense/backend, transfers and reporting.

Preserve all request/raw/resource/clock/cleanup receipts, final placements,
source/archive/dependency/binary/input identities, test logs, summary and total
`du -sb "$ROOT"`. Maxima remain cumulative. Return the measured breakdown and
STOPs before choosing any optimization. No additional experiment is authorized
by success, and separately deployed build ISA parity remains unverified.
