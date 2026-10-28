# AMD05 existing CPU pool phase exposure — one bounded resident run

Pending parent review and Hermes dispatch. Do not alter the NVIDIA resident run
pinned to45513fd8. This slice is stacked on PR48 docs head
`92a58ceeb9be19f09d73bee83b7ba036289ed5ed`; the final handoff supplies this new
candidate's exact SHA and complete-source archive/checksum. No fleet run or new
performance claim is recorded here.

Only existing CPU pool accumulators are exposed. There are no new clocks in the
kernels, worker instrumentation, scheduling/precision/placement changes, or new
source-identity rules. See [timing semantics](../docs/PORTABLE_MODEL.md#existing-cpu-expert-pool-phase-deltas).
The previous [AMD resident result](portable-resident-amd-results-v1.md) remains
separate evidence. Shared compiled `strata_kernels_cpu`/`ggml-cpu` libraries within
one build do not establish a portable-only scalar kernel; separately deployed
native/portable build ISA parity remains unverified.

Local validation passed before publication: Vulkan-linked Release build; six
focused CTests in80.42s (provider idempotent snapshots/phase containment, reset
accumulation, real stdio finite deltas and unchanged one-shot output/logit controls);
52 portable Python tests in0.776s, including NaN/negative phase rejection; CLI
translation unit compiled without Vulkan; recipe shell syntax/local links and
git diff checks. No pool/kernel implementation changed. Physical timing results
for this exposure remain pending; this is not an additional hardware run.

## Fresh exact-source build

Parent exports `git archive <EXPECTED_SHA>` plus source.commit and SHA256 as in
[the resident recipe](portable-resident-hermes-v1.md). Hermes stages them under a
fresh `/srv/models/inferswarm-artifacts/portable-pool-<SHA>` root, preserving all
prior binaries/source/receipts. Reuse the successful retained05 CMake, pinned
GGML and Vulkan prefix/include/loader settings. No downloads or installs.

```sh
set -euo pipefail
umask 077
: "${ROOT:?fresh secondary-disk portable-pool-SHA root}"
: "${EXPECTED_SHA:?reviewed pool-phase candidate from final handoff}"
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
PRIOR=/srv/models/inferswarm-artifacts/portable-resident-45513fd8
# Revalidate accepted tokens against the unchanged frozen suite and local05 GGUF.
PYTHONPATH="$ROOT/source/tools/validation-harness" "$PYTHON05" -B - "$PRIOR/tokens.json" <<'PY'
import json,sys
from pathlib import Path
import portable_quality_v1 as P
tok=P.Q.tokenizer(P.S.MODEL_ROOT/'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf')
prompts,binding=P.frozen(tok,Path('/srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v2/suite'))
assert json.loads(Path(sys.argv[1]).read_text())==prompts[0]
assert P.Q.MAX_NEW==32
print(json.dumps(binding,sort_keys=True))
PY
printf '%s  %s\n' \
  74514c6c5329d2cd518ee8b13015621bc3ba67d3815e3a15753e823dffa8257f /home/hermes/amd-quality-pr37-split/plan-die0.plan \
  064002c422b53f3508b9a771cafe8b8604ca5786865542c61db3a15a094fa061 /home/hermes/amd-quality-pr37-split/plan-die1.plan \
  | sha256sum -c -
sha256sum "$PRIOR/tokens.json" /home/hermes/amd-quality-pr37-split/plan-die0.plan \
  /home/hermes/amd-quality-pr37-split/plan-die1.plan > checks/inputs.sha256
```

Stop if any build/check/identity fails. Run heavy model fixtures serially outside
the clock window. Retain the frozen-binding verifier output in the parent/Hermes
preflight receipt.

## Exactly one process; one excluded warmup plus three warm requests

After parent review, Hermes wraps this whole supervisor/model lifetime in the
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
  source/tools/validation-harness/portable_resident_v1.py \
  --binary "$ROOT/build/portable_model" --binary-sha256 "$BINARY_SHA" \
  --model-root /srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66 \
  --manifest /home/hermes/strata-pr10-9ed478e-20261025-05-peer/assets/manifest \
  --tokens-file /srv/models/inferswarm-artifacts/portable-resident-45513fd8/tokens.json \
  --max-new 32 \
  --plan /home/hermes/amd-quality-pr37-split/plan-die0.plan \
  --plan /home/hermes/amd-quality-pr37-split/plan-die1.plan \
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
Every request receipt additionally retains `cpu_pool_gu_ms_delta`,
`cpu_pool_quant_ms_delta`, `cpu_pool_down_ms_delta`, `cpu_pool_elapsed_ms_delta`.
All must be finite/nonnegative. Report these per request alongside existing
prompt/decode/selection/reset/count fields, never sum enclosing elapsed with its
phases. Whole-request pool counters are not split into prompt/decode counters.
Do not label forward wall time outside pool elapsed as dense compute; it includes
routing/preparation, GPU experts, dense/backend, transfers and reporting.

Preserve all request/raw/resource/clock/cleanup receipts, final placements,
source/archive/dependency/binary/input identities, test logs, summary and total
`du -sb "$ROOT"`. Maxima remain cumulative. Return the measured breakdown and
STOPs before choosing any optimization. No additional experiment is authorized
by success, and separately deployed build ISA parity remains unverified.
