# Portable resident requests: exact-source Hermes handoff

Pending parent review. Do not run on fleet hardware before that review. This
slice is stacked on PR47 source `fb2ea3c92093eacc8663d2b90b88813ec2bc4076` in
Zutfen-LLC/Strata. The final handoff supplies the exact candidate commit SHA.
No fleet execution, clock operation, model copy or placement retuning occurred
in this coding task. Original strict numerical parity remains FAILED.

## Stage and build outside the clock window

Parent exports the complete committed tree (not selected source files):

```sh
set -euo pipefail
: "${EXPECTED_SHA:?exact candidate SHA from final handoff}"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
git archive --format=tar "$EXPECTED_SHA" > source.tar
sha256sum source.tar > source.tar.sha256
printf '%s\n' "$EXPECTED_SHA" > source.commit
```

Hermes stages that archive, checksum and source.commit on05 under a fresh
`/srv/models/inferswarm-artifacts/portable-resident-<SHA>` directory. Preserve old
binaries and all evidence. Verify archive SHA256 after transfer and require
source.commit equals EXPECTED_SHA. Extract only into a fresh source subdirectory;
retain the archive and receipts. This full source archive plus the existing
pinned dependency is the complete build input; do not download or install.

```sh
set -euo pipefail
umask 077
: "${ROOT:?fresh /srv/models/inferswarm-artifacts/portable-resident-SHA}"
: "${EXPECTED_SHA:?reviewed exact candidate SHA}" "${CMAKE:?retained executable}"
: "${LLAMA_SRC:?retained pinned llama.cpp}" "${PYTHON05:?existing qualified Python}"
cd "$ROOT"
test "$(cat source.commit)" = "$EXPECTED_SHA"
sha256sum -c source.tar.sha256
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
mkdir source checks
mkdir build
# This must be a new empty build directory, never a prior qualified build.
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
  -p test_portable_resident_v1.py -v > checks/harness-tests.log 2>&1
sha256sum build/portable_model > checks/binary.sha256
```

Stop on any failure. Run heavy sparse-model tests serially. They need several GiB
of RAM but no complete model payload. The native CUDA/HIP/SYCL executables are
unchanged; those builds and physical AMD execution are not established locally.

Freeze case1 from the existing suite, using its unchanged verifier/tokenizer:

```sh
PYTHONPATH="$ROOT/source/tools/validation-harness" "$PYTHON05" -B - \
  "$ROOT" <<'PY'
import json,sys
from pathlib import Path
import portable_quality_v1 as P
root=Path(sys.argv[1])
gguf=P.S.MODEL_ROOT/'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf'
tok=P.Q.tokenizer(gguf)
prompts,binding=P.frozen(tok,Path('/srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v2/suite'))
(root/'tokens.json').write_text(json.dumps(prompts[0])+'\n')
(root/'checks/frozen-binding.json').write_text(json.dumps(binding,sort_keys=True)+'\n')
(root/'checks/max-new').write_text(str(P.Q.MAX_NEW)+'\n')
PY
sha256sum tokens.json \
  /home/hermes/amd-quality-pr37-split/plan-die0.plan \
  /home/hermes/amd-quality-pr37-split/plan-die1.plan > checks/inputs.sha256
```

## One guarded process, then stop

Hermes uses the existing verified finite high-clock/temperature guard unchanged,
covering the **whole** supervisor/model process from load through cleanup. Retain
sensor-failure and temperature STOP, active-clock samples on both dies and auto
restoration. No voltage/power/driver changes. Keep the prior 30-minute outer guard;
this run has a 960 s model-process cap, 180 s load cap, 180 s/request cap, 15 s
normal-exit cap, plus at most10 s forced owned cleanup. The command below has a
1000 s outer timeout and fits that guard. No sleeps between GENs, no retries, no
native rerun or full twelve-case campaign. Build and checks above are outside the
clock window. The guard's already verified host-local invocation remains Hermes's
responsibility; do not invent a replacement or run the command unguarded.

```sh
set -euo pipefail
cd "$ROOT"
sha256sum -c checks/inputs.sha256
BINARY_SHA=$(cut -d' ' -f1 checks/binary.sha256)
timeout --signal=TERM --kill-after=15s 1000s "$PYTHON05" -B \
  source/tools/validation-harness/portable_resident_v1.py \
  --binary "$ROOT/build/portable_model" --binary-sha256 "$BINARY_SHA" \
  --tokens-file "$ROOT/tokens.json" --max-new "$(cat checks/max-new)" \
  --plan /home/hermes/amd-quality-pr37-split/plan-die0.plan \
  --plan /home/hermes/amd-quality-pr37-split/plan-die1.plan \
  --devices 0,1 --pci 0000:07:00.0,0000:0b:00.0 --split 32 \
  --device-cap-mib 7680 --out "$ROOT/run" > checks/supervisor.log 2>&1
```

Physical settings remain480 experts/die, tensor2404MiB/die, runtime128MiB/die,
reserve512MiB/die, total cap7680MiB/die, dense32/16, CPU workers12/threads4.
The adapter starts exactly one `--stdio` model child and sends four identical GENs,
then QUIT. It also runs the existing read-only `--list-vulkan` identity query before
loading; that query does not load a model. No intermediate GEN reloads providers,
weights or plans. Requests2–4 must have zero verification/copy/digest/invalidation
deltas and identical token/greedy rows to request1. Every request needs positive
entry deltas on both dies. No row-check diagnostic overhead or new ranking.

The existing host/cgroup-v2 working-set admission,8GiB effective host reserve,
no-new-swap-out and child VmSwap checks, per-die VRAM growth cap and512MiB free
reserve run at0.2 s intervals across load, idle/request work and normal exit.
The failing sample and compact extrema are retained before STOP. Forced cleanup
terminates only the owned process group and waits/reaps it; it does not use a
failed session report as resource evidence. SIGINT/SIGTERM during a backend call
are cooperative; the supervisor's hard cleanup deadline covers stalls.

Preserve all four request JSON files, stdout/stderr, final placements, summary,
argv, source/archive/dependency/binary/input hashes, ctest/harness logs, outer
clock/sensor/restore receipts and final `du -sb "$ROOT"`. Report load separately
from warmup and the three measured requests, with actual tokens and setup/reset,
prompt, decode and selection timings plus verification/hit and per-die GPU-entry
deltas. Maxima in expert_device_cumulative are lifetime maxima. Warmup is excluded
from measured request aggregates but kept as first-use evidence. Do not aggregate
a failed/partial request. Check exit_code0, child_reaped, all four completion
receipts, GPU activity on both dies, cleanup, restored clocks and no owned model
or guard remaining. On any STOP preserve the result and return it to parent.

This establishes a fairer lifecycle and usable resident operation. It does not
attribute the earlier latency gap without stage evidence, claim a speedup,
establish cross-backend numerical identity, or alter retained failed parity gates.
