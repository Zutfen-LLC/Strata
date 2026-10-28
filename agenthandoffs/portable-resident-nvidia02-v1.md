> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# validation host resident checkpoint — no code change

The AMD resident checkpoint [passed](portable-resident-amd-results-v1.md).
validation host dispatch remains pending parent review. Reuse PR48 runtime commit
`45513fd885babcf81d252b73804a14d2a751a74a`, not a rebuilt/moved branch head.
The existing adapter supports all required explicit paths/device options.
No fleet action was performed while preparing this recipe. No clock changes,
new placement, model copy, native rerun or retries.

operator stages the existing complete source archive and small metadata only under
`${PRIVATE_ARTIFACT_BABCDE0A}`: source.tar/source.commit, the two retained
480 plans renamed plan-die0.plan/plan-die1.plan, and the **verified AMD run's**
tokens.json plus its frozen-binding/max-new receipt. Copy the tokens hash from
that reviewed receipt into TOKENS_SHA256; do not guess/re-tokenize a different
prompt. Existing suite case1 has max-new32. Alternatively verify the existing
suite with `portable_quality_v1.frozen` using the local02 model and take prompts[0].

## Fresh build and focused checks

Use the exact retained CMAKE, LLAMA_SRC, prefix/include flags and loader/ICD
environment from the passing PR47 build under
`${PRIVATE_ARTIFACT_7A6C4B63}`. Resolve these host-local paths from its
retained configuration; do not install/download dependencies or change drivers.
The variables below must be actual retained paths/settings, not05 paths.

```sh
set -euo pipefail
umask 077
ROOT=${PRIVATE_ARTIFACT_BABCDE0A}
MODEL_ROOT=${PRIVATE_ARTIFACT_E45E0609}
MANIFEST=${PRIVATE_ARTIFACT_3B7D17F7}
EXPECTED_SHA=45513fd885babcf81d252b73804a14d2a751a74a
: "${CMAKE:?retained PR47 CMake executable on02}"
: "${LLAMA_SRC:?retained pinned GGML source on02}"
: "${PYTHON02:?qualified existing Python on02}"
: "${TOKENS_SHA256:?reviewed05 case1 tokens.json hash}"
# MODEL_CMAKE_PREFIX02 and MODEL_CXX_FLAGS02 reuse the successful02 settings.
cd "$ROOT"
test "$(cat source.commit)" = "$EXPECTED_SHA"
printf '%s  %s\n' \
  ffc5f6a885bb6ed9aee1b12c225feb177598a99bd3afeef8cd322e746b5ee52f source.tar \
  74514c6c5329d2cd518ee8b13015621bc3ba67d3815e3a15753e823dffa8257f plan-die0.plan \
  064002c422b53f3508b9a771cafe8b8604ca5786865542c61db3a15a094fa061 plan-die1.plan \
  8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318 "$MANIFEST" \
  "$TOKENS_SHA256" tokens.json | sha256sum -c -
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
mkdir source build checks
# mkdir must fail if any target already exists; preserve prior builds/evidence.
tar -xf source.tar -C source
"$CMAKE" -S "$ROOT/source" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE_MODEL=ON \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON -DSTRATA_PORTABLE_QSA=ON \
  -DSTRATA_BUILD_TESTS=ON -DGGML_VULKAN=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" -DCMAKE_PREFIX_PATH="${MODEL_CMAKE_PREFIX02:-}" \
  -DCMAKE_CXX_FLAGS="${MODEL_CXX_FLAGS02:-}" > checks/configure.log 2>&1
"$CMAKE" --build build --parallel 2 --target portable_model portable_stdio_test \
  portable_stdio_cli_driver portable_model_test portable_cpu_experts_test \
  portable_local_experts_test portable_timing_cli_test > checks/build.log 2>&1
"$(dirname "$CMAKE")/ctest" --test-dir build --output-on-failure \
  -R '^portable_(stdio|stdio_cli|model|cpu_experts|local_experts|timing_cli)_test$' \
  > checks/ctest.log 2>&1
"$PYTHON02" -B -m unittest discover -s source/tools/validation-harness \
  -p test_portable_resident_v1.py -v > checks/harness-tests.log 2>&1
sha256sum build/portable_model > checks/binary.sha256
"$ROOT/build/portable_model" --list-vulkan > checks/vulkan.tsv 2> checks/vulkan.log
nvidia-smi --query-gpu=pci.bus_id,uuid,name,memory.used,memory.total,temperature.gpu,clocks.sm \
  --format=csv > checks/nvidia-before.csv
```

Stop if the logical mapping has drifted: expected Vulkan0→0000:02:00.0 RTX3060Ti,
Vulkan1→0000:03:00.0 RTX3060. Keep the same loader/visibility environment for listing
and execution. The supervisor independently verifies this exact mapping against
physical PCI telemetry; NVIDIA management indices are not Vulkan indices.

## Exactly one model process; one excluded warmup and three measured GENs

After parent review, and after the tests pass, run once at the existing NVIDIA
clock settings. Preserve PR47's idle/model-writer and health checks. This starts
no service and changes no clocks, cgroups, drivers, credentials or other processes.

```sh
set -euo pipefail
cd "$ROOT"
BINARY_SHA=$(cut -d' ' -f1 checks/binary.sha256)
timeout --signal=TERM --kill-after=15s 1000s "$PYTHON02" -B \
  source/tools/validation-harness/portable_resident_v1.py \
  --binary "$ROOT/build/portable_model" --binary-sha256 "$BINARY_SHA" \
  --model-root ${PRIVATE_ARTIFACT_E45E0609} \
  --manifest ${PRIVATE_ARTIFACT_3B7D17F7} \
  --tokens-file "$ROOT/tokens.json" --max-new 32 \
  --plan "$ROOT/plan-die0.plan" --plan "$ROOT/plan-die1.plan" \
  --devices 0,1 --pci 0000:02:00.0,0000:03:00.0 \
  --split 32 --device-cap-mib 7680 --out "$ROOT/run" \
  > checks/supervisor.log 2>&1
```

The adapter uses host cap32GiB, CPU workers12/threads4, tensor2404MiB/GPU,
workspace128MiB/GPU, reserve512MiB/GPU. Both total caps remain7680MiB even on the
12GiB card; no pooling or rank/placement changes. Retained plans must bind the
local02 manifest/model identities; the existing loader checks that before use.
A mismatch is STOP, not permission to regenerate plans or swap a manifest.

Same boundaries as AMD: process960s, load180s, request180s, normal-exit15s,
forced TERM/KILL/reap at most10s; output caps1MiB stdout/16MiB stderr. Host/cgroup
working-set admission and8GiB reserve, no-new-swap-out, child VmSwap0, physical
per-GPU capacity/growth/free-reserve checks cover the whole model lifetime.
NVIDIA observations use PCI-addressed nvidia-smi and MiB granularity, including
other device users; failed queries/capacity drift stop. The sampling target is
0.2s, with actual cadence bounded by telemetry query duration (5s timeout/GPU).
Retain existing host02 health observations without introducing an AMD clock guard.

Require four completed receipts, identical same-NVIDIA-backend token/greedy rows,
positive per-device GPU entry deltas on every request, and zero CPU verification,
copy, digest and source-invalidation deltas for requests2–4. Exclude request1 only
from timing aggregates; retain its lazy verification evidence. Compare NVIDIA
repeats to themselves, not bit identity with AMD/native. Keep cumulative maxima
separate from request deltas. Require exit_code0 and child_reaped; verify no owned
model remains. Preserve any STOP with no retry.

Retain source/archive/dependency/binary/input identities, argv, six CTests/six
harness tests, all four request JSONs, stdout/stderr, placements, summary, before/
after NVIDIA observations, cleanup receipt and `du -sb "$ROOT"`. Do not overwrite
PR47 or AMD evidence. No performance or original strict-parity verdict follows
from this checkpoint.

Genuine inputs still needed on02: transferred small plans with the hashes above;
verified case1 tokens hash/binding (or a locally verified existing suite); retained
PR47 toolchain/include/loader settings; current idle headroom and exact PCI mapping.
The full02 model path and manifest identity are supplied by parent, not inferred
from05 paths. Parent owns review and operator dispatch after the AMD result.
