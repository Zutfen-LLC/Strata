# Portable local Vulkan expert residency: bounded handoff

Draft implementation, for parent review and Hermes dispatch. No fleet run by
this coding task. Stacked on PR44. This is a product capability slice, not a
CUDA-versus-Vulkan performance verdict. No CUDA de-offload work.

The default has no resident plans and retains CPU expert execution. With plans,
the local adapter borrows the existing dense Vulkan backends, loads only selected
Q8 expert blobs directly from immutable mapped model ranges, verifies manifest
and selected digests, and retains their tensors. There is no listener, network
transfer, duplicate disk cache or full-model hash. Each expert has exactly one
owner. Cold slots use the existing CPU pool; GPU rows scatter to original slots.
The existing router, weights, original-order host F32 weighted merge and all dense
placement remain unchanged. GPU failure stops/poisons the step; no silent backend
fallback or automatic retry. Reset/replay remains explicit; changed model sources
require reopen, not reuse of stale resident weights.

## Memory and execution contract

Initial smoke: **16 experts total per die**, never 16 per layer. One expert payload
is 5,222,400 bytes, so each initial plan holds 83,558,400 bytes (79.69 MiB) of weights.

CLI options, indexed by the dense backend slot (0/1), not another device identity:

```text
--expert-plan 0:/absolute/plan0 --expert-plan 1:/absolute/plan1
--expert-tensor-mib 256,256
--expert-workspace-mib 128,128
--expert-free-reserve-mib 512,512
--expert-check-rows
```

Plans use the existing GpuPlan format and must match the model manifest exactly.
No hardcoded expert-count limit is imposed beyond existing plan bounds: explicit
larger plans/caps can be tested later without rebuilding. Duplicate ownership,
wrong model/digest, invalid shapes and missing/duplicate/nonfinite reply rows stop.

Preparation occurs **after** dense/head admission. The same per-device cap covers
existing dense graph/tensor/runtime bounds plus expert tensor cap and additional
workspace allowance; caps are not pooled. Backends are borrowed, serial and never
freed by the executor. Expert graphs die before the caller's backend/model owner.

The tensor cap includes actual padded expert tensors, graph inputs/outputs and
scratch, reconciled against the GGML dry-run size. Backend-private allocations
are separate: a 128 MiB/die allowance initially, with free-memory observations
before loading, between assets and around inference. Observed physical-device
usage growth above the dense-loaded baseline must fit actual owned tensor bytes
plus that allowance, and 512 MiB/die must remain free. These observations are not a
measurement of every transient driver allocation; retain outer per-die guards.
The existing roughly1.2 GiB/die dense/runtime observation is context, not a fixed
subtraction or a replacement for the current admission ledger.

Host admission additionally prices each expert's 2 MiB GGML metadata arena and
32 KiB persistent boundary buffers, plus 16 MiB per plan for the one selected
loading blob and plan/request/reply slack. Initial host metadata alone is 32 MiB
per die, independent of the GPU tensor cap. At larger placements, this grows with
expert count and is checked against the existing host anonymous cap.

Vulkan hybrid Q8 retains Q8 weights, host native activation/intermediate
quantization and host SwiGLU, but **is not CPU bit parity**. Existing Vulkan
expert controls report differences without a cross-backend tolerance acceptance.
`--expert-check-rows` adds CPU reference evaluations for resident slots and logs
finite row max-absolute/relative-L2 errors; it is diagnostic overhead and must be
off for later performance work. Do not invent a universal tolerance pass or relax
existing gates. Existing same-backend repeat/order controls and the frozen answer
screen remain separate checks; original full-model logit gates remain FAILED.

## Complete source and build, before any clock window

Parent exports this draft PR's exact committed tree with `git archive`; record
commit and archive SHA256 across transfer. Export the complete tree, not selected
Python files. Fresh isolated SOURCE/BUILD/CHECKS/ROOT on 05 must all live below
`/srv/models/inferswarm-artifacts/portable-local-residency-v1`. Preserve qualified
binaries, models, resident services and previous evidence. No installs/downloads.

Use the retained CMAKE and pinned dependency LLAMA_SRC, with the successful
PR43/Vulkan prefix/include flags. A metadata-free source archive needs verified
archive provenance, not invented Git metadata. Existing Python with regex only.

```sh
set -euo pipefail
umask 077
: "${SOURCE:?complete exact export}" "${BUILD:?fresh build path}"
: "${CHECKS:?fresh checks path}" "${ROOT:?fresh evidence parent}"
: "${CMAKE:?retained executable}" "${LLAMA_SRC:?retained pinned dependency}"
test ! -e "$BUILD"
mkdir "$CHECKS" "$ROOT"
"$CMAKE" -S "$SOURCE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE_MODEL=ON \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON -DSTRATA_PORTABLE_QSA=ON \
  -DSTRATA_BUILD_TESTS=ON -DGGML_VULKAN=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" -DCMAKE_PREFIX_PATH="${MODEL_CMAKE_PREFIX05:-}" \
  -DCMAKE_CXX_FLAGS="${MODEL_CXX_FLAGS05:-}" > "$CHECKS/configure.log" 2>&1
"$CMAKE" --build "$BUILD" --parallel 2 --target portable_model strata-gpu-manifest \
  portable_local_experts_test portable_timing_cli_test portable_cpu_experts_test \
  portable_model_test > "$CHECKS/build.log" 2>&1
"$(dirname "$CMAKE")/ctest" --test-dir "$BUILD" --output-on-failure \
  -R '^portable_(local_experts|timing_cli|cpu_experts|model)_test$' \
  > "$CHECKS/ctest.log" 2>&1
sha256sum "$BUILD/portable_model" > "$CHECKS/binary.sha256"
```

Run heavy model tests alone on the host, not concurrent with other heavy tests.
Stop on failure. Local synthetic CPU tests are not physical offload qualification.

## First physical smoke: one discovery token and one resident token

Parent/Hermes performs current idle/model-writer checks. Preserve all existing
host/cgroup/swap/per-die guards. Devices remain verified Vulkan0/1, PCI 07:00.0 and
0b:00.0, contiguous dense split 32/16, caps 7680 MiB each, CPU workers 12, threads 4.
The wrapper imports those same admission/guard/placement checks, starts only its
owned child, samples every 0.2 s, limits each run to 300 s and retains STOP/cleanup.
It performs no clocks or remote operations. Use Hermes's already-verified finite
high/temperature/clock guard if parent requests the full-clock profile; build
outside the window. Discovery plus smoke use at most 650 s plus owned cleanup,
inside the existing 30-minute window; no voltage/power/driver changes.

```sh
BINARY_SHA=$(sha256sum "$BUILD/portable_model" | cut -d' ' -f1)
SMOKE="$SOURCE/tools/validation-harness/portable_local_expert_smoke_v1.py"
timeout --signal=TERM --kill-after=15s 325s "$PYTHON05" -B "$SMOKE" \
  --binary "$BUILD/portable_model" --binary-sha256 "$BINARY_SHA" \
  --phase discover --out "$ROOT/discover"
"$BUILD/strata-gpu-manifest" plan \
  /home/hermes/strata-pr10-9ed478e-20261025-05-peer/assets/manifest \
  "$ROOT/discover/assignments-0.txt" "$ROOT/plan0"
"$BUILD/strata-gpu-manifest" plan \
  /home/hermes/strata-pr10-9ed478e-20261025-05-peer/assets/manifest \
  "$ROOT/discover/assignments-1.txt" "$ROOT/plan1"
timeout --signal=TERM --kill-after=15s 325s "$PYTHON05" -B "$SMOKE" \
  --binary "$BUILD/portable_model" --binary-sha256 "$BINARY_SHA" \
  --phase smoke --plan "$ROOT/plan0" --plan "$ROOT/plan1" --out "$ROOT/smoke"
```

Discovery consumes only the first retained prompt token (248045), max-new 0,
without GPU experts. It records actual routes and chooses 16 unique layer/expert
pairs per die's dense-layer range, sorted and frozen before resident output.
It is a functional placement-discovery operation, not a speed baseline. No new
quality prompts or tokenization is introduced. Plans select ownership, not routes.

Smoke consumes the same one token. Require exit 0, strict dense placements, actual
expert backend Vulkan0/1, **entries>0 on both dies**, finite row diagnostics and
row_checks=entries. A plan may miss after changed floating-point routing; zero
hits is a failed offload proof, not permission to retry/tune plans. Retain it and
report. No CPU bit-identical whole-model requirement is claimed for this mode.

Retained stderr fields:

- expert_device: assigned, backend/numerics, weight_bytes, tensor_bytes,
  tensor_scratch_bytes, tensor_cap, workspace_cap, metadata_bound, staging_peak,
  free_before/after, peak_growth, requests, entries, row_checks and row errors.
- expert_dispatch: cumulative gpu_entries and cold cpu_entries. Diagnostic CPU
  reference work is extra and is not counted as cold entries; original native
  pool counters include any reference evaluations.
- Existing expert_work/device_memory and full dense placements remain.

Verify owned process cleanup and any clock restoration. Stop at this first
physical checkpoint for independent review of rows, resources and actual hits.

## Subsequent explicit placements and answer screen

After the smoke is reviewed, parent may supply larger immutable assignment files
and new GpuPlans, tensor/workspace/free-reserve caps. Require admission on each
die and the host; never grow from free VRAM by assuming the metadata is free or
that both dies share one budget. Existing make_gpu_plan validates sorted unique
assignments and manifest identity. Retain the exact plan hashes for every run.

For the next answer check use the existing frozen suite, once, with resident
options forwarded by the unchanged owned-child quality adapter. Do not rerun the
native quality arm: retain its existing result and compare with the unchanged
scorer. This later stage is separate from the first smoke dispatch.

```sh
"$PYTHON05" -B "$SOURCE/tools/validation-harness/portable_quality_v1.py" \
  --gguf /srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
  portable --binary "$BUILD/portable_model" --binary-sha256 "$BINARY_SHA" \
  --suite /srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v2/suite \
  --expert-plan "0:$ROOT/plan0" --expert-plan "1:$ROOT/plan1" \
  --expert-tensor-mib 256,256 --expert-workspace-mib 128,128 \
  --expert-free-reserve-mib 512,512 --out "$ROOT/quality"
```

Exactly 12 cases, existing 600 s/case and 3600 s suite guards, no retries or logits.
That screen does not fit a single30-minute high window in its worst case: parent
must select an explicit shorter bound or a separately guarded operation; **do
not silently extend the clock window**. Smoke completion does not authorize
launching this later stage. Reuse the existing score command on 02 with the
retained native quality root; on 02 suite is
`/home/hermes/portable-quality-ac85ff3-v2/suite`. Baseline 10/12 means the scorer's
inconclusive-baseline exit 1 is expected, distinct from a runtime failure. Report
both scores, changed case IDs/tokens and any stops; do not change gold or gates.

Keep small private receipts/row metrics, plans/hashes, complete source/build
identity and cleanup/resource/clock evidence. Record `du -sb`. New05 source/build
and all outputs stay on the secondary disk. No weight/cache duplicates, source
archives of models, deletion, fleet SSH by this coding task, merge or deployment.
