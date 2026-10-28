# Shared Vulkan executor: tiny NVIDIA smoke on03

## Physical result: both RTX3060 devices passed

Hermes completed the bounded synthetic probe once per GPU, sequentially.
[Primary receipt](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793166423091349).
This coding task checked that receipt and records Hermes's execution; it did not
run the probes itself. This proves synthetic shared-executor residency, returned
row order and release on NVIDIA Vulkan. Full-model loading, routing, answer
quality and performance remain untested by this probe. It makes no AMD/NVIDIA
comparison and does not change the status of the separate AMD quality run.

The physical mapping was:

| Raw Vulkan index | PCI | UUID | Selected logical device |
| --- | --- | --- | --- |
| 0 | 01:00.0 | e1f2f90c-49ab-2689-0cf1-e5d9da520176 | 0 |
| 1 | 03:00.0 | a57bd3fb-c072-67ed-166c-ce52cf504ac0 | 0 |

Both devices were NVIDIA 10de:2504. llvmpipe at raw index 2 was excluded.
Each probe exited 0 and produced RESIDENT_ACCEPTED, RESIDENT_COMPLETED and final
COMPLETED. Each returned two experts, indices 0,2 in order 3,1, row_bytes 20480,
with bit_differences 0 against its same-device direct-hybrid reference. Owned
tensor bytes 10501120 were below the 16777216-byte tensor cap; final
released_bytes 0/live_allocations 0 and all 28 metric rows per GPU were finite.
The cap covers executor tensors, not all reference/backend/driver allocations.

Native CPU diagnostics were identical on both GPUs and were not bit-exact:

| Stage | Differing values | Maximum absolute error | Relative L2 |
| --- | ---: | ---: | ---: |
| Gate | 479 | 1.34110451e-7 | 1.27911482e-7 |
| Up | 489 | 1.19209290e-7 | 1.25771106e-7 |
| Hidden | 563 | 5.96046448e-8 | 2.01668318e-7 |
| Output | 467 | 5.58793545e-9 | 5.06116380e-8 |

The complete pinned source was built with disconnected fetching, only the probe
target and parallelism 2, using GGML pin 3cf03257f219afbe7334045ff7c6a06ac68c627d.
The Strata archive hash below identifies the exact exported source artifact.
Debian installed glslc 2025.2-1, libshaderc1 at the same version and
spirv-headers 1.6.1+1.4.309.0-1. No packages were upgraded or removed. No model,
wrapper, listener, clock, service or driver change occurred.

| Artifact | SHA-256 |
| --- | --- |
| Strata source archive | 592d9b7b670eab1b582c2b496fb37dc5667a0804213973b217802d60839712de |
| Pinned llama.cpp/GGML archive | cb7b39ca7e6fc5b9adc79d0ae0f14367207b59ae6a859d9a46156d1a3c697889 |
| Probe binary | a3aad410b23f366ebf06daac612faf8123a73788822ded340f331a9b5438bd42 |
| Evidence bundle | c8d32aa8ff33777d00470f4c796e4132d930969916875e4aeddad882aff2f4c7 |

Source, build, evidence and synthetic fixtures remain on 03 under
`/home/hermes/nvidia-vulkan-smoke-v1`. The control-side bundle/report remain under
`/home/zutfen/.hermes/cache/scratch/nvidia-vulkan-smoke-v1-control`.
No run-owned or NVIDIA compute processes remained; both GPUs returned to 1 MiB
used. No further fleet run or code change accompanies this report.

## Original pre-run handoff (historical)

The following recipe predates the successful physical result above. Its missing
prerequisites were resolved for that run; they are not current blockers.
No NVIDIA run or install was performed by this coding task. This is independent
of AMD correctness/performance and makes no comparative claim. Hermes's read-only
snapshot reports two RTX3060 devices, PCI01:00.0 and03:00.0, device10de:2504,
NVIDIA driver615.71.09, no compute processes and1 MiB used per device. Vulkan
loader/development headers/NVIDIA ICD exist; vulkaninfo also enumerates llvmpipe.
[Readiness receipt](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793164846282679).

## Exact prerequisites at handoff

Use a **complete export of the current PR45 head**, with exact commit/archive
SHA recorded across transfer, and the same GGML pin
3cf03257f219afbe7334045ff7c6a06ac68c627d. Do not reuse the older03 source tree
strata-cand-66235df1-src as if it were this candidate. No model weights required.

The tests/amd CMake project needs CMake>=3.24, working C/C++17 compiler/build tool,
Threads and Python3. Its Vulkan-enabled dependency explicitly requires:

- `find_package(Vulkan COMPONENTS glslc REQUIRED)` at pinned ggml-vulkan/CMakeLists:9.
  This requires Vulkan loader/link library, headers including Vulkan-Hpp, and
  **glslc**. glslangValidator is not an accepted substitute in this build.
- `find_package(SPIRV-Headers CONFIG REQUIRED)` at line14. Confirm a discoverable
  compatible SPIRV-HeadersConfig.cmake/header tree, or a previously retained
  compatible prefix. Vulkan development headers alone do not establish this.
- Preserve the isolated build's default worker-TLS setting; its existing OpenSSL
  development package must resolve if TLS is enabled. No endpoint is started.

Hermes confirmed glslc is absent from PATH; that is the definite missing tool.
A known existing SDK path can instead be passed as Vulkan_GLSLC_EXECUTABLE.
SPIRV-Headers availability/version and03's distro/release remain unverified.
Read `/etc/os-release` before selecting an official package; do not guess it.

On Debian/Ubuntu the narrow compiler package is **glslc**, with normal package
manager dependencies, not a full Vulkan SDK or glslang-tools. Official sources:
[Debian package](https://packages.debian.org/bookworm/glslc),
[Ubuntu package](https://packages.ubuntu.com/noble/glslc).
If the separate SPIRV-Headers package is missing, report that exact configure
failure; use an existing compatible retained prefix or let parent resolve the
matching distro spirv-headers package. Do not install arbitrary tools, pull a
new GGML revision, change drivers or weaken a build check. No install command
is authorized by this handoff.

## Build only the shared probe

Hermes resolves fresh SOURCE03/BUILD03/CHECKS03/EVIDENCE03 directories and the
exact retained GGML03 and GLSLC03 paths. Keep old source/build/results untouched.
SPIRV_PREFIX03 is an existing compatible prefix if needed; empty uses system
package discovery. No FetchContent downloads or CUDA build.

```sh
set -euo pipefail
umask 077
: "${SOURCE03:?complete exact PR45 export}" "${BUILD03:?fresh build}"
: "${CHECKS03:?fresh receipts}" "${EVIDENCE03:?fresh evidence parent}"
: "${GGML03:?exact retained pinned GGML tree}" "${GLSLC03:?actual glslc executable}"
test ! -e "$BUILD03"
test -x "$GLSLC03"
mkdir "$CHECKS03" "$EVIDENCE03"
"$GLSLC03" --version > "$CHECKS03/glslc-version.txt"
cmake -S "$SOURCE03/tests/amd" -B "$BUILD03" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_GGML_DIR="$GGML03" -DSTRATA_AMD_PROBE_VULKAN=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DVulkan_GLSLC_EXECUTABLE="$GLSLC03" \
  -DCMAKE_PREFIX_PATH="${SPIRV_PREFIX03:-}" > "$CHECKS03/configure.log" 2>&1
cmake --build "$BUILD03" --parallel 2 --target strata-vulkan-q8-expert \
  > "$CHECKS03/build.log" 2>&1
sha256sum "$BUILD03/strata-vulkan-q8-expert" > "$CHECKS03/probe.sha256"
```

Retain full configure/shader/build errors and STOP if a dependency or shader
feature fails. No speculative alternate flags, SDK bootstrap or source edits.
Despite its historical tests/amd directory/name, this target's Vulkan executor
has no AMD hardware requirement and selects the explicit Vulkan device.

## One bounded two-expert acceptance per verified RTX3060

Resolve **raw Vulkan physical-device indices** under the exact launch ICD/loader
environment, paired with PCI01:00.0/03:00.0, vendor10de/device2504 and GPU UUID.
Do not assume nvidia-smi indices equal Vulkan indices. Do not select llvmpipe.
Keep that mapping and current no-process/memory snapshots in receipts. No clock,
ICD file, driver, service or system configuration changes are needed.

The pinned GGML visibility filter takes raw Vulkan enumeration indices and
remaps the selected device to logical0. This confines both probe and executor
contexts to the one confirmed device. After the first run, inspect its identity
and result before repeating exactly once on the other verified raw index.

```sh
: "${RAW_VK_INDEX03:?verified raw index for the first RTX3060, not llvmpipe}"
PROBE03="$BUILD03/strata-vulkan-q8-expert"
set +e
timeout --signal=TERM --kill-after=5s 120s \
  env GGML_VK_VISIBLE_DEVICES="$RAW_VK_INDEX03" \
  "$PROBE03" --backend vulkan --device 0 --native-q8-boundaries \
  --resident-experts-check --out-dir "$EVIDENCE03/first" \
  > "$EVIDENCE03/first.log" 2>&1
RC03=$?
set -e
printf '%s\n' "$RC03" > "$EVIDENCE03/first.exit"
test "$RC03" -eq 0
```

For the second GPU, use its independently verified raw index and new `second`
output/log/exit filenames. No retry loop. On timeout/error, retain partial output,
verify owned-child cleanup and STOP; do not start the other GPU.

Acceptance: manifest.txt must identify the selected backend Vulkan0, device0 and
NVIDIA GeForce RTX3060 description, consistent with the retained physical mapping.
There is no CPU backend fallback: missing Vulkan support/device/init/operations
fail. CPU computation present in this probe is the explicit numerical reference
and hybrid quantization/SwiGLU, not an alternative selected backend. Vulkan
software drivers are prevented by the verified preflight selection; the CLI
itself does not prove a PCI identity from the device name alone.

Require RESIDENT_ACCEPTED with experts2, indices 0,2, expert_order 3,1,
row_bytes 20480, bit_differences 0, tensor_budget16777216 and owned_tensor_bytes
no greater than that cap. Require RESIDENT_COMPLETED and final COMPLETED files,
resident_manifest.txt released_bytes 0/live_allocations 0, and finite metrics.
Both returned rows must match the same-device direct-hybrid references exactly;
report native CPU max-absolute/relative-L2/bit differences separately, without a
universal CPU bit-parity or model-quality pass.

Scope is synthetic H2560/FF640 Q8 experts1 and3, not a model/prompt/answer test.
The executor owns two resident graphs within16 MiB. The separate direct-reference
graph and backend/driver allocations are additional;16 MiB is not total process
VRAM. The existing diagnostic writes a small synthetic two-blob resident-cache
inside its new output directory; these are fixture bytes, not model/cache
regeneration. Keep the output/cache/metrics, measure its actual `du -sb`, and
retain before/after device usage and process-cleanup observations. No listener,
network inference, full-model wrapper, clock change or service is involved.

The actual CLI and resource/order/release assertions above were checked against
the handoff source. At handoff, this coding task had not performed03 compilation,
package resolution/install or NVIDIA execution; usable glslc and SPIRV-Headers
configuration were pending. Hermes subsequently resolved these and completed
the physical probe as recorded at the top of this document.
