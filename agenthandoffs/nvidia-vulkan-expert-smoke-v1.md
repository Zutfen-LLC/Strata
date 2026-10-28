> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Shared Vulkan executor: tiny NVIDIA smoke on03

Handoff only; no NVIDIA run or install by this coding task. This is independent
of AMD correctness/performance and makes no comparative claim. operator's read-only
snapshot reports two RTX3060 devices, PCI01:00.0 and03:00.0, device10de:2504,
NVIDIA driver615.71.09, no compute processes and1 MiB used per device. Vulkan
loader/development headers/NVIDIA ICD exist; vulkaninfo also enumerates llvmpipe.
[Readiness receipt](PRIVATE_RECEIPT).

## Exact prerequisites and current blocker

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

operator confirmed glslc is absent from PATH; that is the definite missing tool.
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

operator resolves fresh SOURCE03/BUILD03/CHECKS03/EVIDENCE03 directories and the
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

Require RESIDENT_ACCEPTED with experts2, indices0,2, expert_order3,1,
row_bytes20480, bit_differences0, tensor_budget16777216 and owned_tensor_bytes
no greater than that cap. Require RESIDENT_COMPLETED and final COMPLETED files,
resident_manifest.txt released_bytes0/live_allocations0, and finite metrics.
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
current source. No03 compilation, package resolution/install or NVIDIA execution
was performed here. The first concrete prerequisite remains usable glslc; exact
distro package applicability and SPIRV-Headers configuration must be confirmed
before parent dispatches the build.
