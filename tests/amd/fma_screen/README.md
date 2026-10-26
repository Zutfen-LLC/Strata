# One-die Vulkan FMA screen

This standalone diagnostic answers whether the installed Vulkan compiler returns
single-rounding FP32 FMA results for three finite witnesses. The fourth case is a
control. It does not run ggml, an expert projection, a model, or a worker.

The source-backed concern and its limits are in
[Q8_REFERENCE_FEASIBILITY.md](../Q8_REFERENCE_FEASIBILITY.md). Source inference
is not installed-driver execution evidence. This screen is the next measurement;
passing it does not establish real-expert or universal exactness.

## Build with tools already installed

Run from the repository root. Use a fresh output directory; retain all files.
No package install, downloads, CMake configuration or ggml shader build is needed.
A C++17 compiler, Vulkan SDK headers/loader, `glslc` must already be available. `spirv-val` is optional when already
installed; do not install anything for this screen. Vulkan 1.2 headers are sufficient; older headers can report that
optional feature queries are unavailable while still reporting extension names.

```sh
set -eu
screen_dir=$(mktemp -d "${TMPDIR:-/tmp}/strata-fma-screen.XXXXXXXX")
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -Wno-missing-field-initializers -ffp-contract=off \
  tests/amd/fma_screen/fma_screen.cpp -lvulkan -o "$screen_dir/fma-screen"
glslc --target-env=vulkan1.2 -O0 tests/amd/fma_screen/fma_probe.comp \
  -o "$screen_dir/fma_probe.spv"
if command -v spirv-val >/dev/null 2>&1; then
  spirv-val --target-env vulkan1.2 "$screen_dir/fma_probe.spv"
else
  printf '%s\n' 'SPIR-V validation not run: spirv-val absent' > "$screen_dir/validation-not-run.txt"
fi
c++ --version > "$screen_dir/tool-versions.txt"
glslc --version >> "$screen_dir/tool-versions.txt"
sha256sum "$screen_dir/fma-screen" "$screen_dir/fma_probe.spv" \
  > "$screen_dir/sha256.txt"
"$screen_dir/fma-screen" --host-check > "$screen_dir/host-check.log" 2>&1
```

`--host-check` does not call Vulkan. It checks the four oracle constants with
host `std::fma` and separately rounded FP32 operations; it is not qualification
of the native GCC14.2 AVX2 reference. The optional `fma_avx2_witness.cpp` constructs
the positive Q8 case with AVX2 byte/sign/saturating-pair operations and FMA:

```sh
c++ -std=c++17 -O2 -mavx2 -mfma -ffp-contract=off \
  tests/amd/fma_screen/fma_avx2_witness.cpp -o "$screen_dir/avx2-witness"
"$screen_dir/avx2-witness" > "$screen_dir/avx2-witness.log" 2>&1
```

Only run that optional CPU witness where AVX2/FMA are already supported.
`fma_cases.json` records the same exact input and expected-output constants as
the helper. The shader reads three runtime input words, evaluates legacy GLSL
`fma` and a `precise` separate multiply/add, and writes two words. It requests
no FloatControls2, explicit RTE execution mode, or shader_fma feature.

## Parent-dispatched run on one confirmed V340 die

Do not execute a GPU screen until the parent dispatches it. Inventory queries
create no logical device and dispatch nothing:

```sh
timeout --signal=TERM --kill-after=2s 15s \
  "$screen_dir/fma-screen" --list > "$screen_dir/inventory.log" 2>&1
```

Inspect `inventory.log`. Select exactly one physical V340 (`1002:6864`),
preferably the die at PCI BDF `0000:07:00.0`, using its printed 32-digit UUID.
Do not guess a UUID or reuse an old enumeration index. The screen refuses other
GPU identities, software devices, missing UUIDs, or duplicate UUID matches.

```sh
# Set this from the just-read inventory for the one confirmed die.
: "${FMA_DEVICE_UUID:?set the confirmed physical V340 device UUID}"
if timeout --signal=TERM --kill-after=2s 30s \
    "$screen_dir/fma-screen" --device-uuid "$FMA_DEVICE_UUID" \
    --shader "$screen_dir/fma_probe.spv" > "$screen_dir/gpu.log" 2>&1; then
  screen_rc=0
else
  screen_rc=$?
fi
printf '%s\n' "$screen_rc" > "$screen_dir/gpu.exit-code"
cat "$screen_dir/gpu.log"
printf 'screen_rc=%s artifacts=%s\n' "$screen_rc" "$screen_dir"
```

The screen creates one logical device and compute queue, one 20-byte storage
buffer with at most 1 MiB memory allocation, one pipeline, one descriptor set,
one command buffer and one fence. It performs exactly four `(1,1,1)` dispatches
with local size one. It requires coherent host-visible memory, waits at most
two seconds per fence, and never selects a fallback device. Pipeline/driver
internal allocations are outside the explicit buffer bound. The outer timeout
also bounds driver calls and pipeline compilation. A failed fence causes an
immediate process exit without destroying potentially in-flight Vulkan objects.

Output includes device UUID/PCI identity, driver strings and raw version,
FP32 float controls, optional FloatControls2 and shader_fma extension/feature
queries, resource bounds, all input/output hex words, and expected values.
If build headers lack an optional feature structure, output says
`query_unavailable_in_build_headers`; that is not evidence of device absence.

| Exit | Meaning |
| --- | --- |
| 0 | All four GPU cases match both fused and separate RNE oracles; inventory/host-check also use 0 on success. |
| 2 | At least one returned word differs. Preserve all rows; do not infer the cause from the exit code alone. |
| 3 | Fence timeout/error. Stop this screen. |
| 1 | Input, device, capability, resource or Vulkan error. |
| 124/137 | Outer timeout/kill. Treat as a failed or incomplete screen. |

A legacy result equal to the separate oracle on the three discriminating cases,
with correct controls, directly observes the suspected behavior. If legacy FMA
matches the fused oracle and the separate path matches its own oracle, continue
only toward the bounded projection prototype and exact one-real-expert stage
qualification. Any other result needs interpretation first. No model or trace
rerun follows automatically. Return the exact Git SHA, tool versions, hashes,
inventory, host-check log, GPU rows and exit code to the parent. Preserve old
builds, resident models and prior evidence. No independent Hermes dispatch.

## Validation performed for this PR

- Passed locally: shader compile with `glslc` 2026.4, Vulkan 1.2 SPIR-V validation,
  unchanged module SHA256
  `e8eaa3fda0cb484f5ed3d9b37d07b18c3af53254dee84f697a288a9fdbba92a2`.
- Passed locally: AVX2 witness build/run with GCC 16.2.1. Fused `3b401000`,
  separate `3b400000`. This CPU witness is not a GPU measurement.
- Blocked locally: wrapper compilation because `vulkan/vulkan.h` is absent.
  The installed Vulkan loader and `pkg-config` metadata do not supply SDK headers.
- Not run: wrapper host-check, device inventory, GPU screen or real-expert parity.
  No GPU execution was authorized on this executor.
