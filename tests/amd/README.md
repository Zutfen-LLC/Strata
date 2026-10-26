# One Q8 expert on Vulkan

This is an offline measurement, not a worker or a model runner. It addresses
[issue #23](https://github.com/Zutfen-LLC/Strata/issues/23). It has its own CMake entry
point; no production build file, CUDA/HIP kernel, default or launch path changes.

The fixture is one plain Q8_0/Q8_0 expert at H=2560, FF=640, T=1. Gate, up and down
weights total 5,222,400 bytes. A fixed unsigned integer generator writes legal Q8_0
blocks directly, with exactly representable scale 2^-9 and codes in [-15,15]. The
nonzero input is fixed dyadic F32; a second input is all zero. Neither uses a model,
random seed from the OS, network, existing expert arena or model cache.

## What is compared

The reference calls Strata's actual `native_quant_act`, `native_gu_rows`,
`native_quant_h` and `native_down_rows` from `strata_kernels_cpu`. Its activations
are Q8_0. The gate/up diagnostic dots use the same ggml CPU type trait; their
recomposed hidden must bitmatch the production reference's hidden.

A separate ggml CPU graph and, when requested, a direct Vulkan graph compute:

```text
Q8_0 gate × F32 input ─→ SiLU ─┐
                              × ─→ Q8_0 down projection ─→ F32 output
Q8_0 up × F32 input ──────────┘
```

The graph backend owns activation conversion and math. In particular, Vulkan's
internal quantized matvec activation path is not declared equivalent to Strata's
native Q8_0 activation path. Saved Vulkan hidden values are F32 graph values, not
a dump of its internal packed down-projection activation. There is no scheduler
and no CPU fallback: a declined operation or backend failure stops the run.

`metrics.tsv` reports gate, up, hidden and output comparisons to both the native
reference and the ggml CPU graph. It records the number of values whose raw bits
differ, maximum absolute error and relative L2. There is **no cross-backend
tolerance pass**. `COMPLETED` means finite computation, exact same-backend repeats,
the zero control and evidence writing finished. It does not mean numerical
equivalence was accepted. The metrics and raw rows must be read before deciding
the next step. Equal final rows on these synthetic inputs would not qualify real
model rows or model-level logits/tokens.

Every backend is explicit. `--backend vulkan --device N` never selects CPU after
an error; `--backend cpu` forbids a device number. The selected backend name,
device description, fixture identities and tensor-buffer size are recorded.

## Build and check

Use pinned llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Reuse retained
source, with a fresh build directory outside it. Vulkan requires installed
Vulkan C/C++ headers/libraries, `glslc` and the SPIRV-Headers CMake package. These
commands install nothing and disable automatic dependency fetching.

```bash
: "${STRATA_SRC:?isolated source containing this diagnostic}"
: "${GGML_SRC:?retained pinned llama.cpp source}"
: "${AMD_BUILD:?new absolute build directory}"
cmake -S "$STRATA_SRC/tests/amd" -B "$AMD_BUILD" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_GGML_DIR="$GGML_SRC" \
  -DSTRATA_AMD_PROBE_VULKAN=ON
cmake --build "$AMD_BUILD" --target strata-vulkan-q8-expert -j2
ctest --test-dir "$AMD_BUILD" -R '^amd_q8_expert_host$' \
  --output-on-failure --timeout 120
```

The host test never initializes Vulkan. It runs the actual native reference and
ggml CPU graph twice, verifies fixed fixture hashes and byte-identical evidence
across processes, finite stage metrics and zero controls, and checks invalid
backend/device options and refusal to overwrite an existing output directory.
For a CPU-only development machine use `-DSTRATA_AMD_PROBE_VULKAN=OFF`; that build
refuses Vulkan execution. This checks the fixture, not a GPU.

## Two dies, sequential execution

First bind physical Vulkan indices to the actual PCI BDFs from the retained
inventory. `GGML_VK_VISIBLE_DEVICES` makes each selected physical device logical
device 0. Do not assume physical indices from GPU names alone. Use the existing
owned-process resource guard and fresh output paths; run the second die only
after inspecting the first run's completion, metrics and resource receipt.

```bash
: "${AMD_OUT_DIE0:?new output directory, must not exist}"
: "${VK_PHYS_DIE0:?confirmed physical Vulkan device index}"
timeout --signal=TERM --kill-after=5s 120s \
  env GGML_VK_VISIBLE_DEVICES="$VK_PHYS_DIE0" \
  "$AMD_BUILD/strata-vulkan-q8-expert" --backend vulkan --device 0 \
  --out-dir "$AMD_OUT_DIE0"
```

Then use a new output directory and the other confirmed index for die 1. Each
invocation computes signal and zero cases with exact same-backend repeat checks;
there is no automatic retry, performance sweep or model inference.

The explicit weight payload is about 5 MiB; each graph's tensor allocation is
refused above 16 MiB. The native blob, CPU graph and Vulkan graph use separate
bounded storage. This is not a 16 MiB total-process or driver allocation limit:
shader pipelines, runtime buffers and allocator effects must be measured on the
GPU. The prior tiny matvec screen's RSS is not a VRAM estimate for this fixture.
Keep the parent run's process/VRAM budget below 1 GiB and stop on its guard.

Output is about 5.2 MB of weights plus small input, native packed-activation and
three sets of raw stage rows for both cases. Existing output paths are refused,
including after a failed run. Partial failure evidence is preserved without
`COMPLETED`; choose a new directory only after diagnosing the failure.

## Validation so far

Local Linux x86-64 CPU build (Intel Core Ultra 9 185H, GCC 16.2.1, Release,
ggml native CPU build) against the pinned ggml source passed. The focused host
CTest passed. For the fixed signal input, ggml CPU graph gate/up and final output
matched the native reference bitwise on the tested build; 159 of 640 hidden values
differed, max absolute error 1.4901161193847656e-8, relative L2
4.4293903655581662e-8. Quantization erased those hidden differences in this sample.
This is fixture/build-specific evidence, not a universal guarantee.

The Vulkan-enabled diagnostic translation unit compiled against the pinned API
with `-Wall -Wextra -Wpedantic`. Full Vulkan shader/backend compilation and this
expert's GPU execution are **not run here**. CUDA/HIP builds are not required for
these isolated diagnostic-only additions; their source/build paths are unchanged.
