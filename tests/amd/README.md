# One Q8 expert on Vulkan

This standalone project provides an offline measurement and an opt-in in-process
executor experiment. It is not a listener or a model runner. It addresses
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

## Opt-in native Q8 boundary control

The default remains `mode=raw_f32`. Add **`--native-q8-boundaries`** for one
diagnostic-only **hybrid** control:

```text
host native_quant_act -> Q8_0 -> dequantized F32
  -> selected-backend Q8 gate and up matvecs
  -> host native-expression SwiGLU
  -> host native_quant_h -> Q8_0 -> dequantized F32
  -> selected-backend Q8 down matvec
```

When Vulkan is selected, all three weight matrices remain on Vulkan and all
three matvecs execute there. The host does activation quantization/dequantization
and the same F32 SwiGLU expression checked against the actual native reference.
It does not substitute CPU weight projections. This introduces a host boundary
and synchronization between gate/up and down; it is a correctness discriminator,
not an optimized worker or throughput claim.

The manifest labels `mode=hybrid_native_q8_boundaries`; the raw-F32 comparator
refuses this mode. Gate/up/down accumulation can still differ from native CPU
integer dots and reduction order. No CPU bit-parity or numerical acceptance is
promised for Vulkan. `selected.hidden.f32` is the host SwiGLU value before its
quantization, while `selected_down_input.f32` is what actually enters down.

In addition to the unchanged native-reference files, each case saves:

| File suffix | Meaning |
| --- | --- |
| `selected_input.q8_0` | Native quantization of original input; 2720 bytes |
| `selected_input.f32` | Dequantization uploaded for gate/up; 10240 bytes |
| `selected_hidden.q8_0` | Native quantization of selected-backend host hidden; 680 bytes |
| `selected_down_input.f32` | Dequantization uploaded for down; 2560 bytes |

Host checks exercise this control on CPU, verify its stage rows against the
actual native reference, packed/dequantized boundary bytes and cross-process
determinism. These host-only assertions do not impose a Vulkan parity gate.

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

For an existing configured **Unix Makefiles** build at the same source path,
after updating only this diagnostic's source/test files, use:

```bash
cmake --build "$AMD_BUILD" --target strata-vulkan-q8-expert/fast -- -j2
ctest --test-dir "$AMD_BUILD" -R '^amd_q8_expert_host$' \
  --output-on-failure --timeout 120
```

The Make `/fast` target compiles/links this executable without traversing the
existing dependency targets, including ggml's always-checked shader generator.
Use it only after the original complete build passed, with unchanged CMake,
dependency sources, compiler and configuration. Do not reconfigure, relocate the
source, edit ggml or delete its cache for this incremental control. Preserve the
old executable and choose new output/log paths. Add `--native-q8-boundaries` to
each die's invocation below; the original raw screen needs no repeat.

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


## Opt-in in-process adapter

`strata_vulkan_executor` is built only by this standalone CMake project. Call
`create_vulkan_gpu_executor(device)` explicitly and use the existing
`GpuExecutor::prepare -> infer_rows -> release` interface. No production target,
worker negotiation, listener, scheduler, transport or default selects it.

The adapter accepts one trusted, cache-verified plain Q8_0/Q8_0 expert at
H2560/FF640. Requests require T=1, one active entry in a bounded K-entry frame,
F32 activations and the prepared model/placement identity. The reply preserves
session, sequence and the original entry index. Its row is unweighted; the
routing weight is never applied. Changed plans, other shapes, multiple active
entries, packed input and weighted `infer` are rejected. Calls must be serial.
The caller opens/prepares/publishes the existing `GpuCache`; the adapter rereads
and rehashes the assigned blob before initializing a backend.

The shared graph retains the hybrid computation above. Runtime code contains no
CPU reference weight projections, evidence dumps or model access. The explicit
CPU substitution factory is compiled only with `STRATA_AMD_EXECUTOR_HOST_TEST`
in this standalone test project; Vulkan errors never select it.

The executor owns its backend and one tensor buffer, capped below 16 MiB, plus
bounded graph metadata and host activation storage. `gpu_allocated_bytes` counts
that tensor buffer; `gpu_weight_bytes` counts its 5,222,400 weight bytes and
`gpu_scratch_bytes` is the remaining tensor allocation/alignment. These counters
exclude ggml internal compute buffers, shader pipelines and driver allocations.
`gpu_allocation_count` is the live owned tensor-buffer count for this adapter;
release clears all four owned counters, retaining the historical peak. A nonzero
runtime VRAM reserve is explicitly unsupported. Use the existing external
process/VRAM guard; counters alone cannot establish total device memory use.

Add `--native-q8-boundaries --executor-check` to the diagnostic. It prepares the
same deterministic expert through the cache and adapter, submits the signal at
entry index 2 with weight 0.125, and requires its 10,240 returned row bytes to
match the direct hybrid diagnostic row on the same device/build. It then releases
the adapter and checks zero owned counters. `ADAPTER_ACCEPTED` and
`ADAPTER_COMPLETED` record that exact comparison; native CPU differences stay in
`metrics.tsv` with no tolerance gate. `COMPLETED` is still required for the whole
diagnostic. The raw/default invocation and prior artifacts remain unchanged.

`amd_vulkan_adapter_host` runs this through an explicit CPU graph substitution.
It checks before/after-release errors, missing cache, a valid unsupported-shape
plan, reserve rejection, malformed/batched/unassigned requests, a valid changed
plan, indexed unweighted output, repeated prepare and failed reprepare cleanup.
It never initializes Vulkan. See [EXECUTOR_HANDOFF.md](EXECUTOR_HANDOFF.md) for the
incremental Vulkan build and one-run acceptance request.

Local validation for this slice: Linux x86-64, GCC 16.2.1, Release, pinned ggml
source. Both host CTests passed. The substituted adapter returned index 2,
10,240 byte-identical row bytes, a 5,250,560-byte owned tensor buffer and zero
owned bytes/live allocations after release. All 34 raw and 42 hybrid evidence
files matched the retained PR #25 CPU executable byte for byte. Both changed
Vulkan-enabled translation units passed `-Wall -Wextra -Wpedantic` syntax checks
against the pinned API. Full Vulkan compilation and GPU execution are not run
here; host substitution is not AMD execution evidence. CUDA/HIP/SYCL code and
production build files are untouched.
