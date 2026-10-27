# Portable QSA local results — v1

Source: the PR head and `portable-qsa-source-v1.sha256`. Linux x86_64,
GCC 16.2.1, CMake 3.31.6, retained upstream ggml at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Vulkan device: llvmpipe
(LLVM 23.1.1, 256 bits), explicitly selected. These are software-Vulkan results.

## Passed

Two isolated builds used:

```sh
/tmp/cmake-3.31.6-linux-x86_64/bin/cmake -S . -B /tmp/strata-portable-qsa-build \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE_FFN=ON \
  -DSTRATA_PORTABLE_GDN=ON -DSTRATA_PORTABLE_QSA=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DSTRATA_GGML_DIR=/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d
```

The second, `/tmp/strata-portable-qsa-vulkan`, adds `-DGGML_VULKAN=ON`,
`-DCMAKE_PREFIX_PATH=/tmp/strata-spirv-headers-prefix` and
`-DCMAKE_CXX_FLAGS=-I/tmp/strata-spirv-headers-prefix/include`. No installs.
Built the three QSA targets, existing FFN/GDN/PLE CPU and Vulkan tests, and
GGUF reader/split/geometry tests, with `--parallel 2`.

Both builds passed these eight CTests:

```sh
/tmp/cmake-3.31.6-linux-x86_64/bin/ctest --test-dir BUILD \
  -R '^(portable_(qsa|qsa_gguf|ffn|gdn|ple)_test|gguf_(reader|split|geometry)_test)$' \
  --output-on-failure
```

Strict Vulkan commands ran sequentially, one executable at a time:

```sh
export GGML_VK_VISIBLE_DEVICES=0
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json
timeout 180s /tmp/strata-portable-qsa-vulkan/portable_qsa_vulkan_test \
  --device 0 --expect-device llvmpipe
timeout 180s /tmp/strata-portable-qsa-vulkan/portable_ple_vulkan_test \
  --device 0 --expect-device llvmpipe
timeout 120s /tmp/strata-portable-qsa-vulkan/portable_gdn_vulkan_test \
  --device 0 --expect-device llvmpipe
timeout 120s /tmp/strata-portable-qsa-vulkan/portable_ffn_vulkan_test \
  --device 0 --expect-device llvmpipe
```

QSA passes positions 0–8, including per-head query/gate splitting, weighted
RMS/IMRoPE64, raw-key pooling before norm/RoPE at block-first position,
ReLU-per-head scores, finite tail bias, FP16 full KV, grouped attention,
sigmoid gate, output projection, HC combine and the existing FFN callback.
Scalar projection checks allow activation-quantization rounding; pooling,
cache, scores and attention are checked independently over projected inputs.

The two-cell regression and cached-cell fixtures 2050/2051/2052/2055/2056 pass
selection and all 24 attention outputs. Every tested tensor operation is
actually assigned to Vulkan with no fallback. At 2052/2055/2056, `TOP_K=1`,
width2051 and exact ascending-ID tie resolution are checked. Each cache fixture
has 48 attention matrix products. The explicit 1 GiB admission limit bounds
conservative per-view copies; actual scheduler allocation at 2056 cells is
19,030,176 bytes. It is not a 1 GiB physical allocation.

Whole-layer Vulkan result: `projection_MUL_MAT=8 ROPE=3`, weights 79,235,072
bytes; peak aggregate graph allocations plus admitted CPU/Vulkan workspace
allowances 190,338,372 bytes; conservative host-state bound 34,973,984 bytes
at capacity9. Default graph limit is 256 MiB; host-state limit is 64 MiB.
K/V and pooled history copies stage only active cells. Long contexts may
need a higher graph admission limit. Driver pipelines, command pools and
other uses of the borrowed backend remain caller-owned, outside layer limits.

CPU tests cover callback rejection/throw/malformed/nonfinite rows at block
completion and following token, exact uninterrupted-control retry, reset,
capacity exhaustion, nonfinite/malformed inputs, missing callback, FP16 cache
overflow, weight/graph/state budgets, strict unavailable-backend rejection
and explicit fallback reporting. GGUF tests bind actual-shaped Q8_0/F32
payloads, close the model and run, check required metadata and types, and
reject unsupported RoPE/scaling aliases/extensions and PLE at layer3.

Default-OFF configure `/tmp/strata-portable-qsa-default` succeeds and its target
list contains no portable libraries/tests. Existing FFN/GDN/PLE sources,
shared scheduler, CUDA/HIP/SYCL/Metal code have no diff against PR40. This is
source/target gating evidence, not a default-engine binary comparison.

## Resolved development failures

The initial batched attention product `[256,cells,2] x [256,12,2]` returned
incorrect llvmpipe logits for query heads starting at column8 with two cached
cells. At token1, scalar logits near 6.26/8.56 became near-zero for head8;
softmax and attention consequently differed. Explicit FP32 accumulation and
F32 key conversion alone did not fix that batched shape. The final lowering
uses explicit per-query-head products, F32 arithmetic and `head/12`; the
original full-layer case and a strong two-cell fixture now pass. No backend
or production CUDA source was changed, no fusion was globally disabled, and
no CPU fallback replaced this case.

An early placement assertion incorrectly included the documented host
`ROUTE_AND_MERGE` callback boundary; it now checks scheduled tensor work.
The larger cached-attention fixture initially exceeded the default
conservative graph admission bound; it now explicitly declares 1 GiB while
retaining allocation checks. An attempted test treating actual allocated
bytes as the minimum admission bound was discarded: conservative admission
can exceed allocations. Small-budget rejection remains tested. A default
configure initially omitted retained ggml; supplying the existing source
path fixed it. No outstanding test failure remains.

## Not run

Physical V340 QSA, fleet GGUF weights, real expert callbacks, full-model
assembly/output/quality, CUDA/HIP/SYCL builds, simultaneous dual-die operation
and performance. No fleet operations were attempted. Keep the PR in draft
pending independent review and parent-dispatched physical checks. Exact
bounded physical commands are in `portable-qsa-layer3-v1.md`.
