# Portable PLE local validation — v1

Source is identified by `portable-ple-source.sha256` and the PR commit. Built
on Linux x86_64 with GCC 16.2.1, CMake 3.31.6, retained ggml source directory
`/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d`. Device execution
uses software Vulkan llvmpipe (LLVM 23.1.1, 256 bits), separate Vulkan buffers,
and strict scheduler placement. It is not an AMD hardware result.

## Passed

CPU-only configure and build used:

```sh
/tmp/cmake-3.31.6-linux-x86_64/bin/cmake -S . -B ${BUILD_ARTIFACT_FDE67CFC} \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_HIP_GFX906=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DSTRATA_GGML_DIR=/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d
```

The isolated Vulkan build `${BUILD_ARTIFACT_CC1A42EB}` used those settings
plus `-DGGML_VULKAN=ON`,
`-DCMAKE_PREFIX_PATH=${BUILD_ARTIFACT_D2E5D17A}`, and
`-DCMAKE_CXX_FLAGS=-I${BUILD_ARTIFACT_A3EFA7CB}`.
Built targets: `portable_ple_test portable_ple_vulkan_test portable_gdn_test
portable_gdn_vulkan_test portable_ffn_test portable_ffn_vulkan_test
gguf_reader_test gguf_split_test gguf_geometry_test`, with `--parallel 2`.

```sh
/tmp/cmake-3.31.6-linux-x86_64/bin/ctest \
  --test-dir ${BUILD_ARTIFACT_CC1A42EB} \
  -R '^(portable_(ple|gdn|ffn)_test|gguf_(reader|split|geometry)_test)$' \
  --output-on-failure
```

All six passed. CPU-only build ran the same six tests successfully as well.
The PLE check covers independent small-table hash/EOS/zero/image/uint64 wrap
vectors, Q8 decoding against pinned ggml, invalid/nonfinite row rejection,
actual-shape sparse mapped-table binding behind a nonzero shard offset,
model-lifetime independence, every required PLE metadata field missing or
corrupt, F16 conv/flat norm binding, ten tokens with scalar PLE/GDN oracles,
callback rejection/throw/nonfinite rollback, token retry, reset and QSA
rejection. Layer1 weight bytes: 121375616; CPU graph bytes: 7616544;
conservative host-state bound: 8519680. The logical table size is excluded
from those weight/graph allocations.

Device commands, one process at a time:

```sh
export GGML_VK_VISIBLE_DEVICES=0
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json
timeout 180s ${BUILD_ARTIFACT_DFC0618A} \
  --device 0 --expect-device llvmpipe
timeout 120s ${BUILD_ARTIFACT_789AAF19} \
  --device 0 --expect-device llvmpipe
timeout 120s ${BUILD_ARTIFACT_755BD796} \
  --device 0 --expect-device llvmpipe
```

The PLE result reports actual H2560 layer1 PLE+GDN+FFN,
`attention_MUL_MAT=10 GATED_DELTA_NET=1 SSM_CONV=1`, strict execution,
ten tokens, independent scalar PLE/GDN, CPU parity, EOS/reset/rollback.
The unchanged GDN and FFN executables also pass.

Opt-out configure in `${BUILD_ARTIFACT_67BFC229}` with
`STRATA_PORTABLE_FFN=OFF STRATA_PORTABLE_GDN=OFF STRATA_BUILD_TESTS=OFF`
has no portable libraries/test targets. Existing CUDA/HIP/SYCL sources and
prior portable regression sources have no diff. This is source/target gating
evidence, not a complete default-engine binary comparison.

## Resolved development failures

The first unquantized scalar projection oracle differed from CPU activation
Q8 rounding. Fixture blocks now have exactly representable Q8 endpoints;
tolerances were preserved. Vulkan initially lacked an explicit installed
SPIR-V include path. Default Vulkan enumeration excludes software CPU devices,
so llvmpipe was explicitly selected. PLE exposed a real placement-reporting
index error after Vulkan graph reordering; tensor-identity reporting fixes it
without weakening strict placement or disabling optimization. No outstanding
failure remains in the checks above.

## Not run

Physical V340L checks, actual fleet GGUF payload binding/execution, real expert
callbacks, full-model output/quality, CUDA/HIP/SYCL builds and performance.
No fleet operations were attempted. Parent-dispatched commands are in
`portable-ple-layer1.md`. The implementation rejects unsupported QSA/layers
and table encodings rather than changing numerical behavior.
