> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# PR #38: bounded Vulkan placement validation on 05

Parent dispatch only, **after the additional CPU SMT arm is stopped and cleaned
up**. Do not overlap the CPU trial. No service/worker launch, listener, model
payload, clock, driver, credential or access change is required here.

This validates the product FFN building block, not full-local model inference.
It uses synthetic hidden=32 / shared-FF=32 tensors, four HC streams and two
selected experts. The projection encoding mix matches the fleet (six Q8_0
projections with F32 norm/router/shared gate); production 2560/640 dimensions,
GDN/QSA/PLE, full-model logits and dual-die concurrent execution remain untested.

## Source and isolated build

Use a candidate source checkout of the parent-selected PR #38 head and a **new**
build directory. Do not reconfigure the retained AMD diagnostic or live worker
build. Source-file hashes in `agenthandoffs/portable-ffn-source.sha256` support a
copied snapshot without Git metadata; bind that manifest to the trusted parent
receipt rather than inventing provenance.

```sh
set -eu
: "${PORTABLE_SRC05:?candidate PR38 source supplied by parent}"
: "${PORTABLE_BUILD05:?new isolated candidate build directory}"
: "${EXPECTED_HEAD:?exact PR38 commit selected by parent}"
: "${EVIDENCE05:?new evidence directory supplied by parent}"
GGML_SRC05=${PRIVATE_ARTIFACT_9CA76AB8}

if test -e "$PORTABLE_SRC05/.git"; then
  test "$(git -C "$PORTABLE_SRC05" rev-parse HEAD)" = "$EXPECTED_HEAD"
else
  echo 'Copied source: using parent-bound source manifest; Git metadata absent'
fi
(cd "$PORTABLE_SRC05" && sha256sum -c agenthandoffs/portable-ffn-source.sha256)
test "$(git -C "$GGML_SRC05" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test -z "$(git -C "$GGML_SRC05" diff --name-only HEAD -- ggml)"

cmake -S "$PORTABLE_SRC05" -B "$PORTABLE_BUILD05" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE_FFN=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DGGML_VULKAN=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DSTRATA_GGML_DIR="$GGML_SRC05"
cmake --build "$PORTABLE_BUILD05" --parallel 2 \
  --target portable_ffn_test portable_ffn_vulkan_test
ctest --test-dir "$PORTABLE_BUILD05" -R '^portable_ffn_test$' --output-on-failure
```

Require an empty scoped ggml diff above. Reuse retained compiler/header-prefix
settings if they were needed by the accepted AMD build. If configuration cannot
find an existing Vulkan/glslc/SPIRV-Headers component, return that exact error;
do not install/download dependencies or modify the old build. No broad backend
suite is needed.

## One tiny check per physical die, sequentially

Use the retained private UUID/device mapping, reconfirmed in current read-only
enumeration. Bind the **ggml logical** indices from the current enumeration to
those two physical dies; do not assume a Vulkan loader index is the same. The
previous logical indices were 0 and 1. Do not apply the local llvmpipe
ICD/visibility environment to 05.

```sh
: "${DIE0:?verified ggml logical index for first physical V340 die}"
: "${DIE1:?verified ggml logical index for second physical V340 die}"
test "$DIE0" != "$DIE1"
mkdir "$EVIDENCE05"
timeout 60s "$PORTABLE_BUILD05/portable_ffn_vulkan_test" \
  --device "$DIE0" --expect-device V340 > "$EVIDENCE05/die0.log" 2>&1
timeout 60s "$PORTABLE_BUILD05/portable_ffn_vulkan_test" \
  --device "$DIE1" --expect-device V340 > "$EVIDENCE05/die1.log" 2>&1
cat "$EVIDENCE05/die0.log" "$EVIDENCE05/die1.log"
```

A missing device or unexpected description is a failure, never a skipped pass.
Require each command to exit 0 and print its correct `VulkanN` backend,
`actual_MUL_MAT=8`, `actual_HC_post=2`, `Q8_mix_MUL_MAT=8`, `fallback=0`, plus
execution/parity/strict/reset/rollback/budget success. The test also asserts that
the Vulkan backend rejects ggml CPU buffer types, so a CPU-like surrogate cannot
pass the device-placement assertion. Metadata-only views can alias host storage;
their arithmetic consumers must execute on the selected Vulkan backend.

Return both complete logs, logical-index/physical-UUID mapping, exact source head,
source-manifest check, ggml identity and build result. A nonzero exit/timeout
stops the sequence; preserve the failing log and do not try a larger case or
change settings. No speed comparison or V340L full-model qualification follows
from these tiny checks. The next product work remains real expert dispatch,
persistent attention/PLE state, embedding/head and the full-layer loop.
