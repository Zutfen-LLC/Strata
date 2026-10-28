> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Parent-dispatched actual-dimension GDN layer-0 check

This is a new slice stacked on PR #38 at `11a5a3f`. Preserve the accepted PR #38
source/build/evidence snapshot and its historical six-file manifest. Do not
reconfigure the accepted old AMD expert-worker build or reuse its output paths.
Run only after parent dispatch confirms previous tests are cleaned up.

Use the new PR's exact source head and a new isolated build. Existing ggml stays
at `${PRIVATE_ARTIFACT_9CA76AB8}`, commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Reuse known compiler/header-prefix
settings discovered from the accepted PR #38 candidate build if necessary.
Do not install/download anything, alter access/clocks/drivers, start workers or
read the fleet model. The test creates actual-size synthetic weights locally.

```sh
set -eu
: "${GDN_SRC05:?new candidate checkout supplied by parent}"
: "${GDN_BUILD05:?new isolated build directory}"
: "${EXPECTED_GDN_HEAD:?exact new PR head supplied by parent}"
: "${GDN_EVIDENCE05:?new evidence directory path}"
: "${DIE0:?verified ggml index of first V340 die}"
: "${DIE1:?verified ggml index of second V340 die}"
GGML_SRC05=${PRIVATE_ARTIFACT_9CA76AB8}
if test -e "$GDN_SRC05/.git"; then
  test "$(git -C "$GDN_SRC05" rev-parse HEAD)" = "$EXPECTED_GDN_HEAD"
fi
(cd "$GDN_SRC05" && sha256sum -c agenthandoffs/portable-gdn-source.sha256)
test "$(git -C "$GGML_SRC05" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test -z "$(git -C "$GGML_SRC05" diff --name-only HEAD -- ggml)"
cmake -S "$GDN_SRC05" -B "$GDN_BUILD05" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_HIP_GFX906=OFF -DSTRATA_ENABLE_SYCL=OFF -DGGML_VULKAN=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DSTRATA_GGML_DIR="$GGML_SRC05"
cmake --build "$GDN_BUILD05" --parallel 2 \
  --target portable_gdn_test portable_gdn_vulkan_test portable_ffn_test
ctest --test-dir "$GDN_BUILD05" -R '^portable_(gdn|ffn)_test$' --output-on-failure
test "$DIE0" != "$DIE1"
mkdir "$GDN_EVIDENCE05"
timeout 120s "$GDN_BUILD05/portable_gdn_vulkan_test" \
  --device "$DIE0" --expect-device V340 > "$GDN_EVIDENCE05/die0.log" 2>&1
timeout 120s "$GDN_BUILD05/portable_gdn_vulkan_test" \
  --device "$DIE1" --expect-device V340 > "$GDN_EVIDENCE05/die1.log" 2>&1
cat "$GDN_EVIDENCE05/die0.log" "$GDN_EVIDENCE05/die1.log"
```

Require manifest/ggml checks, build, both CPU CTests and both Vulkan commands to
exit 0. Verify actual descriptions and the retained private logical/physical die
mapping. Each device log must report `GATED_DELTA_NET=1 SSM_CONV=1
attention_MUL_MAT=8`, strict execution, independent state, complete-layer CPU
parity, reset and rollback success. Return the full logs, exact head and mapping.
On failure/timeout stop, retain evidence and return the exact error; do not retry
with changed tolerances/settings or escalate to full-model inference.

These are sequential, one-device-at-a-time, synthetic actual-dimension checks.
They do not establish real-weight model output, real expert dispatch, PLE/QSA,
simultaneous dual-die inference or performance. No fleet process should remain
after the bounded test executables exit.
