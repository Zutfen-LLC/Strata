# Parent-dispatched portable QSA layer 3 check — v1

Candidate is stacked on PR40 head `c3625a290066a50e397a32cdab8040038acc0b62`
(`codex/portable-ple-layer`). Use the new PR's exact full head and manifest,
not the earlier implementation checkpoint. Parent/Hermes owns physical testing.
No physical QSA result is claimed here.

The bounded test creates actual H2560 Q8_0/F32 weights, runs the complete
HC/QSA/FFN layer at text positions 0–8 through the existing expert callback,
and checks an independent scalar oracle and CPU parity. It also runs focused
cached-cell fixtures 2/2050/2051/2052/2055/2056 without thousands of projections.
At 2052/2055/2056, actual TOP_K width is 2051, followed by deterministic
ascending-ID tie repair, full-cache masked attention and all 24 heads.

Use a fresh isolated source/build/evidence set. Preserve prior accepted builds,
services, resident models and device mappings. Reuse installed compilers,
Vulkan headers and retained pinned ggml; no installs/downloads, access changes,
workers or real-model workloads. Stop on failure/timeout and retain the logs.
Do not substitute CPU fallback or alter tolerances to obtain a pass.

```sh
set -eu
: "${QSA_SRC05:?fresh candidate source supplied by parent}"
: "${QSA_BUILD05:?fresh isolated build path}"
: "${QSA_EVIDENCE05:?fresh evidence directory path}"
: "${EXPECTED_QSA_HEAD:?full candidate PR head supplied by parent}"
: "${GGML_SRC05:?retained pinned llama.cpp checkout}"
: "${DIE0:?verified first physical die ggml index}"
: "${DIE1:?verified second physical die ggml index}"
: "${QSA_CMAKE05:?existing cmake executable, version 3.24 or newer}"
test "$DIE0" != "$DIE1"
test ! -e "$QSA_BUILD05"
test ! -e "$QSA_EVIDENCE05"
test "$(git -C "$QSA_SRC05" rev-parse HEAD)" = "$EXPECTED_QSA_HEAD"
test -z "$(git -C "$QSA_SRC05" status --porcelain --untracked-files=no)"
(cd "$QSA_SRC05" && sha256sum -c agenthandoffs/portable-qsa-source-v1.sha256)
test "$(git -C "$GGML_SRC05" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test -z "$(git -C "$GGML_SRC05" diff --name-only HEAD -- ggml)"
mkdir "$QSA_EVIDENCE05"
# Add only the previously verified installed compiler/SPIR-V include settings
# if this host needs them. Record those exact flags in the returned evidence.
"$QSA_CMAKE05" -S "$QSA_SRC05" -B "$QSA_BUILD05" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE_FFN=ON \
  -DSTRATA_PORTABLE_GDN=ON -DSTRATA_PORTABLE_QSA=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DGGML_VULKAN=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DSTRATA_GGML_DIR="$GGML_SRC05" \
  > "$QSA_EVIDENCE05/configure.log" 2>&1
"$QSA_CMAKE05" --build "$QSA_BUILD05" --parallel 2 \
  --target portable_qsa_test portable_qsa_gguf_test portable_qsa_vulkan_test \
  portable_ffn_test portable_gdn_test portable_ple_test \
  gguf_reader_test gguf_split_test gguf_geometry_test \
  > "$QSA_EVIDENCE05/build.log" 2>&1
# Use the ctest installed alongside the selected cmake.
"$(dirname "$QSA_CMAKE05")/ctest" --test-dir "$QSA_BUILD05" \
  -R '^(portable_(qsa|qsa_gguf|ffn|gdn|ple)_test|gguf_(reader|split|geometry)_test)$' \
  --output-on-failure > "$QSA_EVIDENCE05/cpu.log" 2>&1
timeout 180s "$QSA_BUILD05/portable_qsa_vulkan_test" \
  --device "$DIE0" --expect-device V340 > "$QSA_EVIDENCE05/die0.log" 2>&1
timeout 180s "$QSA_BUILD05/portable_qsa_vulkan_test" \
  --device "$DIE1" --expect-device V340 > "$QSA_EVIDENCE05/die1.log" 2>&1
```

If installed cmake/ctest are found through PATH rather than absolute paths,
set `QSA_CMAKE05` to the absolute result of `command -v cmake`. Preserve any
existing verified Vulkan visibility mapping; the two indexes above refer to
that mapping. Return exact commit, commands/flags, full logs, exit statuses,
and verified physical die identities. Both device logs must report strict
selection width2051, the five large cached-cell cases, and complete layer
`projection_MUL_MAT=8 ROPE=3` with scalar/CPU parity/rollback/reset success.
The fixture also reports `MUL_MAT=48` for the two explicit products per query
head. No scheduled tensor computation may have a CPU fallback. The existing
host expert callback/route merge and host transactional cache staging are
intentional coordinator work, not hidden tensor-operation fallback.

The cached-cell fixture explicitly admits a 1 GiB conservative graph bound;
observed llvmpipe scheduler allocation at 2056 cells was 19,030,176 bytes.
The whole-layer nine-token test uses the normal 256 MiB graph limit, including
the new workspace allowances. Physical shapes/pipelines must be measured.
Long contexts can require a larger graph limit due to conservative per-view
copy admission; exhausting that limit rejects the step transactionally.

These tests do not establish real-weight execution, real expert dispatch,
full-model quality, simultaneous dual-die inference or performance. K/V are
retained in FP16 and converted to F32 in the attention graph. This is a
correctness-oriented per-head lowering, not an optimization claim. Full-model
embedding/head and the all-layer loop remain the next integration step.
