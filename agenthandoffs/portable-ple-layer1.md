# Portable configured PLE layer 1 validation — v1

Stacked on PR39 `a06902e452257824e454e1d00e00928161a3ef79`.
Parent owns dispatch and physical die mapping. Do not run fleet commands until
parent dispatch. Preserve accepted PR38/PR39 builds and evidence. Use a fresh
candidate checkout, build and evidence directory. No SSH/access changes, model
scan, workers, external engine, downloads or deployment are needed.

The product entry is `PortableGdnLayer::from_gguf(model, 1, options, error)` and
`step(token_id, residual, dispatch, route, error)`. It binds the configured
Q8_0 `[160,320001536]` table from whichever shard contains it and retains a
read-only host mapping independently of the caller's model lifetime. The table
never becomes a ggml tensor. Sixteen rows (2720 encoded bytes, 2560 decoded
floats) are gathered for a token. Only the embedding and bounded activations
enter the graph. Layer0's token-less entry remains supported; layer1 requires
a token. For image positions pass the configured image ID 248056 explicitly.
One object is one ordered sequence; calls are serialized by its owner.

PLE runs before HC/GDN attention and FFN, matching retained upstream
`src/models/qwen4exp.cpp::{llm_graph_input_ple::set_input,graph::build_ple}`
at `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Host hashing and Q8 row
encoding reuse Strata's existing implementation. Grouped RMS norms, the signed
square-root sigmoid gate and dilation-three normalized convolution history
match that source. The current EOS retains its predecessors; an EOS predecessor
cuts itself and older hash context. It does not erase convolution/GDN state.
`reset()` explicitly starts a new sequence and clears all histories/state.

Token IDs, PLE normalized history, GDN convolution/state and residual commit
together after successful FFN/expert dispatch. Failure, throw and nonfinite
expert output leave the last committed public state unchanged. QSA and layers
other than 0/1 are rejected. New code is behind existing opt-in
`STRATA_PORTABLE_GDN`; CUDA/HIP/SYCL source and defaults are unchanged.

The PLE graph exposed Vulkan reordering of graph nodes during optimization.
Placement records now follow original tensor identity through allocation,
preventing an alias's backend from being attributed to a computation after
reordering. Strict execution still checks actual scheduler assignments.

## Parent-dispatched synthetic device checks

Use retained ggml and already-installed compiler/header-prefix settings from
the accepted build. Set `CMAKE_PREFIX_PATH` and `CMAKE_CXX_FLAGS` to those same
known settings if SPIR-V headers require them. Do not change tolerances or
backend settings after failure.

```sh
set -eu
: "${PLE_SRC05:?new candidate source checkout}"
: "${PLE_BUILD05:?new isolated build directory}"
: "${EXPECTED_PLE_HEAD:?exact candidate PR head}"
: "${PLE_EVIDENCE05:?new evidence directory}"
: "${DIE0:?verified ggml device index for first physical die}"
: "${DIE1:?verified ggml device index for second physical die}"
GGML_SRC05=/home/hermes/vulkan-q8-screen-20261026/src
test "$(git -C "$PLE_SRC05" rev-parse HEAD)" = "$EXPECTED_PLE_HEAD"
(cd "$PLE_SRC05" && sha256sum -c agenthandoffs/portable-ple-source.sha256)
test "$(git -C "$GGML_SRC05" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test -z "$(git -C "$GGML_SRC05" diff --name-only HEAD -- ggml)"
cmake -S "$PLE_SRC05" -B "$PLE_BUILD05" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_HIP_GFX906=OFF -DSTRATA_ENABLE_SYCL=OFF -DGGML_VULKAN=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DSTRATA_GGML_DIR="$GGML_SRC05"
cmake --build "$PLE_BUILD05" --parallel 2 --target \
  portable_ple_test portable_ple_vulkan_test portable_gdn_test \
  portable_gdn_vulkan_test portable_ffn_test portable_ffn_vulkan_test \
  gguf_reader_test gguf_split_test gguf_geometry_test
mkdir "$PLE_EVIDENCE05"
ctest --test-dir "$PLE_BUILD05" \
  -R '^(portable_(ple|gdn|ffn)_test|gguf_(reader|split|geometry)_test)$' \
  --output-on-failure > "$PLE_EVIDENCE05/cpu.log" 2>&1
test "$DIE0" != "$DIE1"
for die in "$DIE0" "$DIE1"; do
  timeout 180s "$PLE_BUILD05/portable_ple_vulkan_test" \
    --device "$die" --expect-device V340 > "$PLE_EVIDENCE05/ple-die-$die.log" 2>&1
  timeout 120s "$PLE_BUILD05/portable_gdn_vulkan_test" \
    --device "$die" --expect-device V340 > "$PLE_EVIDENCE05/gdn-die-$die.log" 2>&1
  timeout 120s "$PLE_BUILD05/portable_ffn_vulkan_test" \
    --device "$die" --expect-device V340 > "$PLE_EVIDENCE05/ffn-die-$die.log" 2>&1
done
cat "$PLE_EVIDENCE05"/*.log
```

Require all commands to exit 0. Return full logs, exact source head, retained
private physical/logical die mapping and actual device descriptions. PLE logs
must report `attention_MUL_MAT=10 GATED_DELTA_NET=1 SSM_CONV=1`, strict
execution, ten tokens, independent scalar PLE/GDN checks, CPU parity and
EOS/reset/rollback. Tests use actual layer geometry and bounded synthetic
weights. The CPU PLE test also creates an actual-shape sparse table, touching
only its header, layer weights and sixteen selected rows; the approximately
54 GB logical file does not allocate/fill/scan the table. Ensure the temporary
filesystem supports sparse files and at least 256 MB free space.
On failure or timeout stop, retain evidence and return the exact error.

Limits: synthetic row values and expert callback, no fleet real-weight binding
or model-quality claim, no full-model embedding/head/loop, QSA, simultaneous
dual-die inference or performance measurement. The next slice is QSA and then
the complete local model loop with existing expert dispatch.
