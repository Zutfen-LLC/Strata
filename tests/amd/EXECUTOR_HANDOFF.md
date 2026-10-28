# One adapter acceptance run per die

This is a requested handoff, not a record of GPU execution. Stack on
[PR #25](https://github.com/Zutfen-LLC/Strata/pull/25), base
`043922be16d8df66c61cb6da4ab1f903369c0745`; track
[issue #23](https://github.com/Zutfen-LLC/Strata/issues/23).

Reuse the previously successful standalone Unix Makefiles build, pinned ggml
source/libraries, installed dependencies and the same isolated source path.
Stage this PR's exact head into that source path before building. Preserve the
previous executable/evidence and record source head and the retained build flags.
No new dependencies, model files, shader changes or fleet/network configuration
are required. Stop if source/build identity or configuration differs unexpectedly.

CMake changed to add the adapter and existing CPU asset library. Reconfigure
once, preserving the original cache, then build the CPU dependency target and
only the two new/changed targets with `/fast`:

```bash
: "${STRATA_SRC:?existing isolated diagnostic source path, now at exact PR head}"
: "${AMD_BUILD:?previous successful standalone Unix Makefiles build}"
: "${ADAPTER_LOGS:?fresh owned build/log directory}"
mkdir "$ADAPTER_LOGS"
cp "$AMD_BUILD/strata-vulkan-q8-expert" "$ADAPTER_LOGS/probe.before-adapter"
timeout 120s cmake -S "$STRATA_SRC/tests/amd" -B "$AMD_BUILD" \
  > "$ADAPTER_LOGS/configure.log" 2>&1
timeout 300s cmake --build "$AMD_BUILD" --target strata_gpu_assets -j2 \
  > "$ADAPTER_LOGS/assets-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata_vulkan_executor/fast -- -j2 \
  > "$ADAPTER_LOGS/adapter-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata-vulkan-q8-expert/fast -- -j2 \
  > "$ADAPTER_LOGS/probe-build.log" 2>&1
ctest --test-dir "$AMD_BUILD" -R '^amd_(q8_expert|vulkan_adapter)_host$' \
  --output-on-failure --timeout 120 > "$ADAPTER_LOGS/host-tests.log" 2>&1
```

The asset target traverses only CPU dependencies. The `/fast` targets avoid
traversing the Vulkan dependency and shader generator. Do not run the ordinary
all-target build. Inspect logs before execution; stop on missing libraries,
shader rebuild requests or any failed command. These incremental steps were
checked with the local CPU build; the retained Vulkan build still needs checking.

Under the parent's existing owned-process resource guard (below 1 GiB), run one
confirmed physical die, logical device 0, with a fresh output path:

```bash
: "${VK_PHYS_DIE:?physical Vulkan index confirmed against retained PCI inventory}"
: "${ADAPTER_OUT:?new output directory that must not exist}"
timeout --signal=TERM --kill-after=5s 120s \
  env GGML_VK_VISIBLE_DEVICES="$VK_PHYS_DIE" \
  "$AMD_BUILD/strata-vulkan-q8-expert" --backend vulkan --device 0 \
  --native-q8-boundaries --executor-check --out-dir "$ADAPTER_OUT" \
  > "$ADAPTER_LOGS/die.log" 2>&1
```

Require exit 0, `ADAPTER_ACCEPTED backend=vulkan index=2 row_bytes=10240
bit_differences=0 ... released_bytes=0 live_allocations=0`, `ADAPTER_COMPLETED`
and the final `COMPLETED`. Verify `manifest.txt` identifies the intended Vulkan
device and hybrid mode, then inspect `metrics.tsv` and the external resource
receipt. Only after inspection, repeat once for the other confirmed die using
fresh log/output paths. No retry loop, model, performance sweep or raw-mode
rerun is requested.

Acceptance is exact adapter-to-direct-hybrid row bytes and owned-resource
release. Report native CPU bit differences/max absolute error/relative L2
separately; no tolerance acceptance or general model-quality claim follows.
Previous hybrid GPU evidence had 489/2560 output values differing from native
CPU bits despite relative L2 near 5.26e-8; that prior result is not evidence for
this adapter head. Preserve all failed/partial outputs and stop for diagnosis.
