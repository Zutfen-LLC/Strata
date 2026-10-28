# Budgeted resident experts: one acceptance run per die

This follow-on experiment stacks on [PR #28](https://github.com/Zutfen-LLC/Strata/pull/28),
base `4e2d0450fe7aa870ae25a709f850740bdf21d209`, and tracks
[issue #23](https://github.com/Zutfen-LLC/Strata/issues/23). These commands are a
handoff, not hardware execution evidence for the follow-on head.

Reuse the successful PR #28 isolated source/build paths, pinned ggml
`3cf03257f219afbe7334045ff7c6a06ac68c627d` dependencies and compiler configuration.
Stage the exact follow-on PR head into the same isolated source path and record
that identity before building. This slice changes no CMake/dependency files or
shaders. Preserve the old binary and all prior evidence. No downloads, installs,
new credentials or host/security/network changes are required.

```bash
: "${STRATA_SRC:?same isolated diagnostic source, staged at exact follow-on head}"
: "${AMD_BUILD:?successful PR28 standalone Unix Makefiles build}"
: "${RESIDENT_LOGS:?fresh owned log directory}"
mkdir "$RESIDENT_LOGS"
cp "$AMD_BUILD/strata-vulkan-q8-expert" "$RESIDENT_LOGS/probe.before-residents"
timeout 120s cmake --build "$AMD_BUILD" --target strata_vulkan_executor/fast -- -j2 \
  > "$RESIDENT_LOGS/adapter-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata-vulkan-q8-expert/fast -- -j2 \
  > "$RESIDENT_LOGS/probe-build.log" 2>&1
ctest --test-dir "$AMD_BUILD" -R '^amd_(q8_expert|vulkan_adapter)_host$' \
  --output-on-failure --timeout 120 > "$RESIDENT_LOGS/host-tests.log" 2>&1
```

Both changed `.cpp` files rebuild; `/fast` skips dependency traversal and the
Vulkan shader generator. No reconfigure or CPU dependency build is needed when
the retained PR #28 build/dependencies are intact. Stop on build/test failure,
missing libraries or shader rebuild requests. The incremental sequence is
locally checked with the CPU configuration; Vulkan compilation remains pending.

After the parent admits the bounded run, select one physical die using its
retained PCI/device inventory. Logical device 0 below refers only to that die.
Use fresh paths and retain the external resource observations actually available;
the resident tensor budget is not a total-VRAM/process-memory bound. Previous
adapter runs had no external guard receipt or peak-VRAM measurement; do not
invent or carry forward one.

```bash
: "${VK_PHYS_DIE:?confirmed physical Vulkan index for the selected die}"
: "${RESIDENT_OUT:?new output directory that must not exist}"
: "${RESIDENT_DIE_LOG:?new log path for this die}"
timeout --signal=TERM --kill-after=5s 120s \
  env GGML_VK_VISIBLE_DEVICES="$VK_PHYS_DIE" \
  "$AMD_BUILD/strata-vulkan-q8-expert" --backend vulkan --device 0 \
  --native-q8-boundaries --resident-experts-check --out-dir "$RESIDENT_OUT" \
  > "$RESIDENT_DIE_LOG" 2>&1
```

Require exit 0, `RESIDENT_ACCEPTED backend=vulkan experts=2 indices=0,2
expert_order=3,1 row_bytes=20480 bit_differences=0 tensor_budget=16777216`,
owned tensor bytes no greater than that budget, `released_bytes=0
live_allocations=0`, and both `RESIDENT_COMPLETED` and final `COMPLETED` files.
Inspect the Vulkan device/hybrid manifest, resident fixture/order/budget manifest,
raw row bytes and native metrics. Then repeat once on the other confirmed die
with fresh paths, after inspecting the first result. No retry loop, model,
performance sweep, listener or network inference is requested.

Report native CPU differences separately for both experts. The existing first
expert comparison remains `signal.selected_vs_native`; the additional comparison
is `signal.expert3.direct_hybrid_vs_native`. Exact adapter/direct-hybrid rows do
not establish CPU bit equivalence, model quality or production readiness. Keep
partial/failure evidence and stop for diagnosis.
