# Historical Vulkan loopback acceptance

This records the PR #30 handoff at `5d4e44f6`, where T2 was intentionally refused.
The current integration accepts bounded T1..8 windows and compares each saved
reply against its own same-token direct reference. Use
[MODEL_OFFLOAD_HANDOFF.md](MODEL_OFFLOAD_HANDOFF.md) for current commands, six host
tests, real captured weights/activations and the one-request CPU trace/logit gate.
The commands below are historical and must not be used as current acceptance.

# Vulkan loopback service/client acceptance

Stack on [PR #29](https://github.com/Zutfen-LLC/Strata/pull/29), exact base
`146d76beea5493befab228d514f905472aba06a8`; track
[issue #23](https://github.com/Zutfen-LLC/Strata/issues/23). This is a handoff, not
hardware execution evidence for the follow-on head.

Reuse the successful isolated source/build paths, compiler configuration and
pinned ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d` libraries/dependencies.
Stage the exact follow-on head into the same isolated source path. Preserve old
executables and prior evidence. Reconfigure once to add the loopback worker,
client/service linkage and host regression target. The protocol library has
only CPU dependencies; `/fast` targets avoid Vulkan shader traversal.

```bash
: "${STRATA_SRC:?same isolated source, staged at exact follow-on head}"
: "${AMD_BUILD:?successful standalone Unix Makefiles build}"
: "${LOOPBACK_LOGS:?fresh owned log directory}"
mkdir "$LOOPBACK_LOGS"
cp "$AMD_BUILD/strata-vulkan-q8-expert" "$LOOPBACK_LOGS/probe.before-loopback"
timeout 120s cmake -S "$STRATA_SRC/tests/amd" -B "$AMD_BUILD" \
  > "$LOOPBACK_LOGS/configure.log" 2>&1
timeout 300s cmake --build "$AMD_BUILD" --target strata_gpu_protocol -j2 \
  > "$LOOPBACK_LOGS/protocol-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata_vulkan_executor/fast -- -j2 \
  > "$LOOPBACK_LOGS/adapter-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata-vulkan-q8-expert/fast -- -j2 \
  > "$LOOPBACK_LOGS/probe-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata-vulkan-expert-worker/fast -- -j2 \
  > "$LOOPBACK_LOGS/worker-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target amd_gpu_worker_regression/fast -- -j2 \
  > "$LOOPBACK_LOGS/regression-build.log" 2>&1
ctest --test-dir "$AMD_BUILD" --output-on-failure --timeout 120 \
  > "$LOOPBACK_LOGS/host-tests.log" 2>&1
```

Require all four host tests to pass; their graph substitution is explicitly CPU.
Stop on missing libraries, build/test failure or shader rebuild requests. No new
downloads, installs, credentials, LAN binding or security changes are needed.
This increment was checked with the local CPU configuration. The retained Vulkan
configuration and hardware execution remain pending for this follow-on head.

After parent admission, select one physical die from retained PCI/Vulkan
inventory. The one-process harness binds only an ephemeral `127.0.0.1` port and
runs the real worker service/client path; do not launch another worker or open
LAN access for this test.

```bash
: "${VK_PHYS_DIE:?confirmed physical Vulkan index for selected die}"
: "${LOOPBACK_OUT:?new output directory that must not exist}"
: "${LOOPBACK_DIE_LOG:?new per-die log path}"
timeout --signal=TERM --kill-after=5s 120s \
  env GGML_VK_VISIBLE_DEVICES="$VK_PHYS_DIE" \
  "$AMD_BUILD/strata-vulkan-q8-expert" --backend vulkan --device 0 \
  --native-q8-boundaries --loopback-worker-check --out-dir "$LOOPBACK_OUT" \
  > "$LOOPBACK_DIE_LOG" 2>&1
```

Require exit 0, `LOOPBACK_ACCEPTED backend=vulkan requests=3 experts=2 prepares=1
fetches=2 unsupported_t2_refused=1 indices=0,2 expert_order=3,1 row_bytes=20480
bit_differences=0`, stable owned tensor bytes no greater than 16,777,216,
`released_bytes=0 live_allocations=0`, and `LOOPBACK_COMPLETED` plus final
`COMPLETED` files. Verify all three saved loopback request rows byte-match the
concatenation of `signal.expert3.direct_hybrid.out.f32` and
`signal.selected.out.f32`. Inspect the device/hybrid manifest and native metrics
for both experts separately. Expected CUDA negotiation and T2 refusal log lines
are intentional test cases; no fallback or automatic retry is allowed.

Inspect the first result, then repeat once on the other confirmed die with fresh
paths. Retain resource observations actually available. Owned tensor counters
exclude host, backend compute, pipelines and driver allocations; no RSS,
peak-VRAM or external guard receipt is inferred. No full model, multi-host,
performance sweep or mTLS work belongs to this acceptance run.
