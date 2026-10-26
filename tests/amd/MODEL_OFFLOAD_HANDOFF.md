# Bounded Vulkan model offload acceptance

For the subsequently requested single-entry native arithmetic discriminator, use
[NATIVE_ROW_HANDOFF.md](NATIVE_ROW_HANDOFF.md). It adds no model or trace run.

Track [issue #23](https://github.com/Zutfen-LLC/Strata/issues/23). This integration
stacks on PR #30 and selectively incorporates the listener, mTLS and TCP_NODELAY
code from PR #27; those PR branches remain independent. This is a run handoff,
not evidence of model or GPU execution at this revision.

## Contract and admission

The coordinator remains a CUDA native build on 02. An explicit
`--remote-gpu-worker vulkan@HOST:PORT PLAN` selects Vulkan for that endpoint;
unprefixed endpoints and `cuda@` retain CUDA/q8_1. Vulkan negotiates the existing
Vulkan hybrid numerical identifier plus indexed rows. It requires one resident
CPU group endpoint with full trusted backing, `--remote-cpu-entry-rows`, and the
existing ordered CPU+GPU union reducer. It cannot use weighted replies, legacy
CPU backing, packed GPU activations or numerical fallback. CUDA workers may use
separate disjoint plans in the same group; first acceptance uses only one Vulkan
endpoint and one physical V340 die.

Supported Vulkan windows are 1..8 tokens, K=1..10, Q8_0 H2560/FF640. Each active
entry invokes the existing single-row graph with activation row `entry / K`.
Returned indices remain in the original T*K space. The resident tensor budget,
source digests, plan identities and original reducer remain enforced. Larger
windows fail; they are never truncated. No new batched shader is involved.

Only parent-dispatched Hermes execution after review/admission may run the steps
below. Host-specific 05 certificate/network approval is pending. Do not create
credentials, alter trust stores, listeners, SSH/forwards/firewalls or resident
services to make a step work. Reuse approved resources after their separate
admission. Stop on missing installed dependencies; do not download or install.

## Incremental build, reusing retained 05 dependencies

Use the previously successful standalone Unix Makefiles build and the same
isolated source location; stage the exact integration commit there only after
preserving its previous source/binaries/evidence. Keep the ggml source pinned to
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, retained Vulkan SDK/ICD/compiler/library
paths, and existing Vulkan build options. Do not traverse shader targets. These
commands are prospective; they have not been run on 05 for this revision.

```bash
: "${STRATA_SRC:?retained isolated source staged at the exact integration commit}"
: "${AMD_BUILD:?retained successful standalone Unix Makefiles build}"
: "${HANDOFF_LOGS:?fresh owned log directory}"
: "${MTLS_OPENSSL_ROOT:?existing approved OpenSSL 3.5 or newer prefix, no install}"
mkdir "$HANDOFF_LOGS"
cp "$AMD_BUILD/strata-vulkan-q8-expert" "$HANDOFF_LOGS/probe.before"
cp "$AMD_BUILD/strata-vulkan-expert-worker" "$HANDOFF_LOGS/worker.before"
timeout 120s cmake -S "$STRATA_SRC/tests/amd" -B "$AMD_BUILD" \
  -DSTRATA_WORKER_TLS=ON -DOPENSSL_ROOT_DIR="$MTLS_OPENSSL_ROOT" \
  > "$HANDOFF_LOGS/configure.log" 2>&1
# CPU-only dependencies, including the capture codec newly used by the probe.
timeout 300s cmake --build "$AMD_BUILD" --target strata_gpu_protocol strata_worker_tier -j2 \
  > "$HANDOFF_LOGS/protocol-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata_vulkan_executor/fast -- -j2 \
  > "$HANDOFF_LOGS/adapter-build.log" 2>&1
timeout 120s cmake --build "$AMD_BUILD" --target strata-vulkan-q8-expert/fast \
  strata-vulkan-expert-worker/fast amd_gpu_worker_regression/fast \
  amd_gpu_group_regression/fast amd_tls_transport_regression/fast -- -j2 \
  > "$HANDOFF_LOGS/executables-build.log" 2>&1
ctest --test-dir "$AMD_BUILD" --output-on-failure --timeout 120 \
  > "$HANDOFF_LOGS/host-tests.log" 2>&1
```

Require six passing host tests. Stop on any dependency, shader, build or test
failure. On 02, reuse its current CUDA build/options/dependencies and approved
OpenSSL prefix, reconfigure with `STRATA_WORKER_TLS=ON` and incrementally build
`strata` plus its existing host/dispatcher regression targets. Do not replace
running binaries/services. CUDA/HIP/SYCL builds are not established by the local
CPU graph substitution tests; keep the PR draft until the required checks exist.

## One small actually routed subset

Use an existing sealed plain-Q8 CPU capture for the frozen prompt, its exact
trusted local manifest and existing staged/cache real expert blobs. Use the
existing `gpu_capture_replay_check --select CAPTURE PLAN...` / `--inventory
CAPTURE BLOB_DIR PLAN...` to inspect routing and cache basenames. Choose at most
two assigned layer/expert pairs, both active in the first three intersecting
records, then use the existing metadata-only manifest command:

```bash
"$COORD_BUILD/strata-gpu-manifest" plan "$LOCAL_MANIFEST" "$SMALL_ASSIGNMENTS" "$SMALL_PLAN"
```

`SMALL_ASSIGNMENTS` has one zero-based `LAYER EXPERT` per line. Preserve the full
CPU partition. The small plan must be disjoint from local GPU placements and all
other remote plans. Stage that plan, the source manifest, the capture, and only
the one or two needed existing content-addressed blobs in a private approved
location on 05. Do not generate synthetic substitutes or read/copy a full model.
No plan/provenance framework or additional runner is needed.

After parent admission, confirm the physical PCI/Vulkan mapping from retained
inventory and select one V340 die. Do not assume a physical index from old logs.
`GGML_VK_VISIBLE_DEVICES` filters to that die; application `--device 0` then names
the filtered device. Retain the existing ICD environment. The probe creates only
its temporary local service and scratch cache; no LAN service is needed here.

```bash
: "${VK_PHYS_DIE:?confirmed physical index of one V340 die}"
: "${CAPTURE:?sealed real CPU capture for the frozen request}"
: "${LOCAL_MANIFEST:?exact trusted source metadata}"
: "${SMALL_PLAN:?one or two actually routed real Q8 experts}"
: "${REAL_BLOBS:?directory containing the existing SGPU1 blob basenames}"
: "${REAL_ROWS_OUT:?new output path that does not exist}"
timeout --signal=TERM --kill-after=5s 120s \
  env GGML_VK_VISIBLE_DEVICES="$VK_PHYS_DIE" \
  "$AMD_BUILD/strata-vulkan-q8-expert" --backend vulkan --device 0 \
  --native-q8-boundaries --capture "$CAPTURE" --plan "$SMALL_PLAN" \
  --local-manifest "$LOCAL_MANIFEST" --blob-dir "$REAL_BLOBS" --out-dir "$REAL_ROWS_OUT" \
  > "$HANDOFF_LOGS/real-rows.log" 2>&1
```

Require exit 0, `CAPTURE_ROWS_COMPLETED`, `CAPTURE_ROWS_ACCEPTED backend=vulkan`,
one prepare, zero fetches/transfers, stable resident counters and zero released
owned bytes/live allocations. Every saved `capture.recordN.rows.f32` must byte
match its `capture.recordN.direct.f32`; logs identify real record/T/K/entry counts.
The comparison uses the same real weight bytes, activations, backend, device and
hybrid boundary contract. The probe validates capture/source identity/index and
reconstructs the complete trusted small plan before loading it. It selects at
most three intersecting records and requires each assigned expert to be used.
`--backend cpu` is explicitly a graph substitution and cannot qualify Vulkan.

The 16 MiB adapter tensor budget holds one or two resident experts. Direct
comparison graphs also consume memory. Tensor counters do not bound total VRAM,
host memory, ggml compute buffers, pipelines or driver memory. Retain actual
resource observations; do not infer an external resource-guard pass.

## One prompt/decode request against the CPU baseline

Only after row acceptance and separate approval for the existing 02 -> 05 mTLS
arrangement, start one owned Vulkan worker using the same selected die and plan.
All addresses, certificate files and exact SAN identities below must come from
that approved arrangement. No placeholders may be guessed or replaced by a
plaintext or forwarded endpoint. Accepted-side TCP_NODELAY remains omitted/off.

```bash
: "${WORKER05_IP:?approved private numeric bind address}"
: "${COORD02_IP:?approved exact numeric source allowlist address}"
: "${WORKER_PORT:?approved unused private worker port}"
: "${WORKER_CACHE:?dedicated approved cache directory}"
: "${TLS_CA:?approved CA file}"
: "${TLS_WORKER_CERT:?approved 05 certificate chain}"
: "${TLS_WORKER_KEY:?approved 05 private key}"
: "${COORD_SAN:?approved exact coordinator SAN DNS identity}"
env GGML_VK_VISIBLE_DEVICES="$VK_PHYS_DIE" \
  "$AMD_BUILD/strata-vulkan-expert-worker" --device 0 --plan "$SMALL_PLAN" \
  --cache-root "$WORKER_CACHE" --tensor-budget-mib 16 --cache-budget-mib 64 \
  --bind-ip "$WORKER05_IP" --allow-peer-ip "$COORD02_IP" --port "$WORKER_PORT" \
  --tls-ca "$TLS_CA" --tls-cert "$TLS_WORKER_CERT" --tls-key "$TLS_WORKER_KEY" \
  --tls-peer "$COORD_SAN" --deadline-ms 10000 --setup-ms 600000
```

Use the existing validation harness and retained CPU A0 assets/config. Preserve
its one complete resident CPU arena, binary/manifest, local GPU placements,
prompt IDs, GEN line, context, prefill/short-read/spec settings, affinity and
finite limits. Do not change them to obtain agreement. Create a fresh candidate
run root with exactly one request (`requests=[0]`), group CPU route, entry rows,
resident reuse and `external_gpu_rows=true`; attach to the existing full CPU
resident. `final_worker` and `failure_worker` remain `retain`. Replace the prior
external GPU worker list with only the small Vulkan endpoint/plan; the full CPU
backing executes the remaining entries. There is no silent reassignment on error.

Candidate coordinator argv additions (real paths/SAN from approved config):

```text
--remote-gpu-worker vulkan@WORKER05_IP:WORKER_PORT SMALL_PLAN
--remote-gpu-tls-ca TLS_CA
--remote-gpu-tls-cert TLS_COORD02_CHAIN
--remote-gpu-tls-key TLS_COORD02_KEY
--remote-gpu-tls-peer WORKER05_EXACT_SAN
```

Use the existing harness `reference` token gate and
`first_logits_reference` pointing to CPU A0, with `capture_first_logits=true`.
For the first-request trace arm, set `STRATA_FIRST_REQUEST_TRACE` to a new output
path and expect its existing `DONE 0 ... trace` receipt; the trace includes prompt
and final T1 decode evaluation but intentionally stops at first logits. Keep the
existing startup markers (including the historical `CPU+CUDA original-order
union` text), require endpoint report `Vulkan/native-Q8-host-boundaries
approximate` and `transport=mTLS`, and confirm at least one Vulkan entry served.

```bash
python3 -B "$STRATA_SRC/tools/validation-harness/harness/bundle.py" --config "$VULKAN_CONFIG"
python3 -B "$VULKAN_RUN_ROOT/run.py" --run-root "$VULKAN_RUN_ROOT"
"$PYTHON_WITH_NUMPY" "$STRATA_SRC/tools/compare_first_request_trace.py" \
  "$A0_TRACE" "$VULKAN_TRACE" --a0-logits "$A0_LOGITS" --b-logits "$VULKAN_LOGITS" \
  --output "$TRACE_COMPARISON"
```

Preserve sealed outputs even if the harness's existing exact-byte first-logit
gate fails. Use its current trace comparator and fixed logit metrics from
`tools/validation-harness/compare_logits.py` without threshold edits. Report the
first differing stage/token/entry and numeric deltas. Do not reinterpret a
same-backend row pass as CPU bit parity or rerun with another backend as a pass.
The previously reported synthetic native-CPU output had 489/2560 bit differences
and relative L2 about 5.26e-8; full-model CPU parity remains unestablished.

Stop after this one candidate request and comparison, including on mismatch.
Use normal QUIT/reaping for owned processes; preserve the complete CPU resident
and unrelated services. Return build/head/device/environment identities, row
indices/counts, exact comparisons, actual resource observations and all failed
or unrun gates to the parent. No broader fanout, sweep, merge or deployment.
