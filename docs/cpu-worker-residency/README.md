# Bounded CPU weight residency across engine sessions

**7992bb0d: user-reported complete native-OFF PASS; native-ON stopped at the ISA checker.**
ON configure/build, four math and two dispatch checks passed, then `isa_guard_check` failed on the
pre-existing unregistered TQ AVX512 object. Device-plan and all 05 checks did not run. This is **not an
overall PASS**; see the [attributed receipt](evidence/hermes-7992bb0d-receipt.md).
The subsequent checker-only correction is runtime-unvalidated. Use the
[bounded continuation handoff](ISA_CHECKER_HERMES.md), which reuses verified safe outputs.

The [79f4da0 OFF link failure](evidence/hermes-79f4da0-failure.md) and independent 05 results remain
preserved. The [16d065e receipt](evidence/hermes-16d065e-receipt.md) records the user's earlier CPU/CUDA,
compatibility and one full-model cold/warm trial PASS. No full-model reload is needed for the checker
correction. Earlier failed/interrupted coding-task history remains in [validation.md](evidence/validation.md).

The worker can keep a complete verified CPU partition after an engine exits. Enable both worker
`--residency-ms 1800000` and engine `--remote-cpu-resident` with the existing trusted local manifest.
The default is off. [Hermes recipe](HERMES.md) includes GPU release, reconnect, unload and expiry checks.

## Lifecycle and identity

Resident weights have their own lifetime. Session identity, sequence history, setup controls and
request buffers belong to a connection. EOF, frame timeout and every new setup invalidate that session.
A new engine must negotiate a fresh session and READY even on a hit. Old session ids and repeated
request sequences remain errors. Inference framing, kernels, arithmetic and merge order are unchanged.
Each request supplies fresh activations and produces fresh results; none are reused across requests.

Reuse requires a complete successful persistent READY and exact equality of the full 128-bit trusted
local-manifest identity, model id, quantization/layout/geometry, partition id and every ownership bit.
The manifest identity covers expert and role digests, slice mapping, native index and file inventory.
The setup nonce and the model's log-only name are excluded. There is no partial reuse or rebalancing.
Upload-only setup has no trusted per-expert identity table and cannot create or reuse a persistent cache.
Local upload fallback still needs explicit permission on both ends; it is never enabled by residency.

A resident mismatch returns `cache-mismatch`. It invalidates the session and retains the old verified
arena until expiry or explicit unload. It does not allocate a second arena, evict a usable partition,
read model bytes, or fall back to upload. A matching setup may then bind the same cache. To change a
partition/model, unload first. Without a resident cache, invalid setup is refused normally; an incomplete
or failed new setup is never promoted and is freed on disconnect, receive timeout, total local setup
timeout, unload or stop. A reconnect cannot continue an incomplete upload.

On a hit the engine skips all LOCAL_LOAD, fetch and UPLOAD operations. READY checks the retained complete
slot table without rehashing weights. The verified arena is immutable. Changing or deleting the source
files does not change these resident bytes; an intentional asset change requires a new trusted manifest
and unload. The existing noncryptographic two-pass FNV digests remain integrity checks, not authentication.
Pin the manifest SHA-256 and trusted model-copy receipts as before. Loopback plus an owned SSH forward
remains the access boundary. A worker process restart always starts empty.

## Bounds and control

`--residency-ms N` accepts 0 (disabled) or 1..86400000 ms (one day). Idle age starts at successful READY
and restarts after a successful inference reply completes. It does **not** restart on disconnect, status,
refusal, or setup negotiation. A successful new READY starts a new idle window. The lifetime includes
connected idle time. The old default 10000 ms full socket read/send deadline still applies: an idle
engine connection may close at ten seconds, while its verified weights remain available to a new engine.
The tier still fails closed on that loss; it does not retry inference silently.

The listener sweeps expiry every 100 ms. A connected idle read checks stop, setup expiry and residency
expiry at most every 50 ms. Telemetry/setup/refusal sends also observe idle expiry, so a nonreading
peer cannot extend it to the I/O deadline. Successful inference execution and its reply stay under
the existing active-request rules; idle age restarts after that reply completes. Expiry invalidates any
session and unloads the arena; the service remains
available for a cold setup. Native computation is not interrupted by idle expiry. It completes under
the existing execution contract and starts a new idle window. Socket deadlines and whole local setup
budgets are distinct, unchanged in length; regular-file OS calls and native kernels are not forcibly
preempted. Use an owned outer process timeout for a hard wall cap on those operations.

The arena stays charged to `--memory-budget-mib` while disconnected. Reuse allocates no new arena.
Unload releases the old arena before a replacement setup allocates. The budget is for weight payloads,
not total RSS: reserve CPU pool workspace, metadata, allocator overhead and OS memory separately.

After the engine exits, use the supplied tool locally on the worker or through an owned forward:

```sh
python3 "$SRC/tools/worker_control.py" status --port 7431
python3 "$SRC/tools/worker_control.py" unload --port 7431
python3 "$SRC/tools/worker_control.py" stop --port 7431
```

`unload` invalidates the session and frees weights **before** acknowledging; it keeps the service alive.
`stop` acknowledges a stop request; verify process exit and the cleanup log separately. SIGINT/SIGTERM
also stop the worker. Only one connection is served at a time. Close the engine connection before using
the control tool. The tool uses five-second bounded exchanges and numeric IPv4, with no DNS dependency.
Expiry/unload/stop logs name the reason and released bytes. Regular heap release is synchronous;
allocator/OS RSS reclamation is platform-dependent.

The shutdown path now frees the arena and joins the CPU pool **before** printing final summaries.
Stop cancels active socket waits and lets the listener poll finish; only the service thread closes its
sockets. Cleanup logs distinguish `arena released; joining CPU pool` from `CPU pool joined` and process
exit. Static inspection of the base found that its summaries preceded pool destruction, and stop only
closed the listener, leaving an active receive waiting for its deadline. Neither proves the cause of
the reported Hermes RC 124. A timeout wrapper returns 124 when its own wall expires. The small binary
checks exercise both protocol stop and SIGTERM, including a connected idle peer; real-host teardown
still needs measurement. No arena-release diagnosis of that earlier report is claimed.

## Wire compatibility and evidence

LOCAL_SETUP local-v1 flag bit 1 (`2`) explicitly requests persistent residency. Bit 0 remains the
missing-only upload permission. On this flag only, LOCAL_SETUP_ACK appends `hit:u32, reused_bytes:u64`
to its old 28 bytes (40 bytes total). A disabled/new worker and the exact-base worker refuse this flag;
the client fails closed without fallback. The user reports cross-version checks passed at 16d065e
using the actual base worker and legacy source client; see the [receipt](evidence/hermes-16d065e-receipt.md).
The later correction head has not been run. With the flag absent, SETUP, LOCAL_SETUP, UPLOAD, READY,
INFER and STATUS keep their existing wire layouts. A configured TTL alone does not retain legacy
sessions. A legacy setup cannot silently replace an already persistent cache; unload first.

The separate CACHE_STATUS (16)/ACK (17) is empty request / 100-byte reply:
`version:u32=1, session_state:u32 (0 empty/1 setup/2 ready), session:u64, resident_bytes:u64,
idle_remaining_ms:u64, cache_hits:u64, reused_bytes:u64, loaded_bytes:u64, verified_bytes:u64,
peak_arena_bytes:u64, arena_allocations:u64, expirations:u64, unloads:u64, complete:u32`.
UNLOAD (18)/ACK (19) both have empty payloads. Legacy STATUS stays exactly 59 bytes.

Loaded/verified bytes count actual payload reads/received bytes and digest coverage, cumulatively,
including failed attempts. A digest's two passes count its payload length once. Network weights stay
in the existing `setup_weight_bytes`. Cache hits/reused bytes count accepted matching setups, before
READY; prove use with a new session, successful fresh inference and READY too. Peak and allocation
counters describe the single payload arena, not total process allocation. The status tool combines the
cache extension with legacy request/network counts. Status probes do not renew idle age.

The engine's setup and request report includes `cache_hit`, reused bytes and session id. Local read/hash
counters reset for each local session; process totals permit cold/warm deltas. A real reuse proof has
positive reused bytes, zero loaded/verified deltas and zero network weight delta, plus fresh-session
inference. A live PID alone proves nothing.

[Retained failed/interrupted validation and independent static reviews](evidence/validation.md) use generated Q8_0 fixtures and
real worker binaries. The later [user-attributed 16d065e receipt](evidence/hermes-16d065e-receipt.md)
records CPU/CUDA and real-host GPU-release observations. The new native-OFF correction has no build or
runtime validation. No independent full-model numerical parity, HIP/SYCL validation, SATA throughput or
full-size RSS measurement is claimed.

## Native experts disabled

`STRATA_NATIVE_EXPERTS=OFF` remains supported. The shared CUDA/HIP engine excludes worker calls,
construction and reports when the worker libraries are absent. All six `--remote-cpu-*` worker options
are refused with `requires STRATA_NATIVE_EXPERTS=ON` before model loading or GPU initialization. The
main preflight runs before family dispatch; Qwen3.6 also checks actual parsed options after the existing
GGUF metadata probe, covering serve-mode argument-consumption differences.
An attached worker pointer also fails closed before multi-token dispatch reads inputs or accesses a
source/device. Native-ON retains its worker path. The 79f4da0 build found an additional common CPU-kernel
link gap; the [dependency correction](NATIVE_OFF_AUDIT.md) now also moves Q2/activation and BF16 router
implementations into the common library. At 7992bb0d the user reports OFF passed and ON reached the
ISA-checker failure; its [continuation gates](ISA_CHECKER_HERMES.md) remain unrun for the checker-only head. HIP remains unvalidated. SYCL has a synchronized CPU list but separate migrated
sources; the worker feature is not claimed there.
