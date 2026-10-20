# Bounded CPU weight residency across engine sessions

**NOT RUNTIME VALIDATED.** The final source has not been built or tested. Earlier checks were interrupted
and one residency suite failed. See the [retained failure and static corrections](evidence/validation.md).
Run the fixture-first Hermes handoff before any full-model trial.

The worker can keep a complete verified CPU partition after an engine exits. Enable both worker
`--residency-ms 1800000` and engine `--remote-cpu-resident` with the existing trusted local manifest.
The default is off. [Hermes recipe](HERMES.md) includes GPU release, reconnect, unload and expiry checks.

## Lifecycle and identity

Resident weights have their own lifetime. Session identity, sequence history, setup controls and
request buffers belong to a connection. EOF, frame timeout and every new setup invalidate that session.
A new engine must negotiate a fresh session and READY even on a hit. Old session ids and repeated
request sequences remain errors. Inference framing, kernels, arithmetic and merge order are unchanged.
The worker does not retain prompts, activations or results between requests.

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
and restarts after successful inference completes. It does **not** restart on disconnect, status,
refusal, or setup negotiation. A successful new READY starts a new idle window. The lifetime includes
connected idle time. The old default 10000 ms full socket read/send deadline still applies: an idle
engine connection may close at ten seconds, while its verified weights remain available to a new engine.
The tier still fails closed on that loss; it does not retry inference silently.

The listener sweeps expiry every 100 ms. A connected idle read checks stop, setup expiry and residency
expiry at most every 50 ms. Expiry invalidates any session and unloads the arena; the service remains
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
to its old 28 bytes (40 bytes total). A disabled/new worker or the tested old worker refuses this flag;
the client fails closed without fallback. With the flag absent, SETUP, LOCAL_SETUP, UPLOAD, READY,
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
real worker binaries. No full-model numerical parity, CUDA/HIP/SYCL build, GPU release on real hardware,
SATA throughput or full-size RSS measurement is claimed by this candidate.
