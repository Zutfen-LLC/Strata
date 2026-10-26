# Fixed network CUDA expert placement

This opt-in backend executes an explicitly assigned subset of experts on a remote
CUDA worker. Local GPU ownership stays fixed. The existing CPU worker retains its
partition and serves only uncovered routed entries. No discovery, live joining,
rebalancing, arena shrinking or transparent fallback is implemented.

For several independent remote GPUs, use the bounded opt-in
[remote worker group](../REMOTE_GPU_WORKERS.md). The legacy endpoint/plan pair
remains a one-worker shorthand; SGW1 and per-worker selective caches are unchanged.

The supported slice is uniform plain Q8_0/Q8_0 without Hadamard transforms or scale
tails, within the existing CPU geometry bounds. The network service is POSIX/CUDA
only. CPU-only/native-ON builds can build asset/protocol fixtures without CUDA;
native-OFF excludes network worker integration. HIP, SYCL and Windows support is
not claimed. See [validation and limits](VALIDATION.md) before assessing acceptance.

## Configuration and selective cache

Start with an existing trusted SWLOCAL1 manifest and an explicit list of zero-based
`LAYER EXPERT` pairs, one pair per line:

```sh
strata-gpu-manifest plan existing-trusted.local assigned.txt fixed.plan
```

Plan generation reads bounded metadata rather than weights. Copy only that plan
to the worker and create a dedicated owned cache directory. Example limits below
are operator choices, not a recommendation that any assignment fits:

```sh
strata-cuda-expert-worker --plan fixed.plan --cache-root CACHE --device 0 --port 7432 --cache-budget-mib 24576 --reserve-mib 1024
```

The engine adds `--remote-gpu-experts NUMERIC_FORWARD_HOST:PORT --remote-gpu-plan
fixed.plan` to its supported serial remote-CPU configuration, including the trusted
`--remote-cpu-local-manifest`. Every assigned expert must be in that CPU partition
and outside all local GPU caches. Model, layout and placement are frozen for the
session; change them only by closing the session and starting a new one.

For the opt-in encrypted coordinator/GPU-worker path, see [persistent mTLS](MTLS.md).
Plain direct TCP is an explicit test mode; source-IP filtering provides no encryption.

Both services bind loopback by default. Use an existing trusted transport; the protocol does
not provide authentication. SGW1 negotiates CUDA execution, q8_1 activations and
F32 results separately from the unchanged SEW1 CPU native-bitwise contract.
Incompatible peers fail rather than silently selecting another arithmetic path.

The worker rechecks only assigned cache blobs and returns a missing bitmap. The
engine reads and sends only missing or corrupt assigned gate/up/down slices from
its trusted model source. Complete validated files survive restarts. Partial
uploads never become hits; reuse is validated per complete entry, not inferred
from persisted byte totals. No whole-model mirror or automatic eviction occurs.
Keys bind model, layout, layer, expert, length and expected digest. Digest128 is
the existing noncryptographic integrity check; the manifest and trusted transport
remain the trust boundary.

The cache uses generated relative names, directory descriptors and no-follow
regular-file opens. New files publish atomically after length/digest verification.
The disk budget counts retained regular files, including old assignments and
temporary files, and reserves new assets before accepting them. Intentional cache
cleanup requires exclusive ownership with the worker stopped and preserves
unrelated files.

Host staging is bounded by one expert blob (at most 8 MiB), protocol frames and
inference buffers. A contiguous send adds at most 4,194,384 bytes for one SGW1 frame
(4 MiB plan bound + 64-byte allowance + 16-byte header), not an assignment-sized
allocation. Wire bytes and the controlled writer's cancellation/deadline behavior
are unchanged. Copy time is charged before the socket writer; no TCP option changes
are made. Counters distinguish source reads, transfer, persistence, cache reuse,
verification, host staging and GPU allocation. Logical source reads do not measure
physical disk I/O or transport framing overhead.

## Readiness, execution and failure

READY requires verified assigned weights resident on the selected GPU and a
completed CUDA synchronization. Free VRAM must cover weights, actual scratch and
metadata/input/output allocations, plus the configured reserve. Every allocation
and copy is checked; an estimated assignment size is not proof of fit. Disk cache
and VRAM have separate limits and lifetimes. Disconnect releases the GPU session;
complete disk entries may remain.

The engine submits the existing asynchronous CPU request before queueing copied
GPU requests to per-endpoint communication threads. Each socket has one owner,
including idle STATUS probes. Completion remains CPU finish then group finish;
helper/endpoint-configuration-order/CPU weighted merge order is deterministic.
On GPU submission failure or exception, the posted CPU request is joined before
returning. Later peer failure drains the group before any network sum commits.
No CPU fallback/retry is performed. Existing I/O deadlines and outer process
guards remain necessary; filesystem calls and kernels cannot be preempted by
socket timers.

Requests/replies bind model, layout, placement, session, sequence and request
identity. Unassigned entries, stale/duplicate replies, malformed lengths and
nonfinite results are errors. Each owner contributes its weighted partial once,
only after both remote tiers succeed. A failure latches the affected tier dead.

`--remote-gpu-setup-ms` bounds each worker's total setup (default 600000 ms), while
`--remote-gpu-deadline-ms` bounds ordinary frame I/O (default 10000 ms). SETUP/READY
waits use the remaining total setup budget. During GPU setup, probes keep the
already-ready CPU connection active without renewing its residency TTL.

## Explicit durable CPU residency

The existing finite `--residency-ms N` remains supported. Zero or omitted preserves
session-scoped behavior. The worker's new `--residency-forever` is an explicit,
mutually exclusive alternative to a positive finite TTL; it never follows from
zero. The engine must also request `--remote-cpu-resident` with its trusted local
manifest. Nonresident sessions still release their weights on disconnect.

A healthy complete arena then survives engine disconnects with no idle expiry.
It remains owned by the CPU service, not by the engine task. Memory limits,
partition identity, bounded frame I/O and operator monitoring still apply. This
does not install a background service, open persistent remote access or expand
the loaded partition. Forever-mode detail logs stop after an 8 MiB quota; use
status counters for continued observation.

The existing `ExpertWorkerClient::cache_status()` reports `idle_remaining_ms =
UINT64_MAX` for a complete no-expiry arena, with its real byte/allocation/reuse
counters. Reconnect using the same manifest/partition and resident flag. Existing
`unload()` releases weights while leaving the worker available; `shutdown()` or
normal service SIGINT/SIGTERM releases weights and stops it. Use the owning
service's normal stop mechanism, not an unverified numeric PID. Engine or test
completion alone must not unload a healthy explicitly resident arena.

## Focused checks

With the relevant retained build configuration and dependencies, test targets are
`gpu_assets_test`, `gpu_worker_test`, `gpu_worker_dispatch_test`, `gpu_frame_test`,
`cpu_first_remote_post_test` and `worker_service_residency_test`. The first two also
have CPU-only protocol/asset coverage; the dispatcher and CUDA worker cases need
the CUDA executor. The frame fixture uses the real socket/codec seam without CUDA
or model weights. Its optional `--latency` compares split framing, contiguous
framing and fixture-only accepted-side TCP_NODELAY; timing has no CI pass threshold.
The residency fixture starts its own tiny synthetic worker and must never target
an existing service. See [attributed results and unrun checks](VALIDATION.md).
