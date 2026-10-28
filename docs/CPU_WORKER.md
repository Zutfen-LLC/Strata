# The resident CPU-expert worker (`strata-expert-worker`)

The worker is a small service, bound to loopback by default, that holds one model's CPU-owned expert blobs in RAM and computes
their weighted partial sums with Strata's native CPU kernels (`ggml-cpu` arithmetic through
`native_expert.hpp`). A GPU engine process keeps its model, router and cache on its own machine and sends
the CPU-owned experts' work here. It is opt-in and experimental: nothing in the default engine path uses
it. InferSwarm issue #348 is its specification; #349 wires it into the real engine.

## What it is not

Not a model server, not a scheduler, not a second inference engine. It computes exactly one thing: the
weighted sum of the expert slots it was assigned, per token. The network boundary carries typed ids and
values, never weight bytes during inference, never process pointers, and never prompt text.

## Building and running (CPU only)

```sh
cmake -S . -B build-worker -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-worker --target strata-expert-worker
./build-worker/strata-expert-worker --port 7431 --memory-budget-mib 131072 --verbose
```

No CUDA/HIP/SYCL toolkit is involved; the target links only `strata_kernels_cpu` (and ggml-cpu). The
listener binds **127.0.0.1 by default**. To reach it through an authenticated, encrypted connection,
forward a port over an SSH tunnel you own (for example
`ssh -L 7431:127.0.0.1:7431 "$WORKER_HOST"`) and point the client at the tunnel's local end.

For an operator-controlled trusted LAN, explicitly select a local address with `--bind-ip IPv4` and
repeat `--allow-peer-ip IPv4` for each permitted source. A non-loopback bind without peers is refused.
Addresses must be numeric unicast IPv4; wildcards, hostnames, IPv6 and CIDR ranges are refused. The
worker compares the kernel-reported source against the exact list immediately after accepting a
connection, before reading frames or dispatching session, residency or control operations. A denied
connection closes without a protocol reply. When a list is supplied, even loopback callers must match it.
Clients and `tools/worker_control.py --host IPv4` use the selected address without wire changes.
LAN mode has one selected endpoint, so local control must use `--host` with that address and its actual
source must be listed. Explicit bind/peer policies require a positive `--deadline-ms` (default 10000).

Source-address filtering provides **neither authentication nor encryption**. Session IDs are not
credentials. Use this mode only on a trusted LAN with an operator-managed firewall limiting the worker's
interface, port and source peers. Strata never installs firewall rules. An allowed host, any process on
it, or a host able to impersonate its address can send inference and destructive unload/stop controls.
Keep the SSH route when that boundary is insufficient. See the [LAN and NIC operator guide](cpu-worker-lan/README.md)
and the [operator handoff](cpu-worker-lan/HERMES.md).

Options: `--port P` (0 picks a free port and prints it), `--memory-budget-mib N` (default 131072),
`--workers N` (CPU expert-pool worker threads; 0 = the pool's default plan for the machine),
`--deadline-ms N` (budget per full socket send/read, default 10000), `--verbose` (one identity/status line per
request; still no payloads are logged).

For local weight setup, `--local-load-threads N` selects 1–4 loading lanes, separate from inference
`--workers`. The default is 4; explicit 1 keeps the serial reader. See [parallel local loading](cpu-worker-local/PARALLEL_LOADING.md)
for memory bounds, timing interpretation and validation status.

## Fixed CPU worker group (experimental)

The engine can opt into one or two CPU endpoints with repeated `--remote-cpu-worker IPv4:PORT`.
Use it with the existing trusted `--remote-cpu-local-manifest` and local Q8_0 assets on every worker.
Do not combine it with `--remote-cpu-experts` or network GPU worker options. Local helper GPUs keep
their existing placement. The old one-endpoint option and default path are unchanged.

The engine freezes the non-GPU expert set, walks it in layer/expert order, and assigns consecutive
eligible experts round-robin to endpoints in command-line order. Endpoint order defines ownership
and reduction order. Each endpoint must own something; duplicate endpoints, overlap, gaps, different
trusted model/layout or invalid geometry are refused before group connection. Active peers receive
copied inputs before the engine waits. Zero-work peers send no INFER; bounded status probes keep
their sessions alive. A failure drains outstanding requests and exposes no group partial. No failed
inference is retried. Matching resident subsets can reconnect; changing from full ownership to a
subset requires the existing explicit owned replacement/reload procedure.

Without `--remote-cpu-entry-rows`, replies retain the existing weighted-partial wire format. The engine copies the first active
partial, then adds remaining partials in endpoint order. This changes floating-point reduction order
from an unsplit worker and does not promise bitwise equality to that worker. Grouped-reference bits,
independent unsplit error, finite results, token streams and task correctness are separate gates.
Per-peer request/service counters and timings are reported; overlapping peer times must not be summed
as elapsed wall time. Equal expert counts need not mean equal compute or completion time, and transport
overhead can outweigh parallel work. No speedup has been measured for this change.

### Opt-in entry rows

`--remote-cpu-entry-rows` requires a fixed CPU group and capable workers. It sends one batched
request/reply per active peer/window, with the original inputs. Workers return unweighted native
entry rows; they do not pre-sum or multiply by routing weights. After all peers succeed, the engine
places rows at their original token/rank positions and calls the shared `accumulate_partial` once,
with original masks, weights and entry order. The resulting CPU partial enters the existing merge
after helper contributions. No shadow requests, per-row RPCs or extra threads run in this mode.

LOCAL_SETUP negotiates flag4 and appends row schema1/max80 to its ACK only when requested.
Old workers refuse the capability; the client never falls back to weighted replies. Existing setup
and weighted inference bytes remain unchanged when the flag is absent. Mode is session-scoped,
not part of the verified weight-cache key. Upgrading the worker requires controlled replacement
and reload; the resident arena cannot be hot-upgraded into another process.

Message20 reuses the existing INFER payload; message21 returns session/sequence (u64), schema/count
(u16), then ascending `(original entry index u16, n_embd F32 values)` records. The exact requested
set and finite values are checked; missing, duplicate, foreign or reordered indices fail the request.
Legacy and row requests share sequence ordering. Zero-work peers send nothing. Any peer failure
drains all outstanding requests and publishes no partial. Defaults and the legacy endpoint stay unchanged.

At T8/K10/H2560, a reply is at most819396B including its frame header; packed row data and central
row storage each cap at819200B. Receive/encode/send buffers have separate message caps. For T1 with
ten CPU entries across two active peers, total reply traffic is102492B versus20544B for weighted
partials. This restores the intended reduction order at a data cost; performance needs measurement.

See the [fixed two-host validation handoff](validation/CPU-FIXED-GROUP-20261025.md) for the bounded
first comparison, frozen numeric gates and private config contract. This feature and its new fixtures
are authored; executable validation is **NOT RUN** on the coding desktop.

## The service API

One persistent connection, one outstanding request at a time, every frame
`"SEW1" | msg (u16) | status (u16) | payload length (u64)` followed by that many bytes. All integers
are little-endian.

| Message | Direction | Payload |
| --- | --- | --- |
| `SETUP_BEGIN` | client → worker | nonce (u64), model id (u64), model name (48 B), quant name (16 B), gu/d types (u16+u16), layers/experts/n_embd/n_ff (u16 each), partition id (u64), ownership bitmap |
| `SETUP_ACK` | worker → client | session (u64), blob bytes (u64), chunk cap (u32) |
| `UPLOAD` | client → worker | session, layer (u16), expert (u16), total length (u32), chunk offset (u32), chunk length (u32), digest (16 B), chunk bytes |
| `UPLOAD_ACK` | worker → client | session, bytes of that slot now present (u32) |
| `READY` | client → worker | session |
| `READY_ACK` | worker → client | session, missing slot count (u64) |
| `INFER` | client → worker | session, model id, partition id, sequence (u64 each), layer (u16), T (u8), K (u8), entry mask (T·K bytes), expert ids (T·K u16), routing weights (T·K f32), activations (T·n_embd f32) |
| `INFER_ACK` | worker → client | session, sequence, partial sums (T·n_embd f32) |
| `STATUS` / `STATUS_ACK` | both | counters: setup weight bytes, inference in/out bytes, requests, failures, CPU service ns, last error, ready flag |
| `SHUTDOWN` | client → worker | (empty) - stops the worker after the reply |

Status codes are typed (`ok`, `protocol`, `wrong-state`, `bad-request`, `unsupported-layout`,
`over-budget`, `duplicate`, `not-owned`, `digest-mismatch`, `not-ready`, `incomplete`,
`stale-session`, `nonfinite`, `deadline`, `internal`); see `include/strata/worker/expert_wire.hpp`,
which is the contract's single source.

Rules that matter to a caller:

- **Setup.** `SETUP_BEGIN` starts a session and voids any previous one. The manifest declares one
  uniform native layout (gate/up type, down type, n_embd, n_ff) and the ownership bitmap; the worker
  accepts only layouts it has fixture-tested - today **Q8_0 gate/up with Q8_0 down** - and refuses an
  assigned set larger than its memory budget. Blobs upload as bounded in-order chunks (default cap
  1 MiB of data, excluding headers) into a worker-owned arena sized exactly to the assigned set; a slot is verified against the
  declared 128-bit digest when its last byte lands. Chunks must start at the slot's current fill point.
  `READY` succeeds only when every assigned slot is present and verified, and freezes the session.
- **Inference.** The frame carries identity (session, model id, partition id, sequence), the layer, T ≤ 8
  tokens, K ≤ 10 routed experts per token, a per-entry mask saying which entries this worker computes,
  the expert ids, finite routing weights and finite F32 activations. An entry with mask 1 must name a
  slot the worker owns; a mask-1 entry it does not own is refused (`not-owned`), as is any request whose
  session, model or partition identity does not match (`stale-session`) or whose sequence is not past
  the last one served (`bad-request`). The reply is one F32 weighted partial sum per token - zeros for a
  token with no owned entries - with the request's session and sequence. **No weight bytes travel in an
  inference frame**, and the whole frame stays below 96 KiB at the compiled geometry (T=8, K=10,
  n_embd=2560; the exact bound is derived in `expert_wire.hpp`).
- **Failure behavior.** Every refusal is a typed status and the connection continues. A malformed frame,
  a short frame that misses its deadline, or a disconnect closes the connection and **voids the
  session**: the next connection must run `SETUP_BEGIN` again and re-upload. A worker restart is the
  same thing from the other end - old session ids are refused.
- **Counters and logs.** `STATUS` reports setup weight bytes, inference bytes in and out, requests,
  explicit failures, and the time spent inside the native kernels. The log prints identities and
  statuses (model name, session, layer, counts) - never prompts, activations or outputs.

## The determinism contract

The worker's answer is defined to be **bitwise the direct native CPU call**:

- distinct experts are grouped in first-appearance order of their entries (the pool may compute their
  rows concurrently; a row's arithmetic is ggml-cpu's per-token dot either way);
- each token's activation is quantized once with `native_quant_act` (exactly as the engine's CPU
  dispatch does);
- each expert's rows are computed with `native_gu_rows` / `native_quant_h` / `native_down_rows` over the
  full row range (the same ggml-cpu per-token dot products the local pool runs);
- the weighted sum accumulates entries in frame order with `acc = acc + w * o` - no FMA contraction, no
  reassociation.

`ExpertWorkerService::direct_native_partial()` (in the `strata_expert_worker` library) IS that reference,
so any caller - the tests, and #349's engine integration - compares against the identical code path.

## The client library

`strata::worker::ExpertWorkerClient` (`include/strata/worker/expert_client.hpp`) implements the whole
flow: `connect_host`, `setup`, `upload` (whole blob, chunked internally), `mark_ready`, `infer`,
`status`, `shutdown`, `close`. Setup validates the exact ACK and a data cap in 1..1,048,576 B;
upload honors smaller advertised caps and validates each ACK's session and next offset.
Each upload chunk is one frame and its reply; a refusal returns false with
`last_status()` set to the typed code.

## Minimal local reproduction

`tests/worker/expert_worker_e2e_test` is exactly this: it starts the real binary with `--port 0`, packs
two Q8_0 expert fixtures with ggml's quantizer, uploads them, marks ready, sends one mixed-ownership
request (T=2, K=3, entries of both an owned and a GPU-side expert) and checks the transported partial
sums bitwise against `direct_native_partial`. `tests/worker/expert_worker_test` is the focused suite
(codec bounds, every refusal path, restart/disconnect/interrupted upload, malformed frames, deadlines,
and T=1 / T=4 / T=8 direct-versus-transport parity). Build and run them with:

```sh
cmake -S . -B build-worker -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON
cmake --build build-worker --target expert_worker_test expert_worker_e2e_test
./build-worker/expert_worker_e2e_test
```

## Limits (what this issue does not claim)

- One layout today (Q8_0/Q8_0) with direct-versus-transport coverage; other native formats join only
  with their own packed fixtures and parity tests.
- One uniform layout per model (a native pack with per-layer formats is future work; the manifest has
  one geometry).
- One connection, one outstanding request; no pipelining, no concurrency, no authentication. The default
  boundary is loopback plus an owned tunnel; explicit LAN mode relies on the operator's trusted network
  and firewall, with an exact source allowlist.
- No engine integration in the worker itself: the engine side (`--remote-cpu-experts`, above) lives in
  `ExpertWorkerTier` + the dispatch/verifier glue and keeps this service untouched.
- Windows: this correction's nonblocking/select path was not built or run there; validation is Linux
  only (the e2e and socket-bound tests are POSIX-only).
- The setup manifest declares one layout per model, so the blob LENGTH is derived once for the whole
  partition; per-blob digests ride each UPLOAD frame and are enforced when a slot completes.  A manifest
  with per-slot digest tables would be a protocol extension if a pack ever needs it.

## The engine mode (inferswarm #349): `--remote-cpu-experts HOST:PORT`

The engine side is **opt-in and default-off**: `strata --serve --remote-cpu-experts 127.0.0.1:7431`
(`--remote-cpu-deadline-ms N` budgets each full socket send/read; default 10000).  With the flag off, every path below is
untouched. Header and payload reads have separate budgets; connection establishment and native kernel
execution are not preempted by this setting. The pieces:

The latency diagnostic is also default-off. Start only the CPU worker with `--tcp-nodelay` to test checked
`TCP_NODELAY` on its accepted descriptor; GPU-worker accepts still use the unchanged default. Add
`--rpc-timing` to the CPU worker and `--remote-cpu-rpc-timing` to the engine for fixed-memory aggregate
latency histograms. The flags are independent, so socket behavior can be compared without timing overhead.
The engine reports client-side aggregates with its existing worker report. The worker reports its aggregate
only when it stops cleanly; stopping or replacing a worker unloads its arena and must be a separate approved
operation. Use [the current operator handoff](cpu-worker-lan/HERMES.md) for validation of this branch.
The [earlier latency handoff](cpu-worker-rpc-latency/HERMES.md) preserves its revision-specific evidence.

The model-free [representative RPC replay handoff](cpu-rpc-replay/HERMES.md) uses the same CPU wire frame
and persistent socket seam with fixed synthetic payloads. It is a transport/scheduling diagnostic, not an
expert-compute benchmark, and its external fixture port requires separate explicit authorization.

- **The tier** (`strata::worker::ExpertWorkerTier`, `strata_worker_tier`, CPU-only like the service):
  owns the client, the frozen ownership bitmap, one in-flight request slot on one persistent communication
  thread, and the failure latch. The thread starts at the first nonempty post, preserving the former
  posting thread's inherited affinity; the tier sets no affinity. Its OS thread id, queue time, exchange
  time and caller finish wait are aggregate report fields for live placement and timing checks.
  `tests/worker/expert_tier_test` runs its whole lifecycle against the real in-process service.
- **The partition is derived, never configured**: at startup, after CUDA0's cache, every stage cache and
  every helper tier (`--expert-cache-device1..3`) is filled, the complement - every expert no GPU tier
  holds - becomes the worker's.  Every routed expert keeps exactly one executing owner.  Setup streams
  each owned blob from the expert source through ONE blob-sized staging buffer (no expert RAM mirror on
  the engine host; the model files stay on its disk), and inference may not start before the worker has
  verified the whole set (`READY`).
- **In a verify window** (`expert_pool_dispatch_multi`), after the GPU's plan is published, the peer
  launched and the helpers begun - and before the local pool's own work - the worker entries' frame goes
  out (activations, ids, routing weights, ownership mask; nothing for a window with no owned entry).  The
  round trip overlaps the local CPU jobs; `finish` waits before anything downstream may commit.
- **The merge reuses the helper tiers' mechanism** (`RemoteExpertOpt`): the worker's entries' per-entry
  rows are zeroed (the GPU's own weighted combine adds nothing for them) and its already-weighted partial
  is added on top of the combined result, after every helper tier's partial - one contribution each,
  disjoint masks, a deterministic order run to run.  The partial itself is bitwise
  `ExpertWorkerService::direct_native_partial` (the same ordered reduction the #348 tests pin).
- **Failures latch and abort**: any refusal, deadline or disconnect marks the tier dead; the dispatch
  latches `failed`; the request aborts without committing partial state or returning stale data.  A dead
  tier never retries silently - only a full engine restart (a fresh `SETUP_BEGIN` + upload) serves again.

What the mode refuses up front (rather than silently running a different path): everything but `--serve`;
`--batch` slots; `--layer-split`; `--peer-device`; `--pipeline-windows`; no CUDA0 expert cache / pool /
`--expert-profile`; `--short-read 0`; the adaptive tier on (`--adapt-every 0` fixes placement);
`--resident-cpu-experts` (no local RAM mirror); `STRATA_ROUTE_TAIL_SKIP`.  A non-uniform or non-Q8_0/Q8_0
native pack is refused (the worker's one fixture-tested layout).  At request time, a prompt part the
verify windows cannot take - longer than `--short-read N`, or with picture rows - is REJECTED with an
error naming the batched prefill path, for the prompt ingestion as well as the decode (both go through
the windows).  The one-shot `--prompt` path and the single-token expert path are not implemented for
this mode and refuse if reached.

The evidence it reports: a startup line with the frozen partition (blob count, GiB, setup seconds); the
`DONE` line grows ` worker_entries=N worker_req_wall=ms` per request; and a per-request stderr line with
the tier's cumulative counters plus the worker's own STATUS (requests, failures, CPU service seconds -
the worker's kernel time, never the client's wait - and inference bytes in/out).

## The two-host recipe (coordinator and worker)

A self-contained operator recipe for the first bounded run lives in
**[CPU_WORKER_RECIPE.md](CPU_WORKER_RECIPE.md)**.  Its shape: build the CPU-only worker on the **worker host**
(`build-worker`, no GPU toolkit involved) and a CUDA `sm_86` engine on the **coordinator host** (NOT the default
`sm_120` default), forward the worker's loopback port through controller-owned SSH connections, and
start the engine with `--serve --remote-cpu-experts 127.0.0.1:7431`, fixed placement
(`--adapt-every 0`, `--pcie-frac 0`, `STRATA_ROUTE_TAIL_SKIP=0`), short prompts (`--short-read`), and one
text request at a time.  The recipe's checks freeze the model, quant, kernel/window policy, speculation
and KV settings, and its report separates setup from request time and client wait from worker CPU
service.

## Larger setup chunks

The experimental setup optimization, measured small benchmark, bounded compatibility checks and
exact incremental 05/03 build/smoke commands are in
[the setup-chunk handoff](cpu-worker-setup-chunks/README.md). It uses the existing SETUP_ACK cap;
inference frames retain their separate bound. No physical network or full-model speedup is claimed.

## Worker-local sources (default off)

The negotiated local setup candidate can read assigned experts from an operator-configured copy of
the GGUF shards. See [the local source contract](cpu-worker-local/README.md) and
[the operator controller recipe](cpu-worker-local/HERMES.md). The default upload path is unchanged.

## Session-independent residency (default off)

A complete trusted local partition can now survive engine exit for an explicit bounded idle window.
For service-owned retention without idle expiry, explicitly select worker
`--residency-forever` instead of a positive `--residency-ms`, and keep engine
`--remote-cpu-resident`. Zero/omitted TTL still means session-scoped weights.
See [durable residency, status and explicit stop](network-gpu-worker/README.md#explicit-durable-cpu-residency).
See [the lifecycle and compatibility contract](cpu-worker-residency/README.md) and
[the ready-to-use operator recipe](cpu-worker-residency/HERMES.md). Without the new flags, the
session-scoped behavior described above remains in force.

A `STRATA_NATIVE_EXPERTS=OFF` engine excludes the CPU worker integration and refuses its remote-worker
CLI options before resource setup. See the [native-OFF correction and regression handoff](cpu-worker-residency/NATIVE_OFF_HERMES.md).
The [16d065e operator receipt](cpu-worker-residency/evidence/hermes-16d065e-receipt.md) is user-attributed;
its runtime results do not validate the later correction head. HIP is unvalidated and SYCL uses separate
migrated sources; this worker feature is not claimed for SYCL.

## Packed activation requests (automatic for qualified builds)

The fixed CPU group selects packed requests automatically when both native builds carry the same
verified build-pair record and support the version-1 Linux x86-64 Q8_0 format. Replies and original
entry order stay unchanged. Older peers, mismatched or absent records, unsupported formats, and
unsupported per-request inputs use raw F32. Malformed packets and failed deadlines still fail;
they do not trigger a raw retry.

After verifying a native build pair, its builder can embed that record in both artifacts with
`-DSTRATA_CPU_PACKED_QUALIFICATION=<verified 16 lowercase hex ID>`. This is compatibility evidence,
not a user preference. Do not invent an ID from matching source, compiler or CPU names. The cache
value is empty for unqualified builds; no experimental ID is built into the source. Requalify when
the relevant native build changes. This setting does not qualify other platforms or formats.

A normal deployment of those qualified artifacts uses its existing `--remote-cpu-worker` commands
and worker commands, with no packed flags. The sender inherits its build record, probes each peer,
and packs supported requests only after the peer accepts that same record and format version.
Use `--remote-cpu-raw-requests` on the engine or `--raw-requests` on a worker to force raw. These
preferences win even when a qualification override is also present. The existing
`--remote-cpu-packed-qualification` / `--packed-qualification` overrides remain available for a
separately verified pair; operators do not need to repeat them when the artifacts carry the record.
Legacy single-tier, capture and regroup diagnostic engine modes stay raw automatically.

The [wire contract](validation/CPU-PACKED-ACTIVATION-WIRE-20261025.md) describes the original opt-in
implementation. The supplied [live result](validation/CPU-PACKED-LIVE-RESULT-20261026.md) recorded
exact outputs and a request-wall reduction from 43.434 to 38.198 seconds in one raw-then-packed
sample on the retained Linux CUDA/native pair over 1 Gb/s. That sample does not qualify arbitrary
build pairs. New default-selection compiled checks are pending; those earlier results do not test
this change.
