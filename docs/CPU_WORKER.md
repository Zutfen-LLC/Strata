# The resident CPU-expert worker (`strata-expert-worker`)

The worker is a small loopback service that holds one model's CPU-owned expert blobs in RAM and computes
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
listener binds **127.0.0.1 only** - there is no flag to bind a public address. To reach it from another
machine, forward a port over an SSH tunnel you own (for example
`ssh -L 7431:127.0.0.1:7431 inferswarm05`) and point the client at the tunnel's local end.

Options: `--port P` (0 picks a free port and prints it), `--memory-budget-mib N` (default 131072),
`--workers N` (CPU expert-pool worker threads; 0 = the pool's default plan for the machine),
`--deadline-ms N` (budget per full socket send/read, default 10000), `--verbose` (one identity/status line per
request; still no payloads are logged).

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
- One connection, one outstanding request; no pipelining, no concurrency, no auth (session identity is
  correlation, not authentication - the loopback bind plus an owned tunnel is the boundary).
- No engine integration in the worker itself: the engine side (`--remote-cpu-experts`, above) lives in
  `ExpertWorkerTier` + the dispatch/verifier glue and keeps this service untouched.
- Windows: this correction's nonblocking/select path was not built or run there; validation is Linux
  only (the e2e and socket-bound tests are POSIX-only).
- The setup manifest declares one layout per model, so the blob LENGTH is derived once for the whole
  partition; per-blob digests ride each UPLOAD frame and are enforced when a slot completes.  A manifest
  with per-slot digest tables would be a protocol extension if a pack ever needs it.

## The engine mode (inferswarm #349): `--remote-cpu-experts HOST:PORT`

The engine side is **opt-in and default-off**: `strata generate --serve --remote-cpu-experts 127.0.0.1:7431`
(`--remote-cpu-deadline-ms N` budgets each full socket send/read; default 10000).  With the flag off, every path below is
untouched. Header and payload reads have separate budgets; connection establishment and native kernel
execution are not preempted by this setting. The pieces:

- **The tier** (`strata::worker::ExpertWorkerTier`, `strata_worker_tier`, CPU-only like the service):
  owns the client, the frozen ownership bitmap, one in-flight request on its own thread, and the failure
  latch.  `tests/worker/expert_tier_test` runs its whole lifecycle against the real in-process service.
- **The partition is derived, never configured**: at startup, after CUDA0's cache, every stage cache and
  every helper tier (`--expert-cache-device1..3`) is filled, the complement - every expert no GPU tier
  holds - becomes the worker's.  Every routed expert keeps exactly one executing owner.  Setup streams
  each owned blob from the expert source through ONE blob-sized staging buffer (no expert RAM mirror on
  the engine host; the model files stay on its disk), and inference may not start before the worker has
  verified the whole set (`READY`).
- **In a verify window** (`expert_pool_dispatch_multi`), after the GPU's plan is published, the peer
  launched and the helpers begun - and before the local pool's own work - the worker entries' frame goes
  out (activations, ids, routing weights, ownership mask; nothing for a window with no owned entry).  The
  round trip overlaps the local CPU jobs; `finish` joins before anything downstream may commit.
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

## The two-host recipe (inferswarm05 + inferswarm03)

A self-contained Hermes recipe for the first bounded run lives in
**[CPU_WORKER_RECIPE.md](CPU_WORKER_RECIPE.md)**.  Its shape: build the CPU-only worker on **inferswarm05**
(`build-worker`, no GPU toolkit involved) and a CUDA `sm_86` engine on **inferswarm03** (NOT the default
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
