# Representative CPU RPC replay: Hermes handoff

This fixture isolates SEW1 framing, persistent TCP exchange, SSH transport and host scheduling without
loading a model or running an expert kernel. It uses `encode_frame`, `encode_infer`, `encode_infer_ack`,
and the production socket full-read/full-write helpers. The server begins its `handler` duration only
after a complete request has arrived, patches the sequence into a fixed reply, and sends it. It does not
decode activations, quantize, allocate scratch, call an expert pool, reduce rows or touch model assets.

The retained corrected-candidate result motivating this check is user-attributed: on engine02 through a
single-hop direct-LAN SSH route to CPUworker05, default and accepted-side TCP_NODELAY measured 7.682 and
7.687 tok/s, while instrumentation measured 7.621 tok/s (about 0.9% cost); tokens were identical. Each
630-token arm made 32,973 RPCs. The instrumented worker reported 1,094.8 us mean complete request and
981.5 us mean pool time; the client reported 2,542 us mean receive/decode and 2,540 us mean finish wait.
These durations overlap. Histogram p50 values are bucket upper bounds, not exact medians, and clocks on
different hosts must never be subtracted.

## Fixture shape and output

The source-valid shapes observed in the retained worker-wide arm are T=1 and T=2, both with K=10 and
H=2560. They are replayed as discrete shapes, never as an averaged frame: T=1 is a 10,362-byte INFER and
10,272-byte INFER_ACK; T=2 is a 20,672-byte INFER and 20,512-byte INFER_ACK. The worker-wide record is
32,973 calls, 35,661 tokens, 305,075 entries, 298,399 distinct experts, 369,379,506 request bytes and
366,223,776 reply bytes, with T1=30,285 and T2=2,688. A separate client record covers only the first
request: 11,136 calls, 12,000 tokens, 102,724 entries, 100,613 distinct experts, 124,299,072 request bytes
and 123,236,352 reply bytes, with T1=10,272 and T2=864. Do not mix those populations or interpret the
first-request client means as full-arm means.

Nine marked entries is an explicitly synthetic representative value near the worker aggregate's 9.25
entries per call; no per-call entry histogram was measured. Entry count is attribution only and does not
change this schema's full T*K arrays. Use `--tokens 1 --entries 9` and `--tokens 2 --entries 9` as separate
arms unless retained per-call shapes justify another entry value. `--request-bytes` and `--reply-bytes`
exist for an exact retained frame record, not for an averaged size; omit them for the source-valid sizes.
Overrides keep real SEW1 headers and fixed synthetic payload bytes, and every output line labels both the
source-derived and selected sizes.

`cpu_rpc_replay` reports warmup separately. The client reports RTT, send and receive scopes. The server
reports `handler`, from complete request receipt through complete reply send. It also reports completed
exchanges, completed wire bytes and failures. Memory stays bounded by one request frame, one reply frame,
one receive payload and fixed log-scale histograms; no per-exchange samples are retained. Accepted-side
TCP_NODELAY is default-off and selected only with `--accepted-nodelay`. The client socket keeps the
production client's existing TCP_NODELAY behavior.

## Build and local gates

Build the same immutable commit on engine02 and CPUworker05 using only each host's existing pinned
toolchain, dependency cache and Unix Makefiles build directory. A missing compatible retained cache is a
blocker; do not install, refresh dependencies, change generator or borrow a live model directory. Direct
runtime output to a new candidate-specific directory. This target is CPU-only and needs no model, CUDA
execution or large memory allocation.

```sh
: "${SRC:?clean checkout at the exact candidate SHA}"
: "${BUILD:?retained compatible build cache on this host}"
: "${BIN:?new candidate-specific binary directory}"
test -z "$(git -C "$SRC" status --porcelain)"
test "$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD/CMakeCache.txt")" = "Unix Makefiles"
test ! -e "$BIN" && mkdir -m 700 "$BIN"
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN"
cmake --build "$BUILD" --parallel 1 --target cpu_rpc_replay
"$BIN/cpu_rpc_replay" --self-check
"$BIN/cpu_rpc_replay" --loopback --warmup 256 --exchanges 4096 \
  --tokens 1 --entries 9 --deadline-ms 10000
"$BIN/cpu_rpc_replay" --loopback --warmup 256 --exchanges 4096 \
  --tokens 2 --entries 9 --deadline-ms 10000
```

Run those commands separately on02 and on05 and retain stdout, stderr, exit status, source SHA and binary
SHA-256. The self-check must report fragmented, malformed, truncated and finite-timeout cases as passed.
The loopback run must report exactly 256 completed warmups and 4,096 completed measured exchanges on both
roles, zero failures, and the matching source-valid request-plus-reply size times completed exchanges. Each arm is
closed-loop and starts the next request as soon as its prior reply completes; do not pace it to the
roughly 400 calls/s observed in the model run, because that would impose the bottleneck being measured.
There is no latency pass threshold.

## External single-hop arm

The fixture server binds only `127.0.0.1`. The current direct key permits only the live05 service port
17431. A separate fixture port is not currently authorized. Before this arm, the parent must explicitly
approve and install a narrowly scoped security rule for one separate fixture port. Do not point the
fixture at, bind over, replace or otherwise use live port 17431. Do not use a tailnet fallback. Until that
approval is recorded, stop after the local gates.

After approval, bind `FIXTURE_PORT05` to that separately approved port and have the parent-managed direct
SSH tunnel expose it as `FORWARD_PORT02` on engine02. Port 17531 was reported free on both hosts, but that
does not authorize it; use it only if the pending narrowly scoped approval names it. No credential, key
path or production address belongs in source or retained logs. Start the owned fixture server on05, then
the client on02 with identical shape, count and deadline arguments. Run the matched pair for T=1, then
restart the owned server and repeat with `--tokens 2`; do not average the frame sizes into a third shape:

```sh
"$BIN05/cpu_rpc_replay" --serve "$FIXTURE_PORT05" --warmup 256 --exchanges 4096 \
  --tokens 1 --entries 9 --deadline-ms 10000

"$BIN02/cpu_rpc_replay" --connect 127.0.0.1 "$FORWARD_PORT02" --warmup 256 --exchanges 4096 \
  --tokens 1 --entries 9 --deadline-ms 10000
```

The server owns its listener and exits after the configured warmup plus measured exchanges; a malformed,
truncated or timed-out exchange fails the arm and closes its descriptors. Confirm it exits and leaves no
listener or thread. For the optional accepted-NODELAY arm, restart only this fixture server with
`--accepted-nodelay` and add the same flag to the client command so both reports are labeled. This never
requires stopping or signaling an existing model worker.

Acceptance is correctness and completeness, not an arbitrary speed number: both roles report the exact
configured shape, counts and completed bytes; failures are zero; the client validates every reply's
session and sequence; and the owned server exits cleanly. Compare the measured client RTT/send/receive
and server handler distributions with the matched loopback arms descriptively. Do not add the scopes,
subtract clocks between hosts, or claim the synthetic handler reproduces expert compute or CPU scheduling.
Preserve every arm, including slower results. No result authorizes production socket tuning, worker
replacement, merge or deployment.
