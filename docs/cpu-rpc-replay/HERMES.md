# Representative CPU RPC replay: Hermes handoff

This fixture isolates SEW1 framing, persistent TCP exchange, transport routing and host scheduling without
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

The completed replay at `f6cfda9968f463453174f234bd73426cc2735fe1` used the identical binary on both
hosts. Engine02 loopback means were 66.5 us for T=1 and 87.4 us for T=2. The single-hop physical-LAN SSH
means to CPUworker05 were 1,604.9 us and 1,887.3 us; accepted-side TCP_NODELAY measured 1,619.1 us and
1,867.3 us. Every arm completed 256 warmups and 4,096 measured exchanges with zero failures. The next
comparison changes only the route to direct physical-LAN TCP; the frame shapes, entries, counts, one
persistent connection, one outstanding request and unpaced closed loop remain fixed.

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

The merge-review correction uses bounded numeric IPv4 connection setup and a nonblocking listener for
both replay routes. Hostnames are refused before DNS. Its self-check also covers the selected listener
flags, a reset queued peer and hostname refusal. Rebuild/run only `cpu_rpc_replay` for this fixture-only
change; the current [focused follow-up](../cpu-worker-lan/HERMES.md) names the other affected CLI gates.

Build the immutable commit once on CPUworker05 using its existing pinned toolchain, dependency cache and
Unix Makefiles build directory. Engine02 has no retained compatible dependency cache, so do not configure
or build there. A missing compatible cache on CPUworker05 is a blocker; do not install, refresh dependencies,
change generator or borrow a live model directory. Direct runtime output to a new candidate-specific
directory. This target is CPU-only and needs no model, CUDA execution or large memory allocation.

```sh
: "${SRC:?clean checkout at the exact candidate SHA}"
: "${BUILD:?retained compatible build cache on this host}"
: "${BIN05:?new candidate-specific binary directory on CPUworker05}"
test -z "$(git -C "$SRC" status --porcelain)"
test "$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD/CMakeCache.txt")" = "Unix Makefiles"
test ! -e "$BIN05" && mkdir -m 700 "$BIN05"
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN05"
cmake --build "$BUILD" --parallel 1 --target cpu_rpc_replay
"$BIN05/cpu_rpc_replay" --self-check
"$BIN05/cpu_rpc_replay" --loopback --warmup 256 --exchanges 4096 \
  --tokens 1 --entries 9 --deadline-ms 10000
"$BIN05/cpu_rpc_replay" --loopback --warmup 256 --exchanges 4096 \
  --tokens 2 --entries 9 --deadline-ms 10000
if "$BIN05/cpu_rpc_replay" --serve 0 --route direct-lan; then exit 1; fi
if "$BIN05/cpu_rpc_replay" --serve 0 --route direct-lan \
  --bind-ip 0.0.0.0 --allow-peer-ip 127.0.0.1; then exit 1; fi
```

Retain stdout, stderr, exit status, source SHA and the CPUworker05 binary SHA-256. Transfer that exact
executable to a new candidate-specific `BIN02` directory on engine02 through the existing parent-managed
file-transfer path. Verify its SHA-256 equals the CPUworker05 value before execution; do not rebuild or
install dependencies on engine02. Run the same self-check and T=1/T=2 loopback commands there,
substituting `BIN02` for `BIN05`. The self-check must report fragmented, malformed, truncated and
finite-timeout cases as passed.
The loopback run must report exactly 256 completed warmups and 4,096 completed measured exchanges on both
roles, zero failures, and the matching source-valid request-plus-reply size times completed exchanges.
Each arm is closed-loop and starts the next request as soon as its prior reply completes; do not pace it to
the roughly 400 calls/s observed in the model run, because that would impose the bottleneck being measured.
The two negative commands must fail before opening a listener because the peer pair is missing or the bind
is a wildcard. There is no latency pass threshold.

## Direct physical-LAN arm

The default remains loopback-only. A direct-LAN server is possible only when `--route direct-lan`,
`--bind-ip` and `--allow-peer-ip` are all present. Both addresses must be numeric, selected unicast IPv4
addresses; wildcard, multicast and loopback addresses are rejected. The verified values are
`LAN_IP05=10.0.0.124` and `LAN_IP02=10.0.0.180`. The listener binds only `LAN_IP05`,
and immediately after accept it compares the kernel-reported peer address with `LAN_IP02` before reading a
frame. A peer mismatch closes the owned connection and fails the fixture. The selected addresses and route
are printed once and included in aggregate output.

This source-address check is a restriction, not cryptographic authentication. It does not protect against
source spoofing, a compromised allowed host, or another process on the allowed host. Host firewall policy
remains the primary exposure control. Hermes found CPUworker05 INPUT default-accept with only Tailscale
chains; that is evidence for a proposed restriction, not authorization to change it. Obtain approval for
the exact temporary rule before launching the server. It should admit only `10.0.0.180` to the selected
fixture port on `10.0.0.124`. Remove it after the arm and verify removal. Do not bind a wildcard address or
use a public, tailnet or SSH-forward route for this comparison.

The temporary SSH permission for port 17531 was removed, and the original 17431-only key restriction was
restored. The raw-LAN arm does not need or authorize any SSH permission change. Never point this fixture at,
bind over, replace or otherwise use live port 17431. Both full CPU arenas and their worker processes remain
untouched. No credential, key path or production address belongs in source or retained logs.

After the separate firewall approval, set `FIXTURE_PORT05` to the exact approved port and use only the
verified addresses above. First run one non-measurement access check: start the server with
`--allow-peer-ip "$LAN_IP05"`,
then attempt one exchange from engine02. Both processes must fail, the server must identify `LAN_IP02` as the
disallowed actual peer, and the listener must exit. This checks the post-accept restriction; it is not a
latency arm. Then start the synthetic fixture server with the approved peer on CPUworker05 and connect
directly from engine02. Run the matched pair for T=1, then restart the owned server and repeat with
`--tokens 2`; do not average the sizes into a third shape:

```sh
LAN_IP05=10.0.0.124
LAN_IP02=10.0.0.180
: "${FIXTURE_PORT05:?separately approved temporary fixture port}"
"$BIN05/cpu_rpc_replay" --serve "$FIXTURE_PORT05" --warmup 256 --exchanges 4096 \
  --tokens 1 --entries 9 --deadline-ms 10000 --route direct-lan \
  --bind-ip "$LAN_IP05" --allow-peer-ip "$LAN_IP02"

"$BIN02/cpu_rpc_replay" --connect "$LAN_IP05" "$FIXTURE_PORT05" --warmup 256 --exchanges 4096 \
  --tokens 1 --entries 9 --deadline-ms 10000 --route direct-lan
```

The server owns its listener and exits after the configured warmup plus measured exchanges; a malformed,
truncated or timed-out exchange fails the arm and closes its descriptors. Confirm it exits and leaves no
listener or thread. No additional delay, pacing or accepted-NODELAY variant is needed for this initial
direct-LAN comparison. This never requires stopping or signaling an existing model worker.

Acceptance is correctness and completeness, not an arbitrary speed number: both roles report the exact
configured shape, counts and completed bytes; failures are zero; the client validates every reply's
session and sequence; and the owned server exits cleanly. Compare the measured client RTT/send/receive
and server handler distributions with the retained loopback and SSH arms descriptively. Confirm every
aggregate says `route=direct-lan`, and retain the listening and accepted records naming the selected bind
and peer. The direct-LAN route necessarily removes SSH and changes the server bind; label both facts. Do
not add the scopes, subtract clocks between hosts, or claim the synthetic handler reproduces expert compute or
CPU scheduling. Preserve every arm, including slower results. No result authorizes production socket
tuning, worker replacement, merge or deployment.
