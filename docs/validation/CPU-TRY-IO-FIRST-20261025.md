# Try I/O before poll: diagnostic A/B

Tracks [#12](https://github.com/Zutfen-LLC/Strata/issues/12), within parent #11. Packed activations
remain separate in #14. No scatter/gather, framing-copy change, chunk streaming, prefix reduction,
compression integration, worker replacement or GEN is included. Consumer 1 Gb/s remains the baseline.

## Supplied terminal loopback result

Hermes reports PASS at `e5685da112f9e75262176635743b0790586d88c3`: five focused tests in 1.89 seconds,
15 retained shapes, both policies, 1024 measured exchanges per shape/policy, all bytes and counts exact.
Try-first RTT p50 improved in 14 of 15 shapes by 1.6–30.0 microseconds; T2-N12 regressed by
1.4 microseconds. The median ACROSS the 15 shape-specific p50 values was 111.0 microseconds poll-first
versus 94.9 try-first. This is not a pooled latency statistic or a traffic-weighted model result.

| Both-role measured helper counters | Poll-first | Try-first |
| --- | --- | --- |
| Poll calls | 96034 | 30732 |
| Receive attempts | 65314 | 97496 |
| Send attempts | 30720 | 30720 |
| Would-block events | Not supplied | 30732 |
| EINTR | 0 | 0 |

The path exchanges some polls for additional nonblocking receive attempts; it does not eliminate
all readiness waits. Counters exclude connection setup and warmup, and are not system-wide tracing.
One pair per shape establishes no noise band, LAN result or live TPS benefit. Production defaults
remain poll-first. The shared e145df6e runtime and resident worker were untouched. These are supplied
Hermes results, not execution or independent private raw-receipt auditing by this coding thread.
The loopback recipe below is retained evidence; do not repeat its successful gates.

Existing production socket entrypoints still select poll-first. An explicit per-call diagnostic policy
attempts nonblocking send/receive first, polls only after would-block, and retries EINTR without
renewing the absolute full-I/O budget. Cancellation is checked before each attempt and throughout the
existing 50 ms cancellable poll loop. Partial I/O, EOF, error returns and zero-length behavior remain.
No environment variable changes production behavior. Diagnostic counters belong to each I/O thread.
Windows keeps its existing nonblocking setup/select implementation; Windows execution is NOT RUN.

Independent static Standards and Spec reviews completed. Standards found that broad timeout-only
assertions could miss budget renewal; the corrected trickle/EINTR cases deliver ALL requested bytes
after the original deadline, so renewal would wrongly succeed. Cancellation observations precede
thread cleanup. The correction was re-reviewed; no remaining concrete finding. Spec found no blocker.
No compiler, executable test, replay or fleet operation ran in this coding thread.

The existing `cpu_rpc_replay` gains `--operation-first`, `--row-entries N` and bounded `--exact-rtt`.
It prebuilds production-schema exact-row replies at real retained sizes, with deterministic synthetic
contents. This is a transport-size replay, not native expert computation or real-row compression.
Every received payload byte and sequence must match; RTT is captured before the client's byte check.
Both arms include the same server validation and instrumentation. Header/payload reads stay separate,
as in the existing path; no fused frame change. One persistent TCP connection is reused per arm,
with 64 warmups excluded and 1024 unpaced, single-outstanding measured exchanges. Both peers use the
selected I/O policy. Accepted/client NODELAY is fixed on in both arms, matching the CPU transport.

Measured client RTT has exact nearest-rank p50/p95/p99 plus the existing mean/max. Existing send,
receive and server-handler histograms retain their coarse bucket-upper-bound labels. Helper counters
report poll/select, send/recv attempts, would-block, EINTR and completed bytes, excluding connect/accept
and warmup. They are not a system-wide syscall trace. No installed profiling tools are needed.
Scope timings overlap and cannot be summed; no clocks are subtracted across machines.

## Fixed loopback handoff

Use the PREVIOUS compression benchmark's isolated source/build receipts, not the shared runtime
checkout/cache. They are clean at 74ffe391. No old model worker or executable is launched or signaled.
Private bindings:

| Binding | Meaning |
| --- | --- |
| `SRC`, `BUILD` | Previous isolated compression source at 74ffe391 and its compatible CPU cache |
| `PIN` | Exact new candidate SHA |
| `RUN_ROOT` | New absolute disk-backed directory, absent, with an existing parent |
| `REAL_LOG`, `REAL_METRICS` | Previous successful real-row compression.log and metrics.json |
| `BLOCK` | Fresh private temporary shell file outside RUN_ROOT |

Fetch PIN, mechanically extract the marked block, refuse an empty extraction, then execute it once.
Export bindings first. No new driver authoring, packages, settings or security changes.

```sh
set -euo pipefail
git -C "$SRC" fetch origin "$PIN"
git -C "$SRC" show "$PIN:docs/validation/CPU-TRY-IO-FIRST-20261025.md" |
  sed -n '/^# BEGIN TRY_IO05$/,/^# END TRY_IO05$/p' > "$BLOCK"
test -s "$BLOCK"
bash "$BLOCK"
```

```sh
# BEGIN TRY_IO05
set -euo pipefail
umask 077
test ! -e "$RUN_ROOT"
test "$(git -C "$SRC" rev-parse HEAD)" = 74ffe391b6ec32f3e61aab6ce5386d779f0cf007
test -z "$(git -C "$SRC" status --porcelain)"
test "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")" = "$SRC"
grep -Eq '^STRATA_ENABLE_CUDA:BOOL=OFF$' "$BUILD/CMakeCache.txt"
mkdir -m 700 "$RUN_ROOT"
sha256sum "$REAL_LOG" "$REAL_METRICS" > "$RUN_ROOT/input-hashes.sha256"
git -C "$SRC" checkout --detach "$PIN" > "$RUN_ROOT/source.log" 2>&1
test "$(git -C "$SRC" rev-parse HEAD)" = "$PIN"
timeout --signal=TERM --kill-after=5s 60s cmake -S "$SRC" -B "$BUILD" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$RUN_ROOT/bin" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE="$RUN_ROOT/bin" > "$RUN_ROOT/configure.log" 2>&1
timeout --signal=TERM --kill-after=5s 300s cmake --build "$BUILD" --parallel 4 \
  --target worker_socket_test worker_socket_operation_test cpu_rpc_replay > "$RUN_ROOT/build.log" 2>&1
sha256sum "$RUN_ROOT/bin/worker_socket_test" "$RUN_ROOT/bin/worker_socket_operation_test" \
  "$RUN_ROOT/bin/cpu_rpc_replay" > "$RUN_ROOT/binaries.sha256"
( ulimit -v 2097152
  timeout --signal=TERM --kill-after=5s 120s ctest --test-dir "$BUILD" --output-on-failure --no-tests=error \
    -R '^(worker_socket_test|worker_socket_operation_test|cpu_rpc_replay|cpu_rpc_replay_rows_poll|cpu_rpc_replay_rows_first)$' \
    > "$RUN_ROOT/check.log" 2>&1 )
grep -Eq '100% tests passed, 0 tests failed out of 5' "$RUN_ROOT/check.log"
python3 -B - <<'PY'
import json,os,pathlib
e=os.environ; record={}
for line in pathlib.Path(e['REAL_LOG']).read_text().splitlines():
    if line.startswith('COMPRESSION_REFERENCE '):
        v=dict(s.split('=',1) for s in line.split()[1:]); record[int(v['record'])]=int(v['T'])
metrics=json.loads(pathlib.Path(e['REAL_METRICS']).read_text()); assert metrics['done']['ok']=='1'
assert len(record)==16 and metrics['done']['frames']=='32' and metrics['done']['codecs']=='5'
raw=[v for v in metrics['frames'] if v['codec']=='raw_copy']; assert len(raw)==32
shapes=set()
for v in raw:
    t=record[int(v['record'])]; n=int(v['raw_bytes'])-20
    assert t in (1,2) and n>0 and n%10242==0
    count=n//10242; assert 1<=count<=t*10
    shapes.add((t,count))
r=pathlib.Path(e['RUN_ROOT'])
with (r/'shapes.tsv').open('x') as f:
    for t,count in sorted(shapes): f.write(f'{t} {count}\n')
print('TRY_IO_SHAPES_PASS shapes='+str(len(shapes)))
PY
export RUN_ROOT
timeout --signal=TERM --kill-after=5s 180s bash -euo pipefail -c '
  index=0
  while read -r tokens entries; do
    # Alternate which policy runs first across shapes. No parameter tuning or repeated campaign.
    if (( index % 2 == 0 )); then order="poll first"; else order="first poll"; fi
    for policy in $order; do
      flags=(); if test "$policy" = first; then flags+=(--operation-first); fi
      ( ulimit -v 2097152
        timeout --signal=TERM --kill-after=5s 15s "$RUN_ROOT/bin/cpu_rpc_replay" --loopback \
          --tokens "$tokens" --row-entries "$entries" --accepted-nodelay --exact-rtt \
          --warmup 64 --exchanges 1024 --deadline-ms 2000 "${flags[@]}" \
          > "$RUN_ROOT/T${tokens}-N${entries}-${policy}.log" 2>&1 )
    done
    index=$((index+1))
  done < "$RUN_ROOT/shapes.tsv"
'
python3 -B - <<'PY'
import json,os,pathlib
r=pathlib.Path(os.environ['RUN_ROOT']); output=[]
shapes=[tuple(map(int,s.split())) for s in (r/'shapes.tsv').read_text().splitlines()]; assert shapes
for t,n in shapes:
    pair=[]
    for policy,flag in [('poll','0'),('first','1')]:
        lines=(r/f'T{t}-N{n}-{policy}.log').read_text().splitlines(); rows=[]
        for line in lines:
            if line.startswith('CPU_RPC_REPLAY_'):
                rows.append((line.split()[0],dict(v.split('=',1) for v in line.split()[1:])))
        measured=[v for tag,v in rows if tag=='CPU_RPC_REPLAY_V1' and v['phase']=='measure']
        expected_request=16+36+t*70+t*2560*4; expected_reply=16+20+n*10242
        assert len(measured)==4 and all(v['completed']=='1024' and v['failures']=='0' and
            int(v['request_bytes'])==expected_request and int(v['reply_bytes'])==expected_reply and
            int(v['completed_wire_bytes'])==1024*(expected_request+expected_reply) for v in measured)
        exact=[v for tag,v in rows if tag=='CPU_RPC_REPLAY_EXACT']; assert len(exact)==1
        assert exact[0]['samples']=='1024' and exact[0]['operation_first']==flag
        io=[v for tag,v in rows if tag=='CPU_RPC_REPLAY_IO']; assert len(io)==2
        for v in io:
            assert v['operation_first']==flag
            sent,received=(expected_request,expected_reply) if v['role']=='client' else (expected_reply,expected_request)
            assert int(v['sent_bytes'])==1024*sent and int(v['received_bytes'])==1024*received
        pair.append(dict(policy=policy,metrics=rows))
    output.append(dict(T=t,entries=n,arms=pair))
with (r/'try-io-results.json').open('x') as f: json.dump(output,f,indent=2)
print('TRY_IO05_PASS shapes='+str(len(shapes))+' policies=2 worker_contacts=0 GEN=0')
PY
# END TRY_IO05
```

Stop on any nonzero command, timeout, failed byte/sequence/count gate or changed asset binding. Preserve
failure evidence; do not rerun successful gates or tune socket buffers, affinity, pacing or payloads.
The socket regression covers ready I/O, stalled send/receive, partial progress, cancellation with infinite
budgets, EINTR, EOF, peer errors and full fragmented transfers under BOTH policies. Other backends and
all executable tests are NOT RUN here. Expected loopback work under a minute after a small incremental
CPU build; the entire replay set has a 180-second external ceiling and each arm a 15-second ceiling.

## LAN handoff

The current bounded handoff is [CPU-TRY-IO-LAN-20261025.md](CPU-TRY-IO-LAN-20261025.md).
It supersedes the older LAN boundary and stale address/permission examples in
`docs/cpu-rpc-replay/HERMES.md` for issue #12 only. The supplied current tuple is
02 10.0.0.180 to 05 10.0.0.123 over eno1, TCP 17531. The user approved a temporary
restriction of that fixture tuple to source 02, followed by removal. Execution still
requires the exact firewall chain binding and retained ELF hash. LAN and TPS results
remain NOT RUN; the supplied loopback evidence above is unchanged.
