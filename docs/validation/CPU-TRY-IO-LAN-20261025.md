> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Issue #12: bounded 1 Gb/s LAN handoff

Status: **operator LAN campaign terminal PASS; no clear latency winner**. The canonical result
and checked LAN execution acceptance are in [issue #12's operator report](https://github.com/Zutfen-LLC/Strata/issues/12#issuecomment-6403447145).
Do not repeat the completed campaign. No fleet commands or executable tests ran on the coding desktop.
The following results were supplied by the parent/operator; this thread did not independently audit
private raw receipts.

All 15 shapes × 2 policies passed exact bytes/counts and cleanup on the existing 1 Gb/s link.
Try-first won p50 in 8/15 shapes; poll-first won 7/15. First-minus-poll p50 deltas ranged from
-68 to +56 microseconds, with a median about zero and mean -5.9 microseconds across shapes.
These are summaries across shape-specific p50 values, not pooled or traffic-weighted latency.
There was one pair per shape and no noise band. The result establishes neither a LAN latency
winner nor a live model TPS benefit. Production socket defaults remain poll-first.

| Measured helper counters, both roles | Poll-first | Try-first |
| --- | --- | --- |
| Poll calls | 132846 | 70703 |
| Would-block events | Not supplied | 70703 |
| EINTR | 0 | 0 |

Example exact RTT p50 values, microseconds:

| Shape | Poll-first | Try-first |
| --- | --- | --- |
| T1 N1 | 953 | 976 |
| T1 N7 | 1710 | 1665 |
| T2 N7 | 1638 | 1663 |
| T2 N11 | 1876 | 1878 |
| T2 N14 | 2014 | 1976 |

operator confirmed all temporary tagged rules absent, INPUT baseline restored, and zero fixture
listeners/replay processes. Resident worker 809314 on 02 at loopback port 17431 remained alive
and untouched. Controller evidence root:
`${PRIVATE_ARTIFACT_67F107B5}`.
No failed arm was reported; repeat tests and live TPS measurement were not run in this closeout.

## Retained executable handoff (completed; do not rerun)

Verified 05 chain:
`NFT_FAMILY=ip`, `NFT_TABLE=filter`, `NFT_CHAIN=INPUT`, type filter, hook input, priority filter,
policy accept. It currently jumps to ts-input. This is an iptables-nft compatibility-managed
table; retain its marker and every unrelated rule. Use the same narrowly scoped native nft
insertion and captured-handle deletion used by the prior approved fixture. No mutation mechanism
change, iptables rewrite, normalization, flush or persistence is authorized.

Full replay SHA-256, supplied as verified against both manifest and live file:
`696d26871fcca8ce17a103b90646d8894d7ea191f7d51fc9eef47fd455689706`.
The earlier abbreviated `3f2cafc9` report was erroneous; do not use it.

Use the existing e5685da112f9e75262176635743b0790586d88c3 replay ELF at
`${PRIVATE_ARTIFACT_E3C971DE}`.
The same root holds `shapes.tsv`, `binaries.sha256` and successful loopback receipts.
Transfer that exact ELF and shapes through the existing controller relay to a fresh private 02
root; no host-to-host SSH assumption, rebuild, dependency change or repeated loopback gates.
Verify the entire SHA-256 on both hosts against the retained manifest before starting.

Bindings are source 02 `192.0.2.181`, destination 05 `192.0.2.124`, interface `eno1` on both,
1000 Mb/s full duplex, fixture TCP `17531`. Retain current route/link receipts; require 02's
route to 192.0.2.124 to select eno1 and source 192.0.2.181. The client CLI has no source-bind
option; the server checks the actual accepted source. Never contact or signal resident worker
809314 at 127.0.0.1:17431, replace it, or change shared e145df6e runtime, NIC, route, buffers,
IRQ or unrelated services. No merge, deployment or production socket-default change.

The user approved only a temporary restriction of this fixture tuple to source 02, removed after
execution. The rule below drops other sources only on eno1 to 192.0.2.124:17531. It preserves
other rules and is never persisted. Its unique comment and captured handle identify only our rule.

## Controller sequence

Use existing owned-process/RAM/output guards. Create absent disk-backed `ROOT05` and `ROOT02`
with mode 0700, fresh for this campaign. Export on each host: `ROOT` (that host's root), `BIN`
(exact replay ELF), `ELF_SHA256` (64 hex digits from retained manifest), `T`, `N`, `POLICY`.
Retain the original shape file and manifest in each root. On 05 also export verified
`NFT_FAMILY`, `NFT_TABLE`, `NFT_CHAIN`. Retain commands, logs, exit statuses and cleanup receipts.

The exact controller block below visits all 15 shapes once and alternates order. It prestarts
the client shell and waits for CLIENT_READY before starting the server. The waiting client
starts its replay only after a controller-delivered GO, triggered by the server's actual
LISTENING record. Thus no human or agent turnaround or new SSH startup sits inside the
2000 ms accept window. Do not probe the listener. A delayed relay fails the arm; never retry
or widen the deadline. Per-arm limits and firewall cleanup remain in the server block.

Both roles use 64 warmups, 1024 measured exchanges, a 2000 ms I/O deadline, exact RTT, and
accepted/client NODELAY on. Only `--operation-first` changes. Each arm opens one persistent
connection with one outstanding exchange. Production defaults remain poll-first. Actual split
workers differed: remote 05 had accepted NODELAY on; local 02 had it off. This matched transport
fixture does not reproduce that asymmetry or establish a model TPS benefit.

## Server block (05 only)

Run as Bash with the exported bindings. `sudo -n nft` must already be authorized; no interactive
privilege or installation step is implied. A failure to capture/delete the rule or confirm cleanup
is a failed campaign, not permission to flush a chain or ruleset.

```bash
set -euo pipefail
umask 077
: "${ROOT:?}" "${BIN:?}" "${ELF_SHA256:?}"
: "${NFT_FAMILY:?}" "${NFT_TABLE:?}" "${NFT_CHAIN:?}"
[[ $ELF_SHA256 =~ ^[[:xdigit:]]{64}$ ]]
printf '%s  %s\n' "$ELF_SHA256" "$BIN" | sha256sum -c -
[[ $T =~ ^[12]$ && $N =~ ^[0-9]+$ && $POLICY =~ ^(poll|first)$ ]]
D="$ROOT/T${T}-N${N}-${POLICY}"
mkdir -m 700 "$D"
[[ $(basename "$ROOT") =~ ^[A-Za-z0-9_-]+$ ]]
TAG="strata12-$(basename "$ROOT")-${T}-${N}-${POLICY}"
HANDLE= PID=
find_handle() {
  sudo -n nft -a list chain "$NFT_FAMILY" "$NFT_TABLE" "$NFT_CHAIN" > "$D/chain.current" || return
  awk -v tag="\"$TAG\"" '$0 ~ "comment " tag {print $NF}' "$D/chain.current"
}
cleanup() {
  rc=$?; trap - EXIT HUP INT TERM
  if [[ -n $PID ]]; then kill -TERM "$PID" 2>/dev/null || :; wait "$PID" 2>/dev/null || :; fi
  ss -H -ltn 'sport = :17531' > "$D/listeners.after" || rc=1
  [[ ! -s $D/listeners.after ]] || rc=1
  # Also recover the handle if interruption fell between insertion and capture.
  if handles=$(find_handle); then
    for h in $handles; do
      [[ $h =~ ^[0-9]+$ ]] || { rc=1; continue; }
      printf '%s\n' "$h" >> "$D/rule.handles"
      sudo -n nft delete rule "$NFT_FAMILY" "$NFT_TABLE" "$NFT_CHAIN" handle "$h" || rc=1
    done
    if handles=$(find_handle); then [[ -z $handles ]] || rc=1; else rc=1; fi
  else rc=1; fi
  printf '%s\n' "$rc" > "$D/server.status"
  if [[ $rc = 0 ]]; then touch "$D/cleanup.ok"; fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' HUP INT TERM
ss -H -ltn 'sport = :17531' > "$D/listeners.before"
test ! -s "$D/listeners.before"
sudo -n nft -a list chain "$NFT_FAMILY" "$NFT_TABLE" "$NFT_CHAIN" > "$D/chain.before"
test -z "$(find_handle)"
sudo -n nft insert rule "$NFT_FAMILY" "$NFT_TABLE" "$NFT_CHAIN" \
  iifname eno1 ip daddr 192.0.2.124 tcp dport 17531 ip saddr != 192.0.2.181 \
  counter drop comment "$TAG"
HANDLE=$(find_handle)
[[ $HANDLE =~ ^[0-9]+$ ]]
printf '%s\n' "$HANDLE" > "$D/rule.handle"
flags=(); [[ $POLICY = poll ]] || flags+=(--operation-first)
( ulimit -v 2097152
  exec timeout --signal=TERM --kill-after=5s 15s "$BIN" --serve 17531 \
    --route direct-lan --bind-ip 192.0.2.124 --allow-peer-ip 192.0.2.181 \
    --tokens "$T" --row-entries "$N" --accepted-nodelay --exact-rtt \
    --warmup 64 --exchanges 1024 --deadline-ms 2000 "${flags[@]}"
) > "$D/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$D/timeout.pid"
# Stream readiness through the already-open controller connection.
until grep -q '^CPU_RPC_REPLAY_LISTENING_V1 ' "$D/server.log"; do
  if ! kill -0 "$PID" 2>/dev/null; then wait "$PID"; exit 1; fi
  sleep 0.01
done
grep '^CPU_RPC_REPLAY_LISTENING_V1 ' "$D/server.log"
wait "$PID"
PID=
```

## Client block (02 only)

```bash
set -euo pipefail
umask 077
: "${ROOT:?}" "${BIN:?}" "${ELF_SHA256:?}"
[[ $ELF_SHA256 =~ ^[[:xdigit:]]{64}$ ]]
printf '%s  %s\n' "$ELF_SHA256" "$BIN" | sha256sum -c -
[[ $T =~ ^[12]$ && $N =~ ^[0-9]+$ && $POLICY =~ ^(poll|first)$ ]]
D="$ROOT/T${T}-N${N}-${POLICY}"
mkdir -m 700 "$D"
trap 'rc=$?; printf "%s\n" "$rc" > "$D/client.status"' EXIT
printf 'CLIENT_READY\n'
read -r -t 180 go
test "$go" = GO
flags=(); [[ $POLICY = poll ]] || flags+=(--operation-first)
( ulimit -v 2097152
  exec timeout --signal=TERM --kill-after=5s 15s "$BIN" --connect 192.0.2.124 17531 \
    --route direct-lan --tokens "$T" --row-entries "$N" --accepted-nodelay --exact-rtt \
    --warmup 64 --exchanges 1024 --deadline-ms 2000 "${flags[@]}"
) > "$D/client.log" 2>&1
```

## Exact controller block (existing controller host only)

Use the parent's existing controller-to-host execution routes as JSON argv arrays in `EXEC02_JSON`
and `EXEC05_JSON`. Each must accept one final POSIX shell command argument, forward stdin/stdout,
and return the remote command's exit status (for example, the existing SSH argv, without a new
host/key/bypass). These are controller-to-host routes; do not run this on 02 to reach 05. If the
existing relay cannot provide this interface, return its exact invocation to the parent before
execution. No route or credential is guessed here.

Set `HANDOFF` to this pinned document on the controller, `SHAPES` to the relayed retained shapes,
`OUT` to an absent local evidence directory, and export the roots, binary paths, full ELF hash,
and verified NFT bindings named above (the controller pins the SHA, 05 path and NFT tuple). The two host roots must already contain the same retained
`shapes.tsv`; the block checks their bytes. Stage the exact ELF on 02 through the existing relay
before this block. It does no build or model work. Run once with Python 3 under the existing guard.
Only the two Bash blocks above are extracted. The controller has a 180-second campaign limit;
on failure it closes the waiting client's input and allows at most 30 seconds per owned remote
command to finish its bounded arm and cleanup. It does not kill a remote cleanup shell. A stuck
execution route is reported as uncertain with its PID, for reconciliation by the parent.

```python
import base64, csv, json, os, pathlib, re, selectors, shlex, signal, subprocess, time
E = dict(os.environ)
E.update(ELF_SHA256='696d26871fcca8ce17a103b90646d8894d7ea191f7d51fc9eef47fd455689706',
         BIN05='${PRIVATE_ARTIFACT_E3C971DE}',
         NFT_FAMILY='ip', NFT_TABLE='filter', NFT_CHAIN='INPUT')
out = pathlib.Path(E['OUT']); out.mkdir(mode=0o700)
blocks = re.findall(r'^```bash\n(.*?)^```$', pathlib.Path(E['HANDOFF']).read_text(), re.S | re.M)
assert len(blocks) == 2
shape_bytes = pathlib.Path(E['SHAPES']).read_bytes()
shapes = [tuple(map(int, line.split())) for line in shape_bytes.decode().splitlines()]
assert len(shapes) == 15 and shapes == sorted(set(shapes))
assert all(t in (1, 2) and 1 <= n <= t*10 for t, n in shapes)
assert re.fullmatch('[0-9a-fA-F]{64}', E['ELF_SHA256'])
launch = {h: json.loads(E['EXEC'+h+'_JSON']) for h in ('02', '05')}
assert all(isinstance(v, list) and v and all(isinstance(x, str) for x in v) for v in launch.values())
limit = time.monotonic()+180
active = []
def remaining():
    v = limit-time.monotonic()
    if v <= 0: raise TimeoutError('campaign ceiling')
    return v
def command(host, shell):
    return launch[host]+[shell]
def remote(host, code, args):
    r = subprocess.run(command(host, shlex.join(['python3', '-B', '-c', code]+args)),
                       capture_output=True, check=True, timeout=min(20, remaining()))
    return r.stdout
for h in ('02', '05'):
    got = remote(h, 'import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())',
                 [E['ROOT'+h]+'/shapes.tsv'])
    assert got == shape_bytes, (h, 'shape drift')
(out/'shapes.tsv').write_bytes(shape_bytes)
sel = selectors.DefaultSelector()
def start(host, script, t, n, policy):
    env = dict(ROOT=E['ROOT'+host], BIN=E['BIN'+host], ELF_SHA256=E['ELF_SHA256'],
               T=str(t), N=str(n), POLICY=policy)
    if host == '05': env.update({k:E[k] for k in ('NFT_FAMILY','NFT_TABLE','NFT_CHAIN')})
    shell = shlex.join(['env']+[k+'='+v for k,v in env.items()]+['bash','-c',script])
    p = subprocess.Popen(command(host, shell), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    active.append(p); os.set_blocking(p.stdout.fileno(), False)
    sel.register(p.stdout, selectors.EVENT_READ, p)
    return p
def until_line(p, prefix, label):
    buf = b''
    with (out/(label+'.relay.log')).open('ab') as f:
        while True:
            for key, _ in sel.select(min(0.1, remaining())):
                q = key.data
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    sel.unregister(key.fileobj)
                    if q is p: raise RuntimeError(label+' ended before readiness')
                    continue
                if q is p:
                    f.write(data); f.flush(); buf += data
                    if any(line.startswith(prefix) for line in buf.splitlines()): return
            if p.poll() is not None: raise RuntimeError(label+' exited before readiness')
def drain_wait(p, label):
    # Read the remainder to disk while waiting; no bounded pipe can block cleanup.
    try: sel.unregister(p.stdout)
    except KeyError: pass
    os.set_blocking(p.stdout.fileno(), True)
    with (out/(label+'.relay.log')).open('ab') as f:
        # Remote script logs are disk-backed; streamed output is only hash/readiness records.
        data, _ = p.communicate(timeout=min(25, remaining()))
        f.write(data)
    assert p.returncode == 0, (label, p.returncode)
def records(text):
    return [(w[0], dict(x.split('=',1) for x in w[1:]))
            for line in text.splitlines() if (w := line.split()) and w[0].startswith('CPU_RPC_REPLAY_')]
def one(rows, tag, **where):
    found = [v for k,v in rows if k == tag and all(v.get(a)==str(b) for a,b in where.items())]
    assert len(found) == 1, (tag, where, found)
    return found[0]
def gate(rows, t, n, policy):
    req, rep = 16+36+t*70+t*2560*4, 16+20+n*10242
    flag = int(policy == 'first')
    assert sum(k=='CPU_RPC_REPLAY_V1' for k,v in rows)==6
    for role, scopes in [('client',['rtt','send','receive']), ('server',['handler'])]:
        for phase, count in [('warmup',64), ('measure',1024)]:
            for scope in scopes if phase == 'measure' else scopes[:1]:
                v = one(rows,'CPU_RPC_REPLAY_V1',role=role,phase=phase,scope=scope)
                expected = dict(route='direct-lan',accepted_nodelay=1,T=t,entries=n,
                    request_bytes=req,reply_bytes=rep,completed=count,samples=count,failures=0,
                    completed_wire_bytes=count*(req+rep))
                assert all(v[k]==str(x) for k,x in expected.items()), v
        v = one(rows,'CPU_RPC_REPLAY_IO',role=role,phase='measure')
        sent, received = (req,rep) if role=='client' else (rep,req)
        assert all(v[k]==str(x) for k,x in dict(operation_first=flag,row_entries=n,
                   sent_bytes=1024*sent,received_bytes=1024*received).items()), v
        assert all(int(v[k])>=0 for k in ('polls','sends','receives','would_block','interrupted'))
    assert sum(k=='CPU_RPC_REPLAY_IO' for k,v in rows)==2
    assert sum(k=='CPU_RPC_REPLAY_EXACT' for k,v in rows)==1
    v = one(rows,'CPU_RPC_REPLAY_EXACT',role='client',phase='measure',operation_first=flag,samples=1024)
    vals = [int(v[k]) for k in ('p50_ns','p95_ns','p99_ns','max_ns')]
    assert vals == sorted(vals) and vals[0]>=0
    one(rows,'CPU_RPC_REPLAY_LISTENING_V1',address='192.0.2.124',port=17531,
        allowed_peer='192.0.2.181',route='direct-lan',accepted_nodelay=1)
    one(rows,'CPU_RPC_REPLAY_ACCEPTED_V1',peer='192.0.2.181',allowed_peer='192.0.2.181',route='direct-lan')
    one(rows,'CPU_RPC_REPLAY_CONNECTED_V1',host='192.0.2.124',port=17531,route='direct-lan')
def interrupted(sig, frame): raise InterruptedError('controller signal '+str(sig))
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,interrupted)
results = []
try:
    for i,(t,n) in enumerate(shapes):
        for policy in (['poll','first'] if i%2==0 else ['first','poll']):
            label = f'T{t}-N{n}-{policy}'
            c = start('02', blocks[1], t,n,policy)
            until_line(c,b'CLIENT_READY',label+'-02')
            s = start('05', blocks[0], t,n,policy)
            until_line(s,b'CPU_RPC_REPLAY_LISTENING_V1 address=192.0.2.124 port=17531 ',label+'-05')
            c.stdin.write(b'GO\n'); c.stdin.flush(); c.stdin.close(); c.stdin=None
            drain_wait(c,label+'-02'); drain_wait(s,label+'-05')
            rows = []
            for h in ('02','05'):
                d = out/label/h; d.mkdir(parents=True,mode=0o700)
                raw = remote(h, 'import base64,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); '
                    'print(json.dumps({f.name:base64.b64encode(f.read_bytes()).decode() '
                    'for f in p.iterdir() if f.is_file()}))', [E['ROOT'+h]+'/'+label])
                for name,b64 in json.loads(raw).items():
                    assert pathlib.Path(name).name == name
                    (d/name).write_bytes(base64.b64decode(b64,validate=True))
                assert (d/('server.status' if h=='05' else 'client.status')).read_text().strip()=='0'
                if h=='05':
                    assert (d/'cleanup.ok').is_file() and not (d/'listeners.after').read_bytes()
                    tag = 'strata12-'+pathlib.Path(E['ROOT05']).name+f'-{t}-{n}-{policy}'
                    assert 'comment "'+tag+'"' not in (d/'chain.current').read_text()
                rows += records((d/('server.log' if h=='05' else 'client.log')).read_text())
            gate(rows,t,n,policy)
            results.append(dict(T=t,N=n,policy=policy,records=rows))
            (out/'lan-results.json').write_text(json.dumps(results,indent=2))
            print(label+' PASS',flush=True)
    table = []
    for t,n in shapes:
        pair = [next(r for r in results if (r['T'],r['N'],r['policy'])==(t,n,p)) for p in ('poll','first')]
        row = dict(T=t,N=n)
        for p,r in zip(('poll','first'),pair):
            v=one(r['records'],'CPU_RPC_REPLAY_V1',role='client',phase='measure',scope='rtt')
            row[p+'_mean_us']=float(v['mean_us'])
            v=one(r['records'],'CPU_RPC_REPLAY_EXACT')
            for k in ('p50_ns','p95_ns','p99_ns','max_ns'): row[p+'_'+k]=int(v[k])
            for role in ('client','server'):
                v=one(r['records'],'CPU_RPC_REPLAY_IO',role=role)
                for k in ('polls','sends','receives','would_block','interrupted'): row[p+'_'+role+'_'+k]=int(v[k])
        for k in ('mean_us','p50_ns','p95_ns','p99_ns','max_ns'): row['delta_'+k]=row['first_'+k]-row['poll_'+k]
        table.append(row)
    with (out/'comparison.tsv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t'); w.writeheader(); w.writerows(table)
    print('LAN_PASS shapes=15 arms=30 worker_contacts=0 GEN=0',flush=True)
finally:
    for p in active:
        if p.stdin is not None: p.stdin.close(); p.stdin=None
        if p.poll() is None:
            try: p.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                print('UNCERTAIN controller transport PID',p.pid,'remote timeout.pid/rule.handle require reconciliation',flush=True)
```

## Exact result gates and recording

Relay each server arm directory into the corresponding client arm directory, preserving files.
For every arm require both statuses zero, `cleanup.ok`, no residual fixture listener and no tagged
rule. Require LISTENING at 192.0.2.124:17531 with allowed peer 192.0.2.181 and accepted NODELAY 1,
ACCEPTED actual/allowed peer 192.0.2.181, and CONNECTED host 192.0.2.124 port 17531, route direct-lan.
The controller keeps host subdirectories and distinct server/client status files; no receipt is overwritten.

Parse the existing key=value records. Require exactly one warmup RTT client and handler server
record: completed=samples=64, failures=0. Require exactly four measured V1 records: client
rtt/send/receive and server handler, each completed=samples=1024, failures=0. All records must
have the selected T/entries=N, route=direct-lan, accepted_nodelay=1 and exact frame sizes:
`request=16+36+T*70+T*2560*4`, `reply=16+20+N*10242`.
Require completed_wire_bytes = completed * (request+reply), including warmups.
Require exactly one measured client EXACT record, samples=1024, selected operation_first 0/1,
and ordered p50_ns <= p95_ns <= p99_ns <= max_ns. Require exactly two measured IO records,
one per role, selected row_entries=N and operation_first 0/1. Each role's sent_bytes and
received_bytes must equal 1024 times its respective frame size. Binary byte/sequence validation
and zero exit status are mandatory; helper counters cover measured traffic only.

After all 15 pairs pass, write `lan-results.json` retaining every parsed record and a 15-row
comparison table. For each shape show both arms' client RTT mean_us and exact p50/p95/p99/max_ns,
plus first-minus-poll deltas. Show poll/send/receive/would_block/interrupted counts separately
for client and server in both arms. Preserve all raw logs and any failed arm. Do not replace exact
percentiles with bucket bounds, pool shape medians, sum overlapping scopes or subtract host clocks.
One pair per shape gives no noise band. No latency pass threshold and no live TPS claim.

Update issue #12 only after results return, linking the pinned handoff and evidence, separating
passed, failed and not run. Mark the LAN acceptance item only when every exact gate and cleanup
passes. If interrupted, report the exact arm, statuses and listener/rule outcome before retrying.
