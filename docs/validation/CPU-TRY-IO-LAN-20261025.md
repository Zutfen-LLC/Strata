# Issue #12: bounded 1 Gb/s LAN handoff

Status: **NOT RUN**. Parent/Hermes executes this handoff; no fleet commands or executable tests
run on the coding desktop. Before execution, supply 05's exact firewall backend and INPUT
family/table/chain, and the full retained replay SHA-256. Do not infer a chain from its name.
The firewall block below requires a verified nftables INPUT base chain with policy accept.
If the backend differs, stop and return that one binding for a small replacement block.

Use the existing e5685da112f9e75262176635743b0790586d88c3 replay ELF at
`/home/hermes/strata-pr10-try-io-e5685da1-20261025-05/bin/cpu_rpc_replay`.
The same root holds `shapes.tsv`, `binaries.sha256` and successful loopback receipts.
Transfer that exact ELF and shapes through the existing controller relay to a fresh private 02
root; no host-to-host SSH assumption, rebuild, dependency change or repeated loopback gates.
Verify the entire SHA-256 on both hosts against the retained manifest before starting.

Bindings are source 02 `10.0.0.180`, destination 05 `10.0.0.123`, interface `eno1` on both,
1000 Mb/s full duplex, fixture TCP `17531`. Retain current route/link receipts; require 02's
route to 10.0.0.123 to select eno1 and source 10.0.0.180. The client CLI has no source-bind
option; the server checks the actual accepted source. Never contact or signal resident worker
809314 at 127.0.0.1:17431, replace it, or change shared e145df6e runtime, NIC, route, buffers,
IRQ or unrelated services. No merge, deployment or production socket-default change.

The user approved only a temporary restriction of this fixture tuple to source 02, removed after
execution. The rule below drops other sources only on eno1 to 10.0.0.123:17531. It preserves
other rules and is never persisted. Its unique comment and captured handle identify only our rule.

## Controller sequence

Use existing owned-process/RAM/output guards. Create absent disk-backed `ROOT05` and `ROOT02`
with mode 0700, fresh for this campaign. Export on each host: `ROOT` (that host's root), `BIN`
(exact replay ELF), `ELF_SHA256` (64 hex digits from retained manifest), `T`, `N`, `POLICY`.
Retain the original shape file and manifest in each root. On 05 also export verified
`NFT_FAMILY`, `NFT_TABLE`, `NFT_CHAIN`. Retain commands, logs, exit statuses and cleanup receipts.

Traverse the retained sorted 15 shapes once, index starting at zero. Even indices run `poll`
then `first`; odd indices run `first` then `poll`. Per arm, start the server block on 05 through
the existing controller, wait for the exact LISTENING record, then immediately run the client
block on 02 (the accept deadline is 2000 ms). Do not probe the listener with another connection.
Wait/reap both blocks, require both exit zero and `cleanup.ok`, then continue. Stop on the first
failure; preserve all receipts and reconcile cleanup before any further action. Use the existing
controller's 180-second whole-campaign ceiling with its finally path cancelling/reaping the
active owned server block; the per-arm timeout below remains 15 seconds plus 5 seconds kill grace.
On controller cancellation send TERM to the owned block and allow its cleanup to finish; do not
SIGKILL its shell. If its outcome is uncertain, inspect the recorded PID/rule before retrying.

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
TAG="strata12-$(basename "$ROOT")-${T}-${N}-${POLICY}"
HANDLE= PID=
find_handle() {
  sudo -n nft -a list chain "$NFT_FAMILY" "$NFT_TABLE" "$NFT_CHAIN" > "$D/chain.current"
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
  printf '%s\n' "$rc" > "$D/status"
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
  iifname eno1 ip daddr 10.0.0.123 tcp dport 17531 ip saddr != 10.0.0.180 \
  counter drop comment "$TAG"
HANDLE=$(find_handle)
[[ $HANDLE =~ ^[0-9]+$ ]]
printf '%s\n' "$HANDLE" > "$D/rule.handle"
flags=(); [[ $POLICY = poll ]] || flags+=(--operation-first)
( ulimit -v 2097152
  exec timeout --signal=TERM --kill-after=5s 15s "$BIN" --serve 17531 \
    --route direct-lan --bind-ip 10.0.0.123 --allow-peer-ip 10.0.0.180 \
    --tokens "$T" --row-entries "$N" --accepted-nodelay --exact-rtt \
    --warmup 64 --exchanges 1024 --deadline-ms 2000 "${flags[@]}"
) > "$D/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$D/timeout.pid"
# Controller observes server.log's LISTENING record and starts the matching client now.
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
trap 'rc=$?; printf "%s\n" "$rc" > "$D/status"' EXIT
flags=(); [[ $POLICY = poll ]] || flags+=(--operation-first)
( ulimit -v 2097152
  exec timeout --signal=TERM --kill-after=5s 15s "$BIN" --connect 10.0.0.123 17531 \
    --route direct-lan --tokens "$T" --row-entries "$N" --accepted-nodelay --exact-rtt \
    --warmup 64 --exchanges 1024 --deadline-ms 2000 "${flags[@]}"
) > "$D/client.log" 2>&1
```

## Exact result gates and recording

Relay each server arm directory into the corresponding client arm directory, preserving files.
For every arm require both statuses zero, `cleanup.ok`, no residual fixture listener and no tagged
rule. Require LISTENING at 10.0.0.123:17531 with allowed peer 10.0.0.180 and accepted NODELAY 1,
ACCEPTED actual/allowed peer 10.0.0.180, and CONNECTED host 10.0.0.123 port 17531, route direct-lan.
Keep the distinct server and client statuses when relaying; never overwrite one with the other.

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
