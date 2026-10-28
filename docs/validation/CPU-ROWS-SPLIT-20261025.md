# Fixed row-mode local / split / local comparison

Parent-supplied terminal update: all nine outputs and first-logit rows matched bitwise, the full02
worker is retained, and05 stopped cleanly. Timing/scaling conclusions require the separate retained
report. Do not repeat this comparison. The next queued test is [PLE direct/RAM/direct](PLE-RAM-20261025.md),
only after fresh terminal/full-worker checks.

Planning/handoff only. Runtime code remains e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb.
No new implementation, build campaign or sequencer is introduced. Reuse the optimized engine/worker
and passed CPU/CUDA/native-OFF gates from the completed local comparison. The parent dispatches operator;
no fleet action or executable test was performed by the coding thread. No merge is included.

Supplied optimized local result: both modes matched A0 first logits/213 tokens/102724 entries,11136
calls/0 failures and unchanged bytes. Row decode15.6309s versus legacy14.369s; prior row18.3926s.
These single loopback samples do not predict distributed throughput. Cold READY67.3s is separate.
The original split failure remains uncleared until this new comparison meets all frozen references.

All arms use entry rows, the same optimized engine/worker bytes, Q8 assets, primary960/helper1378,
12 workers/4 loaders, original CPU/GPU masks, turbo, OMP policy, monitors and approved direct LAN/control
routes. A0/A1 own all22238 CPU experts locally. B alternates the SAME complement globally in layer/
expert order, local endpoint first:11119 experts/58067865600B per host. No full arena is loaded on05.
Do not keep the full02 arena alongside its local half. Do not change scheduling, clocks, networking,
permissions, protocol, numerical limits or benchmark code. No generation warmups or tuning arms.

The existing driver runs all three original GEN256 requests in each warm arm. Token/entry references
are enforced by that driver; first-logit metrics and bit equality are checked after each completed arm,
not by a new per-arm gate. Preserve all results, including the third request's prior divergence.

## Bindings and mechanical execution

Parent-provided private bindings, not public paths:

| Role | Bindings |
| --- | --- |
| 02 | SRC; ENGINE_INPUT/WORKER_INPUT (validated optimized immutable ELFs); CURRENT_LOAD/CURRENT_LEGACY/CURRENT_ROWS (completed optimized local bundles); SPLIT_TEMPLATE (original passed peer/control config); A0_SUMMARY/A0_RAW0/A0_RAW1/A0_RAW2; LIVE_ROOT (new short disk-backed root); PEER_ROOT (existing forced-command05 bundle path, recreated after verified stopped-root archival); WORKER05 (copied optimized ELF's05 path) |
| 05 | SRC; PRIVATE_CONFIG (exact generated split source config transferred through existing channel); WORKER05; WORKER_SHA (mechanically copied full optimized hash); FIXTURE05 (retained compatible worker_local_test); FIXTURE_ROOT (new) |

Both source checkouts stay at e145df6e. Fetch the documentation SHA, then source each marked block in
the SAME bash session as in the [live recipe](CPU-REPLY-CODEC-LIVE-20261025.md). Do not transcribe blocks
or pipe them into a child bash. Only named block selection and the private bindings vary.
WORKER05 is the already copied ELF; FIXTURE_ROOT and the recreated PEER_ROOT are separate directories, not
the directory already created to receive that ELF. Require the retained FIXTURE05 executable; do not
search for another build or silently compile a replacement.

02 preparation and05 staging/smoke must succeed before A0 or current resident release. Transfer only
the exact CPU worker ELF and small private config/manifest as needed through existing approved paths;
reuse existing model/native files on05. No full model copy, new transport or package installation.

```sh
# BEGIN SPLIT_PREP02
set -euo pipefail
umask 077
test "$(git -C "$SRC" rev-parse HEAD)" = e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb
test ! -e "$LIVE_ROOT"
mkdir -m 700 "$LIVE_ROOT"
cp -- "$ENGINE_INPUT" "$LIVE_ROOT/engine-optimized"
cp -- "$WORKER_INPUT" "$LIVE_ROOT/worker-optimized"
sha256sum "$LIVE_ROOT/engine-optimized" "$LIVE_ROOT/worker-optimized" > "$LIVE_ROOT/optimized.sha256"
export STAGE_ROOT="$LIVE_ROOT/s1"
# END SPLIT_PREP02
```

Run SPLIT_GENERATE02 after preparation. For the already completed ELF-copy phase, skip SPLIT_PREP02:
verify optimized.sha256, export STAGE_ROOT=LIVE_ROOT/s1, and run only this block. An existing s1 must
be empty; existing configs are never overwritten. Bind CURRENT_LOAD to the fresh recovered full worker.

```sh
# BEGIN SPLIT_GENERATE02
set -euo pipefail
sha256sum -c "$LIVE_ROOT/optimized.sha256"
PYTHONPATH="$SRC/tools/validation-harness/harness" python3 -B - <<'PY'
import copy,json,os,pathlib
import bundle,run
E=os.environ;r=pathlib.Path(E['LIVE_ROOT']);s=pathlib.Path(E['STAGE_ROOT'])
if s.exists():assert s.is_dir() and not any(s.iterdir()),'preserve existing config generation'
else:s.mkdir(mode=0o700)
oldload=bundle.verify(pathlib.Path(E['CURRENT_LOAD']));base=bundle.verify(pathlib.Path(E['CURRENT_ROWS']))
assert bundle.sha(oldload['assets']['worker'])==bundle.sha(r/'worker-optimized')
assert bundle.sha(base['assets']['engine'])==bundle.sha(r/'engine-optimized')
assert base.get('cpu_route')=='group' and base['argv'].count('--remote-cpu-entry-rows')==1
assert base['worker_bytes']==116135731200 and len(base['arms'])==1
policy=base['arms'][0];assert policy['workers']==12 and policy['loader_threads']==4
template=json.loads(pathlib.Path(E['SPLIT_TEMPLATE']).read_text())
peers=copy.deepcopy(template['worker_peers']);assert len(peers)==2 and peers[0]['local'] and not peers[1].get('local')
p=peers[1];remote=p['config'];previous=remote['run_root']
assert E['PEER_ROOT']==previous,'forced-command guardian path must remain unchanged'
# Preserve the exact SSH argv: authorized_keys supplies the fixed remote command, not an argv tail.
p['asset_bindings']={'worker':'worker','manifest':'manifest'}
remote['run_root']=E['PEER_ROOT'];remote['assets']['worker']=E['WORKER05'];remote['worker_argv'][0]='@worker'
argv=remote['worker_argv'];assert argv.count('--local-manifest')==1;argv[argv.index('--local-manifest')+1]='@manifest'
assert remote['worker_bytes']==58067865600 and remote['workers']==12 and remote['loader_threads']==4
assert peers[0]['endpoint']==base['worker_host'] and peers[0]['port']==base['port']
def config(name,leaf,attach=None,load=False):
    c=copy.deepcopy(base);a=copy.deepcopy(policy);c['arms']=[a]
    c.update(run_root=str(s/leaf),capture_first_logits=True,final_worker='retain',failure_worker='stop' if load else 'retain')
    c['assets'].update(engine=str(r/'engine-optimized'),worker=str(r/'worker-optimized'),reference=E['A0_SUMMARY'])
    c['argv'][0]='@engine';c['worker_argv'][0]='@worker'
    for av,flag in [(c['argv'],'--remote-cpu-local-manifest'),(c['worker_argv'],'--local-manifest')]:
        assert av.count(flag)==1;av[av.index(flag)+1]='@manifest'
    a.update(name=name,engine='engine',worker='worker',restart=load,requests=[] if load else [0,1,2])
    if load:
        c.pop('attach_root',None);c.pop('attached_worker_idle',None);a.pop('reference',None)
        a['placement_markers']=[text for text in a['placement_markers'] if not any(t in text for t in ('cache_hit=','setup network='))]
    else:c.update(attach_root=attach,attached_worker_idle=True);a['reference']='reference'
    return c
cohort=config('B_load','L',load=True);cohort['worker_peers']=peers
b=copy.deepcopy(policy);b.update(name='B',engine='engine',worker='worker',restart=False,requests=[0,1,2],reference='reference')
restore=copy.deepcopy(policy);restore.update(name='A1_load',engine='engine',worker='worker',restart=True,requests=[]);restore.pop('reference',None)
cohort['arms']=[cohort['arms'][0],b,restore]
for a in cohort['arms']:
    split=a['name']!='A1_load';a.update(cpu_peers=[peers[0]['name'],p['name']] if split else [peers[0]['name']],
                                      peer_bytes=[58067865600,58067865600] if split else [116135731200])
    markers=next(x['placement_markers'] for x in template['arms'] if len(x['cpu_peers'])==2 and x['restart']==a['restart']) if split else policy['placement_markers']
    a['placement_markers']=['fixed CPU group ready peers='+('2' if split else '1'),'CPU entry rows ready schema=1 ordered_reduce=1']+[
        text for text in markers if not any(t in text for t in ('cache_hit=','setup network=','fixed CPU group ready','CPU entry rows ready'))]
    if a['name']=='B':a['placement_markers']+=['cache_hit=1 reused=58067865600 B','setup network=0 B local=0 B']
configs=[('COHORT','L',cohort),('A0','W',config('A0','W',E['CURRENT_LOAD'])),
         ('A1','R',config('A1','R',str(s/'L'))),('RESTORE','F',config('RESTORE_load','F',load=True)),
         ('A1_RECOVERED','RF',config('A1','RF',str(s/'F')))]
for label,leaf,c in configs:
    path=s/(label+'.source.json')
    with path.open('x') as f:json.dump(c,f,indent=2)
# Write sources first; stage L before its attached R, and F before RF, below.
print('SPLIT_GENERATE02_PASS private_config',s/'COHORT.source.json','peer_name',p['name'])
PY
# END SPLIT_GENERATE02
```

The approved SSH argv has no remote tail: authorized_keys forces the existing worker_owner.py/config
path. Leave that argv, forced command, source restriction and credentials unchanged; never append a
command or route around the restriction. Static comparison confirms worker_owner.py is byte-identical
between9ed478e ande145df6e. This refreshes the same worker role/config, not its access or executable scope.

Before staging on05, verify the original guardian/worker/monitor identities are no longer live, old
stop/exit receipts show completed cleanup, the approved listener is free, and no process uses that
root. Archive the ENTIRE stopped root by rename to a distinct absent backup path, preserving original
bytes/permissions and absolute historical bindings. Do not edit/rebase the backup or change old/live
permissions. Then bundle.stage can exclusively recreate the SAME forced path; it refuses an existing
root. Verify source/staged guardian hashes and new worker/manifest pins before START. Only this role's
binary/config/assets change; model files and all approved bindings/policy remain in place. If absence,
ownership or writable parent cannot be established, stop/report rather than changing access settings.
The parent verifies the original peer budget against fresh live headroom.

```sh
# BEGIN SPLIT_STAGE05
set -euo pipefail
test "$(git -C "$SRC" rev-parse HEAD)" = e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb
test -x "$FIXTURE05"
PEER_NAME="$(python3 -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["worker_peers"][1]["name"])' "$PRIVATE_CONFIG")"
test ! -e "$FIXTURE_ROOT"
mkdir -m 700 "$FIXTURE_ROOT"
printf '%s  %s\n' "$WORKER_SHA" "$WORKER05" | sha256sum -c -
timeout --signal=TERM --kill-after=5s 60s "$FIXTURE05" --binary "$WORKER05" --load-threads 4 > "$FIXTURE_ROOT/compatible-worker.log" 2>&1
grep -Eq 'worker_local_test: 0 failures' "$FIXTURE_ROOT/compatible-worker.log"
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$PRIVATE_CONFIG" --worker "$PEER_NAME" > "$FIXTURE_ROOT/staging.log"
printf 'SPLIT_STAGE05_PASS\n'
# END SPLIT_STAGE05
```

Do not carry a small fixture address-space/file-size limit into the production peer: its assigned
arena alone is54.08GiB. Keep the prior production limits and61440MiB worker budget.

Before proceeding, obtain a fresh05 headroom/owned-listener receipt through the existing channel:
available memory must cover58067865600B plus the agreed unchanged safety reserve, without swap as
capacity; no unrelated worker may occupy the selected listener. Keep original resource guards. The
retained05 budget is61440MiB, for the assigned half only. The parent supplies fresh headroom;
installed RAM alone is not admission proof. Host swap can contain unrelated usage: require OWNED
worker VmSwap0, not global SwapUsed0. Record actual owned-process cgroup ancestry/max/high/events
through the existing monitor after START. Empty/unavailable root files do not establish unlimited
memory; label unavailable observations honestly, without changing cgroup or host settings.

```sh
# BEGIN SPLIT_STAGE02
set -euo pipefail
sha256sum -c "$LIVE_ROOT/optimized.sha256"
PYTHONPATH="$SRC/tools/validation-harness/harness" python3 -B - <<'PY'
import json,os,pathlib
import bundle,run
E=os.environ;s=pathlib.Path(E['STAGE_ROOT']);r=pathlib.Path(E['LIVE_ROOT'])
for label in ('COHORT','A0','A1','RESTORE','A1_RECOVERED'):bundle.stage(s/(label+'.source.json'))
for leaf in ('L','W','R','F','RF'):
    c=bundle.verify(s/leaf);resolved=run.arm_config(c,c['arms'][0])
    assert resolved['worker_argv'][0]==c['assets']['worker']
    assert resolved['worker_sha256']==bundle.sha(r/'worker-optimized')
    assert resolved['binary_sha256']==bundle.sha(r/'engine-optimized')
with (s/'LOAD-resolved.json').open('x') as f:json.dump(run.arm_config(bundle.verify(s/'L'),bundle.verify(s/'L')['arms'][0]),f)
with (s/'stage-complete.json').open('x') as f:json.dump({'source':'e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb'},f)
print('SPLIT_STAGE02_PASS')
PY
# END SPLIT_STAGE02
```

## Direct runtime entries

Run A0 against the still-current full resident. No cold load or worker replacement:

```sh
# BEGIN SPLIT_A0
set -euo pipefail
python3 -B "$STAGE_ROOT/W/run.py" --run-root "$STAGE_ROOT/W" > "$STAGE_ROOT/A0-driver.log" 2>&1
export RESULT_ROOT="$STAGE_ROOT/W" RESULT_ARM=A0
# END SPLIT_A0
```

Run EXACT_LOGITS below for A0 before release. Then bind OLD_LOAD_ROOT=CURRENT_LOAD,
OLD_LEGACY_ROOT=CURRENT_LEGACY and OLD_ROWS_ROOT=CURRENT_ROWS, and the full OLD_WORKER_SHA/OLD_ENGINE_SHA
from those current optimized receipts. Source the **unchanged REPLACE02** block from immutable
a29e02880ec939c54a97c6a2f849aaadd42d2784's CPU-REPLY-CODEC-LIVE-20261025.md. Its stage/hash/idle/
identity/one-STOP/pidfd/reaping gates apply directly to the staged L/W/R above. Never adopt the current
guardian with a fake Popen. No B load until old hierarchy cleanup and reusable port are confirmed.

```sh
# BEGIN SPLIT_B_AND_RESTORE
set -euo pipefail
test -f "$LIVE_ROOT/old-stop-completed.json"
python3 -B "$STAGE_ROOT/L/run.py" --run-root "$STAGE_ROOT/L" > "$STAGE_ROOT/B-driver.log" 2>&1
export LOAD_ROOT="$STAGE_ROOT/L" A1_ROOT="$STAGE_ROOT/R"
# END SPLIT_B_AND_RESTORE
```

Normal completion stops/reaps05 first, then02's half, and loads the full02 arena with zero GEN. Source
the existing a29e0288 **LOAD_PROC02** read-only block with LOAD_ROOT above: it verifies full /proc
hash/argv/complete arena and records loaded.json. Run EXACT_LOGITS for B with RESULT_ROOT=STAGE_ROOT/L,
RESULT_ARM=B. Record a numerical failure, but still complete A1 on the restored healthy full resident.

```sh
# BEGIN SPLIT_A1
set -euo pipefail
test -f "$LIVE_ROOT/loaded.json"
python3 -B "$A1_ROOT/run.py" --run-root "$A1_ROOT" > "$STAGE_ROOT/A1-driver.log" 2>&1
export RESULT_ROOT="$A1_ROOT" RESULT_ARM=A1
# END SPLIT_A1
```

```sh
# BEGIN EXACT_LOGITS
set -euo pipefail
status=0
python3 -B "$SRC/tools/validation-harness/compare_logits.py" --baseline "$A0_RAW0" "$A0_RAW1" "$A0_RAW2" --candidate "$RESULT_ROOT/$RESULT_ARM-first-logits.0" "$RESULT_ROOT/$RESULT_ARM-first-logits.1" "$RESULT_ROOT/$RESULT_ARM-first-logits.2" > "$RESULT_ROOT/$RESULT_ARM-logit-metrics.json" || status=$?
cmp -- "$A0_RAW0" "$RESULT_ROOT/$RESULT_ARM-first-logits.0" || status=1
cmp -- "$A0_RAW1" "$RESULT_ROOT/$RESULT_ARM-first-logits.1" || status=1
cmp -- "$A0_RAW2" "$RESULT_ROOT/$RESULT_ARM-first-logits.2" || status=1
printf '%s\n' "$status" > "$RESULT_ROOT/$RESULT_ARM-logit-status.txt"
if [ "$status" = 0 ]; then
  printf 'EXACT_LOGITS_PASS %s\n' "$RESULT_ARM"
elif [ "$RESULT_ARM" = B ]; then
  printf 'EXACT_LOGITS_FAILED B; preserve failure and finish restored A1\n' >&2
else
  return 1
fi
# END EXACT_LOGITS
```

All three raw rows must be finite993280B F32, all three token arrays/entry counts must match the
original A0 summary (213/219/198 outputs). No weakening of bit equality or frozen probability/logit
screens. A0 failure ends before repartitioning; B failure remains a failure even after successful A1.
After A1, require all three saved logit-status files to be0 and no cohort failure.json before declaring
acceptance. B's deliberate record-and-continue behavior only permits finishing A1; it never clears B.
If cmp fails, retain outputs and compute the same frozen metrics read-only for the failure report;
do not rerun GEN. Keep task correctness separate; no invented oracle.

## Finishing after a failed B controller

Do not repeat SPLIT_B_AND_RESTORE. The existing driver aborts/reaps its remote peer and stops/reaps its
owned local half under failure_worker=stop. It does NOT execute later restore entries on failure.
Before restoration, verify original B owner-exit/cleanup/stop receipts, identities absent,05 remote
child/transport reaping acknowledged, and02 reusable port. Uncertain cleanup is a blocker, not proof of
absence; reconcile the original channel/receipt without repeating STOP. No second subset load.

After verified cleanup, the following is the single explicit recovery full load, finishing the test
rather than retrying it. F/RF were already staged from the same validated artifacts/config policy:

```sh
# BEGIN SPLIT_FAILURE_RESTORE
set -euo pipefail
test -f "$STAGE_ROOT/L/failure.json"
test ! -e "$LIVE_ROOT/loaded.json"
test ! -e "$STAGE_ROOT/F/WORKER"
PYTHONPATH="$SRC/tools/validation-harness/harness" python3 -B - <<'PY'
import json,os,pathlib
import bundle
r=pathlib.Path(os.environ['STAGE_ROOT'])/'L';cfg=bundle.verify(r)
assert not list(r.glob('*cleanup-uncertain.json')),'reconcile original local cleanup first'
ids=r/'WORKER/identities.json'
if ids.exists():
    assert not any(pathlib.Path('/proc',str(v['pid'])).exists() for v in json.loads(ids.read_text()).values()),'local hierarchy still present'
else:
    abort=r/'worker-startup-abort.json'
    assert not list(r.glob('worker-owner-*.log')) or (abort.exists() and json.loads(abort.read_text())['outcome']=='completed')
name=cfg['worker_peers'][1]['name'];peer=r/'peers'/name
receipts=[json.loads(p.read_text()) for p in (peer/(name+'-abort.json'),peer/(name+'-stop.json')) if p.exists()]
assert any(v.get('outcome')=='completed' and v.get('transport_reaped') and v.get('remote',{}).get('children_reaped') for v in receipts),'remote child/transport cleanup not proven'
PY
python3 -B "$STAGE_ROOT/F/run.py" --run-root "$STAGE_ROOT/F" > "$STAGE_ROOT/restore-driver.log" 2>&1
export LOAD_ROOT="$STAGE_ROOT/F" A1_ROOT="$STAGE_ROOT/RF"
# END SPLIT_FAILURE_RESTORE
```

Then source the same a29 LOAD_PROC02 gate and, if health/resources permit, run SPLIT_A1/EXACT_LOGITS.
If recovery itself fails, preserve that separate failure and stop; no automatic second restore/retry.
If a full restore is already proven complete, reuse it and never load F. After either path retain the
full02 worker, guardian and original monitor. Confirm zero OWNED worker swap/new OOM, full unchanged accounting,
original compute masks and no remaining05 owned subset, using existing resource/identity receipts.

After A1 and resource/cleanup review, enforce the saved numerical verdicts without rerunning anything:

```sh
# BEGIN SPLIT_NUMERIC_VERDICT
set -euo pipefail
test ! -e "$STAGE_ROOT/L/failure.json"
test "$(cat "$STAGE_ROOT/W/A0-logit-status.txt")" = 0
test "$(cat "$STAGE_ROOT/L/B-logit-status.txt")" = 0
test "$(cat "$A1_ROOT/A1-logit-status.txt")" = 0
printf 'SPLIT_NUMERIC_VERDICT_PASS\n'
# END SPLIT_NUMERIC_VERDICT
```

Return one flat result: exact source/binary/config pins; all nine request times/tokens/entries and
first-logit metrics; per-peer requests/entries/bytes/service/exchange/wait; original endpoints/order;
cold setup separately; cleanup/restore/final full residency. Use the existing two-host per-peer receipts
and monitors; no new profiler. Sum neither overlapping peer times nor client/worker scopes. Equal
expert counts do not imply balanced work. A flat/slower result closes the round; no extra tuning arms.
