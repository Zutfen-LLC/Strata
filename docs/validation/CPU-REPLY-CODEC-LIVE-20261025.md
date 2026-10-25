# Reply codec: fixed local live comparison

Runtime source is **e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb** throughout. This handoff is
documentation only; the later documentation SHA is not a new runtime candidate. Parent authorization
covers the comparison, one controlled replacement/reload and bounded prelaunch staging corrections.
The parent dispatches Hermes. No fleet operations, builds or executable checks occurred in this coding
thread. No merge, two-host campaign, new supervisor, protocol or runtime code is included.

Supplied Hermes evidence at that source:3 focused CTests passed, benchmark source bit-identical to
b9ab00c6,20 metrics/DONE passed with unchanged64/4096 counts and no resident contact. Row T1 encode/
decode improved209560.4/74335.2 to26260.2/18712.5ns; T2 improved419738.4/146577.8 to44084/34839.8ns.
Weighted T1 encode/decode27713.9/3454.1 to735.5/247.4ns; T2 41409.7/6986.8 to1316.4/906.1ns.
Reference phases varied; one pair establishes no noise band or live critical-path attribution. The
original verbose23.7MB success log is preserved. Reuse these passed CPU gates; do not rebuild/retest
them. The original split failure remains uncleared and frozen numerical screens stay unchanged.

Use the existing corrected LOAD2/LEGACY2/ROWS2 bundles, not older G1 worker bindings. Keep GPU960/1378,
CPU22238 experts/116135731200B,12 workers/4 loaders, turbo/affinities, model/profile/native/request
assets and original monitors/deadlines unchanged. No generation warmup. Both warm arms run only original
prompt0 GEN256, on the SAME optimized engine and new full worker. Require A0 first raw993280B logits
bitwise,213 exact tokens and102724 entries before proceeding. Cold loading is separate. Retain the new
complete resident after success or numerical failure. Never control a worker during an engine session.

The following blocks are the whole procedure. Run BUILD02 and OFF03 outside inference, then STAGE02,
REPLACE02, LOAD02, LOAD_PROC02, LEGACY02/WARM_GATE02 and ROWS02/WARM_GATE02 in order. Do not repeat successful blocks. Each block starts
with set -euo pipefail. Use fresh disk-backed output roots, preserve prior ELFs/logs, and stop on any
nonzero result. No package/network/access/settings change is authorized or needed.

Private bindings are supplied by the parent from verified receipts; never guess or publish them:

| Role | Bindings |
| --- | --- |
| 02 | SRC, CUDA_BUILD, ENGINE_OUTPUT, WORKER_OUTPUT, LIVE_ROOT (new), OLD_LOAD_ROOT, OLD_LEGACY_ROOT, OLD_ROWS_ROOT, OLD_WORKER_SHA, OLD_ENGINE_SHA, A0_SUMMARY, A0_LOGITS |
| 03 | SRC, OFF_BUILD, OFF_ENGINE_OUTPUT, OFF_ROOT (new) |
| 02 after OFF03 | OFF_PROOF: exact off-gates.json returned through the existing approved transfer channel; no new transport wrapper |
| 02 staging | STAGE_ROOT: initially LIVE_ROOT/s1; only one corrective root LIVE_ROOT/s2 is allowed |

Export bindings. Fetch the documentation SHA supplied with this handoff once on each source checkout,
then extract the selected named block mechanically. HANDOFF_SHA is the documentation SHA, not runtime
source. For example, replace BUILD02 with the required block name; do not transcribe its contents:

```sh
set -euo pipefail
git -C "$SRC" fetch origin "$HANDOFF_SHA"
source <(git -C "$SRC" show "$HANDOFF_SHA:docs/validation/CPU-REPLY-CODEC-LIVE-20261025.md" |
  sed -n '/^# BEGIN BUILD02$/,/^# END BUILD02$/p')
```

Source the extracted blocks in the SAME bash session. Do not pipe them into a child bash: ARM/LEAF
exported by each warm entry must remain available to its following WARM_GATE02.

## Cached builds: no resident replacement yet

```sh
# BEGIN BUILD02
set -euo pipefail
umask 077
CODE=e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb
test ! -e "$LIVE_ROOT"
test -f "$CUDA_BUILD/CMakeCache.txt"
test "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CUDA_BUILD/CMakeCache.txt")" = "$SRC"
grep -Eq '^STRATA_ENABLE_CUDA:BOOL=ON$' "$CUDA_BUILD/CMakeCache.txt"
test -z "$(git -C "$SRC" status --porcelain)"
git -C "$SRC" cat-file -e "$CODE^{commit}"
git -C "$SRC" checkout --detach "$CODE"
test "$(git -C "$SRC" rev-parse HEAD)" = "$CODE"
python3 -B - <<'PY'
import json,os,pathlib
ids=json.loads((pathlib.Path(os.environ['OLD_LOAD_ROOT'])/'WORKER/identities.json').read_text())
live=pathlib.Path('/proc',str(ids['worker']['pid']),'exe').stat()
out=pathlib.Path(os.environ['WORKER_OUTPUT']).stat()
assert (live.st_dev,live.st_ino)!=(out.st_dev,out.st_ino),'cached output aliases resident ELF'
PY
mkdir -m 700 "$LIVE_ROOT"
cp -- "$ENGINE_OUTPUT" "$LIVE_ROOT/prior-engine"
cp -- "$WORKER_OUTPUT" "$LIVE_ROOT/prior-worker"
sha256sum "$LIVE_ROOT/prior-engine" "$LIVE_ROOT/prior-worker" > "$LIVE_ROOT/prior-elf.sha256"
timeout --signal=TERM --kill-after=5s 60s cmake -S "$SRC" -B "$CUDA_BUILD" -DFETCHCONTENT_FULLY_DISCONNECTED=ON > "$LIVE_ROOT/configure02.log" 2>&1
timeout --signal=TERM --kill-after=5s 900s cmake --build "$CUDA_BUILD" --target strata strata-expert-worker --parallel 8 > "$LIVE_ROOT/build02.log" 2>&1
cp -- "$ENGINE_OUTPUT" "$LIVE_ROOT/engine-optimized"
cp -- "$WORKER_OUTPUT" "$LIVE_ROOT/worker-optimized"
timeout --signal=TERM --kill-after=5s 60s python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$LIVE_ROOT/engine-optimized" --native-experts on > "$LIVE_ROOT/cli02.log" 2>&1
sha256sum "$LIVE_ROOT/engine-optimized" "$LIVE_ROOT/worker-optimized" > "$LIVE_ROOT/optimized.sha256"
git -C "$SRC" rev-parse HEAD > "$LIVE_ROOT/source-sha.txt"
printf 'BUILD02_PASS source=%s\n' "$CODE"
# END BUILD02
```

Reuse the retained cache/compiler/native/ccache configuration; do not reconfigure other flags. The
output-path/inode check prevents overwriting a live worker executable. The build targets create only
new artifacts at this stage. Stop on a missing cache, dirty source or mismatch; no alternate cache.

```sh
# BEGIN OFF03
set -euo pipefail
umask 077
CODE=e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb
test ! -e "$OFF_ROOT"
test -f "$OFF_BUILD/CMakeCache.txt"
test "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$OFF_BUILD/CMakeCache.txt")" = "$SRC"
grep -Eq '^STRATA_NATIVE_EXPERTS:BOOL=OFF$' "$OFF_BUILD/CMakeCache.txt"
test -z "$(git -C "$SRC" status --porcelain)"
git -C "$SRC" cat-file -e "$CODE^{commit}"
git -C "$SRC" checkout --detach "$CODE"
mkdir -m 700 "$OFF_ROOT"
cp -- "$OFF_ENGINE_OUTPUT" "$OFF_ROOT/prior-engine"
timeout --signal=TERM --kill-after=5s 60s cmake -S "$SRC" -B "$OFF_BUILD" -DFETCHCONTENT_FULLY_DISCONNECTED=ON > "$OFF_ROOT/configure03.log" 2>&1
timeout --signal=TERM --kill-after=5s 300s cmake --build "$OFF_BUILD" --target strata --parallel 2 > "$OFF_ROOT/build03.log" 2>&1
cp -- "$OFF_ENGINE_OUTPUT" "$OFF_ROOT/engine-off"
timeout --signal=TERM --kill-after=5s 120s python3 -B "$SRC/tests/worker/worker_disabled_cli_test.py" --engine "$OFF_ROOT/engine-off" > "$OFF_ROOT/disabled03.log" 2>&1
timeout --signal=TERM --kill-after=5s 60s python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$OFF_ROOT/engine-off" --native-experts off > "$OFF_ROOT/family03.log" 2>&1
python3 -B - <<'PY'
import hashlib,json,os,pathlib
r=pathlib.Path(os.environ['OFF_ROOT']);h=hashlib.sha256((r/'engine-off').read_bytes()).hexdigest()
with (r/'off-gates.json').open('x') as f:json.dump(dict(source='e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb',disabled=True,family=True,engine_sha256=h),f)
print('OFF03_PASS',h)
PY
# END OFF03
```

Return off-gates.json with its original logs/hash through the existing approved channel. Bind OFF_PROOF
to that file on02. No large assets or resident controls are involved. Other backend checks remain NOT RUN.

## One mechanical staging operation, before releasing anything

This creates only private source configs/bundles. It normalizes all four G1 @asset positions, forcibly
selects the new worker asset in every arm, validates the resolved argv and hashes all staged copies.
No old asset name or literal executable/manifest path can silently select the previous worker.

```sh
# BEGIN STAGE02
set -euo pipefail
sha256sum -c "$LIVE_ROOT/optimized.sha256"
test "$(git -C "$SRC" rev-parse HEAD)" = e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb
test "$STAGE_ROOT" = "$LIVE_ROOT/s1" || test "$STAGE_ROOT" = "$LIVE_ROOT/s2"
test ! -e "$STAGE_ROOT"
PYTHONPATH="$SRC/tools/validation-harness/harness" python3 -B - <<'PY'
import copy,json,os,pathlib
import bundle,run
E=os.environ;root=pathlib.Path(E['LIVE_ROOT']);stage=pathlib.Path(E['STAGE_ROOT'])
proof=json.loads(pathlib.Path(E['OFF_PROOF']).read_text())
assert proof['source']=='e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb' and proof['disabled'] is True and proof['family'] is True
old=[bundle.verify(pathlib.Path(E[k])) for k in ('OLD_LOAD_ROOT','OLD_LEGACY_ROOT','OLD_ROWS_ROOT')]
assert all(len(c['arms'])==1 for c in old)
assert bundle.sha(old[0]['assets']['worker'])==E['OLD_WORKER_SHA']
assert bundle.sha(old[1]['assets']['engine'])==E['OLD_ENGINE_SHA']
assert all(bundle.sha(c['assets']['manifest'])==bundle.sha(old[0]['assets']['manifest']) and
           bundle.sha(c['assets']['requests'])==bundle.sha(old[0]['assets']['requests']) for c in old)
reference=json.loads(pathlib.Path(E['A0_SUMMARY']).read_text())['requests'][0]
assert reference['index']==1 and len(reference['tokens'])==213 and int(reference['worker_entries'])==102724
assert pathlib.Path(E['A0_LOGITS']).stat().st_size==993280
stage.mkdir(mode=0o700)
loaded=root/'loaded.json';attach=json.loads(loaded.read_text())['root'] if loaded.exists() else str(stage/'L')
for name,leaf,rows in [('LOAD','L',False),('LEGACY','W',False),('ROWS','R',True)]:
    if name=='LOAD' and loaded.exists():continue
    c=copy.deepcopy(old[0] if name=='LOAD' else old[1]);a=c['arms'][0]
    c.update(run_root=str(stage/leaf),capture_first_logits=True,cpu_route='group',final_worker='retain',failure_worker='stop' if name=='LOAD' else 'retain')
    c['assets'].update(engine=str(root/'engine-optimized'),worker=str(root/'worker-optimized'),reference=E['A0_SUMMARY'])
    c['argv'][0]='@engine';c['worker_argv'][0]='@worker'
    for argv,flag in [(c['argv'],'--remote-cpu-local-manifest'),(c['worker_argv'],'--local-manifest')]:
        assert argv.count(flag)==1;argv[argv.index(flag)+1]='@manifest'
    if '--remote-cpu-entry-rows' in c['argv']:c['argv'].remove('--remote-cpu-entry-rows')
    if rows:c['argv'].append('--remote-cpu-entry-rows')
    assert '--remote-cpu-regroup-diagnostic' not in c['argv'] and 'STRATA_CPU_CAPTURE' not in c['env']
    assert '--remote-cpu-rpc-timing' in c['argv'] and '--remote-cpu-resident' in c['argv']
    assert c['argv'].count('--remote-cpu-worker')==1 and '--remote-cpu-experts' not in c['argv']
    assert int(c['argv'][c['argv'].index('--expert-cache')+1])==960
    assert int(c['argv'][c['argv'].index('--expert-cache-device1')+1])==1378
    assert c['worker_bytes']==116135731200 and a['workers']==12 and a['loader_threads']==4
    assert a['caller_mask']==[0] and a['pool_masks']==[[i] for i in range(1,13)]
    assert not c.get('worker_peers')
    a.update(name=name,restart=name=='LOAD',requests=[] if name=='LOAD' else [0],engine='engine',worker='worker')
    if name=='LOAD':c.pop('attach_root',None);c.pop('attached_worker_idle',None);a.pop('reference',None)
    else:
        c.update(attach_root=attach,attached_worker_idle=True);a['reference']='reference'
        a['placement_markers']=list(a['placement_markers'])+['fixed CPU group ready peers=1','setup network=0 B local=0 B','cache_hit=1 reused=116135731200 B']
        if rows:a['placement_markers'].append('CPU entry rows ready schema=1 ordered_reduce=1')
    source=stage/(name+'.source.json')
    with source.open('x') as f:json.dump(c,f,indent=2)
    target=bundle.stage(source);staged=bundle.verify(target);resolved=run.arm_config(staged,staged['arms'][0])
    assert resolved['worker_argv'][0]==staged['assets']['worker']
    assert resolved['worker_sha256']==bundle.sha(root/'worker-optimized')
    assert resolved['binary_sha256']==bundle.sha(root/'engine-optimized')
    with (stage/(name+'-resolved.json')).open('x') as f:json.dump(resolved,f,indent=2)
with (stage/'stage-complete.json').open('x') as f:json.dump(dict(source='e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb',load_root=attach),f)
print('STAGE02_PASS',stage,'load_root',attach)
PY
# END STAGE02
```

## Release the old owned hierarchy once

All build/CLI/staging gates above must pass first. Exclusive idle use is required. Read-only /proc
inspection precedes STATUS; any established worker connection or recorded live engine blocks controls.
The existing worker_control entrypoint sends STOP once. The actual guardian is not this controller's
child: observe its retained pidfd exit and original-parent reaping, never fabricate a Popen/wait status.
The guardian itself reaps worker/monitor through its existing finally path. A timed-out STOP is uncertain;
continue observing the same identities, never resend. Only clean final receipts, all old /proc entries
absent and a reusable port permit LOAD.

```sh
# BEGIN REPLACE02
set -euo pipefail
test "$(git -C "$SRC" rev-parse HEAD)" = e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb
test -f "$STAGE_ROOT/stage-complete.json"
test -f "$STAGE_ROOT/LOAD-resolved.json"
sha256sum -c "$LIVE_ROOT/optimized.sha256"
PYTHONPATH="$SRC/tools/validation-harness/harness" python3 -B - <<'PY'
import json,os,pathlib,select,subprocess,sys,time
import bundle,run,campaign as C
E=os.environ;root=pathlib.Path(E['OLD_LOAD_ROOT']);cfg=bundle.verify(root)
for leaf in ('L','W','R'):
    staged=bundle.verify(pathlib.Path(E['STAGE_ROOT'])/leaf)
    assert bundle.sha(staged['assets']['worker'])==bundle.sha(pathlib.Path(E['LIVE_ROOT'])/'worker-optimized')
    assert bundle.sha(staged['assets']['engine'])==bundle.sha(pathlib.Path(E['LIVE_ROOT'])/'engine-optimized')
C.configure(root,run.arm_config(cfg,cfg['arms'][0]));C.assert_worker_idle()
for key in ('OLD_LOAD_ROOT','OLD_LEGACY_ROOT','OLD_ROWS_ROOT'):
    for path in pathlib.Path(E[key]).glob('*/identities.json'):
        engine=json.loads(path.read_text()).get('engine')
        if engine:
            try:C.ident(engine)
            except (FileNotFoundError,ProcessLookupError,AssertionError):continue
            raise RuntimeError('recorded engine still alive: '+str(path))
C.healthy();C.listener_owned()
address=__import__('socket').inet_aton(C.cfg['worker_host'])[::-1].hex().upper()+':'+format(C.PORT,'04X')
assert not any(r.split()[1]==address and r.split()[3]=='01' for r in pathlib.Path('/proc/net/tcp').read_text().splitlines()[1:]),'worker connection is active'
ids=json.loads((C.W/'identities.json').read_text());assert set(ids)=={'worker','guardian','monitor'}
for row in ids.values():C.ident(row)
assert ids['worker']['argv']==C.cfg['worker_argv']
assert bundle.sha(pathlib.Path('/proc',str(ids['worker']['pid']),'exe'))==E['OLD_WORKER_SHA']==C.cfg['worker_sha256']
status=C.wstatus();assert status['status']['complete'] and not status['status']['session']
assert status['status']['resident_bytes']==116135731200 and status['status']['idle_remaining_ms']==2**64-1
assert not status['legacy']['failures']
fds=[os.pidfd_open(row['pid']) for row in ids.values()]
path=C.W/'stop.json';receipt=dict(operation='stop',outcome='pending',identities=ids,control_outcome='pending')
with path.open('x') as f:json.dump(receipt,f,indent=2) # existing intent means reconcile, never send again
try:
    try:
        p=subprocess.run([sys.executable,'-B',str(pathlib.Path(E['SRC'])/'tools/validation-harness/harness/worker_control.py'),'stop','--host',C.cfg['worker_host'],'--port',str(C.PORT)],capture_output=True,text=True,timeout=10)
        assert p.returncode==0 and json.loads(p.stdout)=={'stop_acknowledged':True},p.stderr
        receipt['control_outcome']='acknowledged'
    except Exception as error:receipt.update(control_outcome='uncertain',error=repr(error))
    C.atom(path,receipt)
    end=time.monotonic()+C.cfg['shutdown_seconds']
    while not all(select.select([fd],[],[],0)[0] for fd in fds):
        if time.monotonic()>=end:raise TimeoutError('original hierarchy exit uncertain; never resend STOP')
        time.sleep(.1)
    end=time.monotonic()+C.cfg['cleanup_seconds']
    while any(pathlib.Path('/proc',str(row['pid'])).exists() for row in ids.values()):
        if time.monotonic()>=end:raise TimeoutError('original parent has not reaped hierarchy')
        time.sleep(.1)
    final=json.loads((C.W/'state.json').read_text())
    assert final['phase']=='exited' and final['exit_rc']==0 and final['monitor_exit_rc']==0 and not final['failure']
    C.port_free(C.cfg['worker_host'],C.PORT,C.cfg['cleanup_seconds'])
    receipt.update(outcome='completed',guardian_exit_observed=True,hierarchy_absent=True,guardian_wait_status=None)
    C.atom(path,receipt);C.atom(pathlib.Path(E['LIVE_ROOT'])/'old-stop-completed.json',receipt)
    print('REPLACE02_OLD_EXIT_PASS')
finally:
    for fd in fds:os.close(fd)
PY
# END REPLACE02
```

No process signal, mlock/swapoff/settings change or automatic restore is used. If STOP intent already
exists, do not execute REPLACE02 again. Reconcile its original receipt and identities read-only; report
any uncertain exit instead of launching another arena.

## Cold load, then exactly one legacy and one row request

```sh
# BEGIN LOAD02
set -euo pipefail
test -f "$STAGE_ROOT/stage-complete.json"
test -f "$LIVE_ROOT/old-stop-completed.json"
test ! -e "$LIVE_ROOT/loaded.json"
test ! -e "$STAGE_ROOT/L/WORKER"
python3 -B "$STAGE_ROOT/L/run.py" --run-root "$STAGE_ROOT/L" > "$STAGE_ROOT/load-driver.log" 2>&1
export LOAD_ROOT="$STAGE_ROOT/L"
# END LOAD02
```

```sh
# BEGIN LOAD_PROC02
set -euo pipefail
PYTHONPATH="$SRC/tools/validation-harness/harness" python3 -B - <<'PY'
import json,os,pathlib
import bundle,run,campaign as C
E=os.environ;root=pathlib.Path(E['LOAD_ROOT']);cfg=bundle.verify(root);resolved=run.arm_config(cfg,cfg['arms'][0])
C.configure(root,resolved);C.assert_worker_idle();C.healthy();C.listener_owned()
ids=json.loads((C.W/'identities.json').read_text())
for row in ids.values():C.ident(row)
assert ids['worker']['argv']==resolved['worker_argv']
actual=bundle.sha(pathlib.Path('/proc',str(ids['worker']['pid']),'exe'))
assert actual==resolved['worker_sha256']==bundle.sha(pathlib.Path(E['LIVE_ROOT'])/'worker-optimized')
st=pathlib.Path('/proc',str(ids['worker']['pid']),'status').read_text().splitlines()
assert next(v for v in st if v.startswith('VmSwap:')).split()[1]=='0'
health=C.healthy();status=C.wstatus();assert status['status']['complete'] and not status['status']['session']
assert status['status']['resident_bytes']==116135731200 and status['status']['arena_allocations']==1
assert status['status']['idle_remaining_ms']==2**64-1
oom={p:{k:int(v) for k,v in (line.split() for line in row.get('memory.events','').splitlines()) if k.startswith('oom')}
     for p,row in health['cgroups'].items()}
receipt=dict(root=str(root),identities=ids,worker_sha256=actual,status=status,oom_counts=oom)
with (pathlib.Path(E['LIVE_ROOT'])/'loaded.json').open('x') as f:json.dump(receipt,f,indent=2)
print('LOAD02_PROC_HASH_PASS',actual)
PY
# END LOAD_PROC02
```

The source driver's masks() also checks /proc worker SHA immediately after listening, before SETUP;
the explicit gate above records the full resolved argv/hash after zero-GEN READY and before any GEN.
It applies broad setup affinity until loaders join, then original caller/pool masks before GEN.

For each warm arm use the direct existing run.py entrypoint. This block is identical except ARM/LEAF:
LEGACY02 exports ARM=LEGACY,LEAF=W; only after its exact gate succeeds, ROWS02 exports ARM=ROWS,LEAF=R.
Select the named blocks below; do not add requests or a sequencing wrapper.

```sh
# BEGIN LEGACY02
set -euo pipefail
export ARM=LEGACY LEAF=W
test -f "$LIVE_ROOT/loaded.json"
test ! -e "$LIVE_ROOT/LEGACY-pass.json"
test ! -e "$LIVE_ROOT/s1/W/LEGACY"
test ! -e "$LIVE_ROOT/s2/W/LEGACY"
test ! -e "$STAGE_ROOT/W/LEGACY"
python3 -B "$STAGE_ROOT/W/run.py" --run-root "$STAGE_ROOT/W" > "$STAGE_ROOT/legacy-driver.log" 2>&1
# END LEGACY02
```

After either warm entry, run WARM_GATE02 with its exported ARM/LEAF, which must come from that entry.
It only reads completed receipts/logits; no worker controls occur. LEGACY must pass this gate before
ROWS02. If numerical gating fails, retain the worker and preserve outputs; no repeated GEN.

```sh
# BEGIN WARM_GATE02
set -euo pipefail
PYTHONPATH="$SRC/tools/validation-harness:$SRC/tools/validation-harness/harness" python3 -B - <<'PY'
import json,os,pathlib
import compare_logits as comparison
import bundle,run,campaign as C
E=os.environ;root=pathlib.Path(E['STAGE_ROOT'])/E['LEAF'];name=E['ARM']
assert (name,E['LEAF']) in [('LEGACY','W'),('ROWS','R')]
summary=json.loads((root/('arm-'+name+'-summary.json')).read_text());rows=summary['requests']
reference=json.loads(pathlib.Path(E['A0_SUMMARY']).read_text())['requests'][0]
assert len(rows)==1 and rows[0]['index']==1 and rows[0]['tokens']==reference['tokens']
assert rows[0]['output_count']==213 and int(rows[0]['worker_entries'])==102724
a=pathlib.Path(E['A0_LOGITS']);b=root/(name+'-first-logits.0')
assert a.stat().st_size==b.stat().st_size==993280
metrics=comparison.compare(comparison.read(a),comparison.read(b));metrics['bitwise_equal']=a.read_bytes()==b.read_bytes()
assert metrics['numeric_pass'] and metrics['bitwise_equal'],metrics
before=json.loads((root/('worker-before-'+name+'.json')).read_text());after=json.loads((root/('worker-after-'+name+'.json')).read_text())
delta={k:after['legacy'][k]-before['legacy'][k] for k in ('infer_in_bytes','infer_out_bytes','requests','failures','cpu_service_ns')}
assert delta['requests']==11136 and delta['failures']==0 and delta['infer_in_bytes']==124299072
assert delta['infer_out_bytes']==(123236352 if name=='LEGACY' else 1052500104)
for key in ('loaded_bytes','verified_bytes','arena_allocations','unloads','expirations'):
    assert before['status'][key]==after['status'][key]
baseline_root=pathlib.Path(E['OLD_LEGACY_ROOT'] if name=='LEGACY' else E['OLD_ROWS_ROOT'])
baseline_cfg=json.loads((baseline_root/'config.json').read_text());oldname=baseline_cfg['arms'][0]['name']
prior=json.loads((baseline_root/('arm-'+oldname+'-summary.json')).read_text())['requests'][0]
result=dict(source='e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb',root=str(root),arm=name,first_logits=metrics,
            request=rows[0],legacy_counter_delta=delta,e56_prompt_s=prior['prompt_s'],e56_decode_s=prior['decode_s'])
state=json.loads((root/name/'state.json').read_text())
assert state['phase']=='exited' and state['exit_rc']==0 and not state['failure']
assert json.loads((root/(name+'-owner-exit.json')).read_text())['returncode']==0
assert not any(pathlib.Path('/proc',str(row['pid'])).exists() for row in json.loads((root/name/'identities.json').read_text()).values())
assert not (root/name/'monitor-failure.json').exists()
assert json.loads((root/name/'monitor-final.json').read_text())['peaks']['swap_kib']==0
cfg=bundle.verify(root);resolved=run.arm_config(cfg,cfg['arms'][0]);C.configure(root,resolved)
loaded=json.loads((pathlib.Path(E['LIVE_ROOT'])/'loaded.json').read_text())
ids=json.loads((C.W/'identities.json').read_text());assert ids==loaded['identities']
for row in ids.values():C.ident(row)
assert bundle.sha(pathlib.Path('/proc',str(ids['worker']['pid']),'exe'))==loaded['worker_sha256']==resolved['worker_sha256']
health=C.healthy();assert not (C.W/'monitor-failure.json').exists() and health['peaks']['swap_kib']==0
oom={p:{k:int(v) for k,v in (line.split() for line in row.get('memory.events','').splitlines()) if k.startswith('oom')}
     for p,row in health['cgroups'].items()}
assert oom==loaded['oom_counts'],'new cgroup OOM event'
st=pathlib.Path('/proc',str(ids['worker']['pid']),'status').read_text().splitlines()
assert next(v for v in st if v.startswith('VmSwap:')).split()[1]=='0'
tasks=C.tasks(ids['worker']['pid']);arm=cfg['arms'][0]
assert len(tasks)==14 and [t['mask'] for t in tasks[:13]]==[arm['caller_mask'],*arm['pool_masks']]
result['final_resource_gate']=dict(healthy=True,swap_kib=0,oom_unchanged=True,worker_tasks=tasks)
with (pathlib.Path(E['LIVE_ROOT'])/(name+'-pass.json')).open('x') as f:json.dump(result,f,indent=2)
print('WARM_GATE02_PASS',name,json.dumps(result,allow_nan=False))
PY
# END WARM_GATE02
```

```sh
# BEGIN ROWS02
set -euo pipefail
test -f "$LIVE_ROOT/LEGACY-pass.json"
export ARM=ROWS LEAF=R
test ! -e "$LIVE_ROOT/ROWS-pass.json"
test ! -e "$LIVE_ROOT/s1/R/ROWS"
test ! -e "$LIVE_ROOT/s2/R/ROWS"
test ! -e "$STAGE_ROOT/R/ROWS"
python3 -B "$STAGE_ROOT/R/run.py" --run-root "$STAGE_ROOT/R" > "$STAGE_ROOT/rows-driver.log" 2>&1
# END ROWS02
```

After ROWS02 run WARM_GATE02 for ROWS. Return both pass files, optimized ELF hashes, source pin,
staged/resolved config pins, old STOP/exit receipt, new /proc hash receipt, cold startup/read/verify
separately, and final resource/ownership state. Read new worker health/identity files only after engine
owners are reaped: require fresh healthy original monitor, unchanged complete arena/allocation1,
unloads/expirations0, VmSwap/OOM0 and original compute masks. Existing arm after-status receipts cover
cache accounting. Existing stderr reports cover bytes, native service, client exchange/receive/wait and
row_reduce_ms; scopes overlap and are not summed. Preserve both whole token arrays and first raw logits.
One sample each supports a local comparison only, not a LAN or production speedup claim.

## Bounded prelaunch correction and resumption

* Before old STOP: a path/import/config/pin failure changes no service. Preserve s1, correct only the
  proven source binding, set STAGE_ROOT=LIVE_ROOT/s2, and run STAGE02 once using the SAME built ELFs.
  Do not repeat passed build/CLI/CPU gates. No staged files/pins are edited in place.
* After completed STOP but before LOAD launched: the same s2 correction is allowed, followed by LOAD02
  once. Never rerun REPLACE02. An occupied port or uncertain old hierarchy blocks loading.
* After successful LOAD: loaded.json makes STAGE02 omit LOAD and attach warm bundles to the SAME arena.
  For a proven prelaunch warm staging error, stage s2 once and run only the missing warm entry. Keep a
  previously successful LEGACY-pass.json and do not repeat its request. No reload is needed.
* A LOAD driver success followed by an identity-receipt error permits rerunning only LOAD_PROC02,
  not LOAD02/run.py. A partial/new-worker failure follows existing owned driver cleanup;
  stop/report instead of automatically restoring or starting another full load.
* If an engine owner started, preserve its recorded requests_sent/completed/phase and cleanup receipts;
  the recipe refuses a second attempt for that arm even if no pass file exists. Any sent/uncertain GEN is preserved and never automatically repeated.
  No worker status/STOP/unload is sent until that owner is closed/reaped. A second staging failure or
  any numerical/resource/ownership failure stops the procedure with the new complete resident retained.
