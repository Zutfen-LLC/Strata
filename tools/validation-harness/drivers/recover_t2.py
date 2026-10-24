#!/usr/bin/env python3
"""recover_t2.py — recover the healthy orphaned T2 loading arm (attempt-1).
The engine session C0-T2 and its owner (alive) complete the load; we wait for
READY through the live owner, send QUIT, reap, and write the arm receipt with
io/cache evidence. Uses retained campaign helpers, absolute paths only."""
import json, pathlib, subprocess, sys, time

R = pathlib.Path('__CAMPAIGN_ROOT__')
sys.path.insert(0, str(R))
import campaign as C

SESS = R / 'C0-T2'
OWNER_PID = None
for p in pathlib.Path('/proc').iterdir():
    if p.name.isdigit():
        try:
            if 'owner.py C0-T2' in (p / 'cmdline').read_bytes().decode():
                OWNER_PID = int(p.name)
                break
        except Exception:
            pass
assert OWNER_PID, 'C0-T2 owner not alive'
st = json.loads((SESS / 'state.json').read_text())
ENG_PID = st['engine']['pid']
assert pathlib.Path('/proc', str(ENG_PID)).exists(), 'engine gone before READY'

wpid = int((R / 'clockmon-worker-pid').read_text().strip())
assert pathlib.Path('/proc', str(wpid), 'cmdline').read_bytes().decode().count('--local-load-threads 2'), 'pid file not the T2 worker'
io0 = {l.split(':')[0]: int(l.split()[1]) for l in pathlib.Path('/proc', str(wpid), 'io').read_text().splitlines()}

st = C.wait_session(SESS, lambda v: v['ready'] is not None, 2450, owner_pid=OWNER_PID)
ready_wall = st['ready']['mono'] - st['started_mono']
C.rpc('C0-T2', 'send', 'QUIT')
end = time.monotonic() + 300
while time.monotonic() < end:
    v = json.loads((SESS / 'state.json').read_text())
    if v.get('phase') == 'exited':
        break
    time.sleep(1)
assert v.get('exit_rc') == 0, v
time.sleep(2)
io1 = {l.split(':')[0]: int(l.split()[1]) for l in pathlib.Path('/proc', str(wpid), 'io').read_text().splitlines()}
mem = {l.split(':')[0]: int(l.split()[1]) for l in pathlib.Path('/proc/meminfo').read_text().splitlines()
       if len(l.split()) >= 2 and l.split()[1].isdigit()}
ws = C.wstatus()
wlog = ((R / 'WORKER/stdout.log').read_text() + (R / 'WORKER/stderr.log').read_text())
line = [l for l in wlog.splitlines() if 'local ready' in l][-1]
rec = {'arm': 'T2', 'threads': 2, 'ready_wall_s': round(ready_wall, 3), 'recovered': True,
       'recovery_note': 'attempt-1 driver died on stale-helper AttributeError after spawning this healthy session; load completed under its live owner; QUIT sent by recovery script',
       'worker_local_ready_line': line,
       'worker_io_cumulative_since_spawn': {k: io1[k] for k in ('read_bytes', 'rchar')},
       'mem_after': {'MemAvailable_kib': mem['MemAvailable'], 'Cached_kib': mem['Cached']},
       'cache_note': 'hits field = local source/manifest hits (22238 blobs), NOT Linux page-cache proof; cache residency inferred from read_bytes vs 116135731200 B payload and Cached/MemAvailable',
       'worker_status': ws['status']}
(R / 'loadcmp-T2.json').write_text(json.dumps(rec, indent=2) + '\n')
C.state('loadcmp-T2-complete-recovered', ready_wall_s=rec['ready_wall_s'])
print('T2 ready_wall_s', rec['ready_wall_s'], 'read_bytes', io1['read_bytes'], 'line:', line)
