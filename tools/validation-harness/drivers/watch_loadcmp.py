#!/usr/bin/env python3
"""watch_loadcmp.py — controller-side watcher for the PR7 loading campaign.
- Logs each arm receipt as it lands.
- Starts the read-only per-TID sampler at each arm launch (T2/T4 focus).
- After the T4 receipt exists: SIGSTOP the driver within milliseconds (before it
  can send arm-4's worker-stop), SIGKILL it, verify the T4 worker stays resident.
- If the race is lost (worker already stopping), reports it and does NOT reload.
"""
import json, pathlib, subprocess, sys, time

ROOT = pathlib.Path('__CAMPAIGN_ROOT__')
REMOTE = str(ROOT)
DRV_PID = int(subprocess.check_output(
    ['ssh', '-o', 'BatchMode=yes', '__HOST__', 'cat %s/loadcmp.pid' % REMOTE]).decode().strip())
SEEN = set()
SAMPLERS = {}

def sh(cmd):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '__HOST__', cmd],
                          capture_output=True, text=True)

def start_sampler(tag):
    if tag in SAMPLERS:
        return
    r = sh('cd %s && nohup setsid /usr/bin/python3 -B loader_sampler.py %s %s > sampler-%s.log 2>&1 < /dev/null & echo started' % (REMOTE, REMOTE, tag, tag))
    SAMPLERS[tag] = time.time()
    print('[watch] sampler started for', tag, r.stdout.strip(), flush=True)

print('[watch] driver pid', DRV_PID, flush=True)
t4_done = False
while not t4_done:
    time.sleep(1)
    ls = sh('ls %s | grep -E "^loadcmp-T[0-9]+\\.json$"' % REMOTE).stdout.split()
    tags = [t.replace('loadcmp-', '').replace('.json', '') for t in ls]
    st = sh('cat %s/campaign-state.json 2>/dev/null' % REMOTE).stdout
    for tag in tags:
        if tag not in SEEN:
            SEEN.add(tag)
            print('[watch] RECEIPT', tag, flush=True)
    for tag in ['T2', 'T4']:
        if ('loadcmp-launch-%s' % tag) in st and tag not in SAMPLERS and tag not in SEEN:
            start_sampler(tag)
    if 'T4' in SEEN:
        t4_done = True
# T4 receipt exists — freeze the driver immediately, then kill it
print('[watch] T4 receipt seen — freezing driver NOW', flush=True)
r = sh('kill -STOP %d' % DRV_PID)
time.sleep(0.5)
# verify worker still alive & resident
alive = sh('pgrep -f "__WORKER_BINARY__ --port __WORKER_PORT__" | head -1').stdout.strip()
stat = sh('cd %s && python3 -c "import socket,sys; sys.path.insert(0,\'\'); import worker_control, json; print(json.dumps(worker_control.status(socket.create_connection((\'127.0.0.1\',__WORKER_PORT__),timeout=5))))"' % REMOTE)
print('[watch] post-freeze worker pid:', alive, flush=True)
print('[watch] post-freeze status:', stat.stdout.strip()[:400], flush=True)
if alive:
    sh('kill -KILL %d' % DRV_PID)
    time.sleep(1)
    alive2 = sh('pgrep -f "__WORKER_BINARY__ --port __WORKER_PORT__" | head -1').stdout.strip()
    print('[watch] driver killed; worker still alive:', alive2, flush=True)
    outcome = 't4-resident-driver-stopped'
else:
    # race lost: worker already stopping/stopped — do NOT reload; report
    sh('kill -CONT %d 2>/dev/null; kill -KILL %d 2>/dev/null' % (DRV_PID, DRV_PID))
    outcome = 'RACE-LOST-worker-stopped-by-arm4-start'
print('[watch] OUTCOME:', outcome, flush=True)
(REMOTE.replace('__CAMPAIGN_ROOT__', '__CONTROLLER_PATH__') if False else None)
open('__CONTROLLER_PATH__', 'w').write(
    json.dumps({'wall': time.time(), 'outcome': outcome, 'seen': sorted(SEEN)}, indent=2) + '\n')
