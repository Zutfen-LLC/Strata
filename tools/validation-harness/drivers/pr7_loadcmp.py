#!/usr/bin/env python3
"""pr7_loadcmp.py — PR7 1/2/4/1 local-loading comparison on 02 (config-driven).

Per handoff CPU-LOCAL-LOADING-20261024: four serial arms with the SAME
candidate worker ELF (d897e16c...), changing only --local-load-threads
1,2,4,1. Each arm: supported stop of the owned worker -> archive -> relaunch
-> C0 engine session whose normal setup performs the full local load -> READY
-> QUIT. No inference requests. Engine/pool-12/clock policy fixed. Final arm's
worker stays resident. All lifecycle from the retained harness files.
"""
import json, os, pathlib, shutil, subprocess, sys, time, traceback

R = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import campaign as C

CFGJ = json.loads((R / 'pr7_loadcmp.json').read_text())
ARMS = CFGJ['arms']
CAND_SHA = CFGJ['worker_sha256']


def stop_worker_supported(tag):
    ids = json.loads((R / 'WORKER' / 'identities.json').read_text())
    wpid = ids['worker']['pid']
    r = subprocess.run(['python3', str(R / 'worker_control.py'), 'stop', '--port', str(C.PORT)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and 'stop_acknowledged' in r.stdout, (r.returncode, r.stdout, r.stderr)
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline and pathlib.Path('/proc', str(wpid)).exists():
        time.sleep(2)
    assert not pathlib.Path('/proc', str(wpid)).exists(), 'worker did not exit'
    for k, v in ids.items():
        assert not pathlib.Path('/proc', str(v['pid'])).exists(), (k, v['pid'])
    C.atom(R / ('worker-stop-' + tag + '.json'), {'wall': time.time(), 'method': 'control opcode 11', 'identities': ids})


def arm_env(threads):
    """Snapshot cache/IO conditions + config for one arm."""
    cfg = json.loads((R / 'config.json').read_text())
    mem = {l.split(':')[0]: int(l.split()[1]) for l in pathlib.Path('/proc/meminfo').read_text().splitlines()
           if len(l.split()) >= 2 and l.split()[1].isdigit()}
    return {'mem_available_kib': mem['MemAvailable'], 'cached_kib': mem.get('Cached', 0),
            'swap_free_kib': mem['SwapFree'], 'boot_idle_s': None}


def run_arm(threads):
    name = 'T%d' % threads
    # 1. stop + archive current worker (authorized named replacement)
    if (R / 'WORKER').exists():
        C.state('loadcmp-stop-' + name)
        stop_worker_supported('before-' + name)
        k = 1
        while (R / ('WORKER-arch%d' % k)).exists():
            k += 1
        shutil.move(str(R / 'WORKER'), str(R / ('WORKER-arch%d' % k)))
    # 2. config: threads flag only
    cfg = json.loads((R / 'config.json').read_text())
    cfg['worker_argv'][0] = str(R / 'strata-expert-worker-cand')
    cfg['worker_sha256'] = CAND_SHA
    i = cfg['worker_argv'].index('--local-load-threads')
    cfg['worker_argv'][i + 1] = str(threads)
    (R / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    pre = arm_env(threads)
    assert hashlib_sha256(R / 'strata-expert-worker-cand') == CAND_SHA
    C.gpu_idle()
    C.state('loadcmp-launch-' + name)
    C.mark(phase='load-arm', arm=name, req=0)
    t_spawn = time.monotonic()
    subprocess.Popen(['python3', '-B', str(R / 'worker_owner.py')],
                     stdin=subprocess.DEVNULL, stdout=(R / 'worker-owner.log').open('ab'),
                     stderr=subprocess.STDOUT, start_new_session=True)
    C.waitfile(R / 'WORKER' / 'listening.json', lambda v: True, 120)
    ids = json.loads((R / 'WORKER' / 'identities.json').read_text())
    (R / 'clockmon-worker-pid').write_text(str(ids['worker']['pid']) + '\n')
    wio0 = read_io(ids['worker']['pid'])
    # 3. C0 engine session performs the load during normal setup
    sess = 'C0-%s' % name
    assert not (R / sess).exists()
    eowner = subprocess.Popen(['python3', '-B', str(R / 'owner.py'), sess],
                              stdin=subprocess.DEVNULL, stdout=(R / ('engine-owner-%s.log' % name)).open('ab'),
                              stderr=subprocess.STDOUT, start_new_session=True)
    st = C.wait_session(R / sess, lambda v: v['ready'] is not None, 2450, owner_pid=eowner.pid)
    ready_wall = st['ready']['mono'] - st['started_mono']
    C.rpc(sess, 'send', 'QUIT')
    engine_rc = None
    end = time.monotonic() + 300
    while time.monotonic() < end:
        try:
            v = json.loads((R / sess / 'state.json').read_text())
            if v.get('phase') == 'exited':
                engine_rc = v['exit_rc']
                break
        except Exception:
            pass
        time.sleep(1)
    assert engine_rc == 0, ('engine rc', engine_rc)
    for v in json.loads((R / sess / 'identities.json').read_text()).values():
        assert not pathlib.Path('/proc', str(v['pid']), 'exe').exists(), v
    C.gpu_idle()
    # 4. counters after load (worker still resident, session disconnected)
    ws = C.wstatus()
    wio1 = read_io(ids['worker']['pid'])
    s = ws['status']
    assert s['complete'] == 1 and s['resident_bytes'] == s['loaded_bytes'] == s['verified_bytes'] == C.WORKER_BYTES
    assert s['arena_allocations'] == 1 and s['requests'] == 0 and s['network_weight_bytes'] == 0
    post = arm_env(threads)
    rec = {'arm': name, 'threads': threads, 'ready_wall_s': round(ready_wall, 3),
           'spawn_to_ready_s': round(st['ready']['mono'] - t_spawn, 3),
           'worker_status': s, 'legacy': ws['legacy'],
           'worker_io_delta': {k: wio1[k] - wio0[k] for k in wio0},
           'env_pre': pre, 'env_post': post,
           'engine_rc': 0}
    C.atom(R / ('loadcmp-%s.json' % name), rec)
    C.state('loadcmp-' + name + '-complete', ready_wall_s=rec['ready_wall_s'])
    return rec


def read_io(pid):
    d = {}
    for l in pathlib.Path('/proc', str(pid), 'io').read_text().splitlines():
        k, _, v = l.partition(':')
        d[k] = int(v.strip())
    return d


def hashlib_sha256(p):
    import hashlib
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def replace_old_root_worker():
    """Authorized named replacement: stop the RAM-CMP root's owned 12-worker
    service through its supported control opcode before the first arm."""
    OLD = pathlib.Path('__CAMPAIGN_ROOT__')
    ids = json.loads((OLD / 'WORKER' / 'identities.json').read_text())
    wpid = ids['worker']['pid']
    if not pathlib.Path('/proc', str(wpid)).exists():
        prior = R / 'old-worker-stop-receipt.json'
        assert prior.exists(), 'old worker gone without receipt'
        return  # already replaced in a prior attempt (receipt retained)
    r = subprocess.run(['python3', str(OLD / 'worker_control.py'), 'stop', '--port', str(C.PORT)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and 'stop_acknowledged' in r.stdout, (r.returncode, r.stdout, r.stderr)
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline and any(pathlib.Path('/proc', str(v['pid'])).exists() for v in ids.values()):
        time.sleep(2)
    for k, v in ids.items():
        assert not pathlib.Path('/proc', str(v['pid'])).exists(), (k, v['pid'])
    C.atom(R / 'old-worker-stop-receipt.json', {'wall': time.time(), 'method': 'control opcode 11 via old root',
                                                'old_root': str(OLD), 'identities': ids,
                                                'note': 'authorized named replacement; arena released once; new candidate loads per arm'})


def main():
    C.state('loadcmp-preflight')
    replace_old_root_worker()
    cfg = json.loads((R / 'config.json').read_text())
    # engine binding unchanged (validated runtime engine stays)
    assert cfg['binary_sha256'] == '65419a1c85159291c535c03e28321e259a4f50832877b0272c786426a5f0718f'
    assert cfg['worker_argv'][cfg['worker_argv'].index('--workers') + 1] == '12'
    # post-swap environment snapshot
    C.atom(R / 'env-baseline.json', {**arm_env(0), 'note': 'post-swap 8x32GB; recorded before first arm'})
    C.freq_ctl('snapshot', str(R / 'freq-snapshot-loadcmp.json'))
    clockmon = subprocess.Popen(['python3', '-B', str(R / 'clockmon.py'), str(R), '14400'],
                                stdin=subprocess.DEVNULL, stdout=(R / 'clockmon.log').open('ab'),
                                stderr=subprocess.STDOUT, start_new_session=True)
    C.atom(R / 'clockmon-launch.json', {'pid': clockmon.pid, 'wall': time.time()})
    time.sleep(2)
    try:
        results = [run_arm(t) for t in ARMS]
        C.atom(R / 'loadcmp-results.json', {'wall': time.time(),
                                            'arms': [{k: r[k] for k in ['arm', 'threads', 'ready_wall_s', 'spawn_to_ready_s', 'worker_io_delta']} for r in results],
                                            'worker_final': C.wstatus()['status']})
    finally:
        C.mark(phase='final-check')
        (R / 'clockmon.STOP').write_text('done\n')
        try:
            clockmon.wait(timeout=60)
        except subprocess.TimeoutExpired:
            clockmon.terminate()
    C.state('completed-final-worker-resident')
    print('LOADCMP COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        C.atom(R / 'loadcmp-failure.json', {'wall': time.time(), 'error': repr(e), 'traceback': traceback.format_exc()})
        try:
            C.mark(phase='aborted')
            (R / 'clockmon.STOP').write_text('abort\n')
        except Exception:
            pass
        C.state('failed')
        raise
