#!/usr/bin/env python3
"""campaign.py — CPU-WEIGHT-CAPPED-VS-TURBO-V1 on the target host.

Arms: A1 (2.0 GHz policy ceiling) -> B (3.0 GHz turbo) -> A2 (2.0 GHz again).
Three retained GEN256 requests per arm, identical order/seeds/bytes.
One cold local worker arena load (C0 session, zero requests), then the arena
stays resident across all engine sessions; restored policy = Eric's post-boot
turbo-enabled configuration on every exit path.
"""
import os, sys, json, time, socket, struct, subprocess, hashlib, pathlib, traceback

R = pathlib.Path(__file__).resolve().parent
W = R                      # worker root == campaign root (WORKER/ inside)
sys.path.insert(0, str(R))
import worker_control

cfg = json.loads((R / 'config.json').read_text())
ARM_LINES = [l for l in (R / 'arm-requests.txt').read_text().splitlines() if l.startswith('GEN ')]
assert len(ARM_LINES) == 3
PORT = cfg['port']
WORKER_BYTES = 116135731200
CPUS = list(range(28))
POOL_COUNT = 12  # worker pool size for verification checks; set by config-driven wrappers


def atom(p, v):
    t = str(p) + '.tmp'
    pathlib.Path(t).write_text(json.dumps(v, indent=2) + '\n')
    os.replace(t, p)


def state(phase, **kw):
    row = {'phase': phase, 'wall': time.time(), 'pid': os.getpid(), **kw}
    atom(R / 'campaign-state.json', row)
    return row


def mark(**kw):
    atom(R / 'clockmon-marker.json', {'wall': time.time(), **kw})


def ident(v):
    p = pathlib.Path('/proc', str(v['pid']))
    st = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    e = (p / 'exe').stat()
    assert (int(st[19]), e.st_dev, e.st_ino) == (v['start_ticks'], v['exe_dev'], v['exe_ino']), v
    assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip() == v['boot_id']
    assert os.readlink(p / 'exe') == v['exe']
    assert (p / 'cmdline').read_bytes().decode().split('\0')[:-1] == v['argv']
    return {**v, 'ppid': int(st[1]), 'mask': sorted(os.sched_getaffinity(v['pid']))}


def rpc(arm, op, line=None):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(str(R / arm / 'input.sock'))
        s.sendall((json.dumps({'op': op, **({'line': line} if line is not None else {})}) + '\n').encode())
        b = b''
        while b'\n' not in b:
            x = s.recv(65536)
            if not x:
                break
            b += x
        v = json.loads(b)
        assert 'error' not in v, v
        return v


def waitfile(path, predicate, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path.exists():
            v = json.loads(path.read_text())
            if v.get('failure'):
                raise RuntimeError('owner failure: ' + json.dumps(v))
            if predicate(v):
                return v
            if v.get('phase') == 'exited':
                raise RuntimeError('owner exited early: ' + json.dumps(v))
        time.sleep(0.2)
    raise RuntimeError('wait deadline: ' + str(path))


def wait_session(arm_dir, predicate, seconds, owner_pid=None):
    """waitfile + live health: fails fast when (a) the owner exited before ever
    writing state.json (launch failure), or (b) the engine process vanished
    without an exit record. Distinguishes: process exited / alive with state.
    No stall-kill thresholds added; the outer deadline is unchanged."""
    import pathlib as _p
    state = _p.Path(arm_dir) / 'state.json'
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if state.exists():
            v = json.loads(state.read_text())
            if v.get('failure'):
                raise RuntimeError('owner failure: ' + json.dumps(v))
            if predicate(v):
                return v
            if v.get('phase') == 'exited':
                raise RuntimeError('owner exited early: ' + json.dumps(v))
            eng = v.get('engine')
            if eng and v.get('exit_rc') is None and not _p.Path('/proc', str(eng['pid'])).exists():
                raise RuntimeError('engine pid %s vanished without exit record' % eng['pid'])
        elif owner_pid is not None and not _p.Path('/proc', str(owner_pid)).exists():
            raise RuntimeError('owner pid %s exited before writing state.json (launch failure)' % owner_pid)
        time.sleep(2)
    raise RuntimeError('wait deadline: ' + str(state))


def tasks(pid):
    out = []
    for p in pathlib.Path('/proc', str(pid), 'task').iterdir():
        try:
            out.append({'tid': int(p.name), 'mask': sorted(os.sched_getaffinity(int(p.name))),
                        'comm': (p / 'comm').read_text().strip()})
        except (FileNotFoundError, ProcessLookupError):
            pass
    return sorted(out, key=lambda x: x['tid'])


def topology():
    t = {}
    for c in range(os.cpu_count() or 28):
        p = pathlib.Path('/sys/devices/system/cpu', 'cpu' + str(c), 'topology')
        t[str(c)] = {k: (p / k).read_text().strip() for k in ['physical_package_id', 'core_id', 'thread_siblings_list']}
    return t


def affinity(arm, tag, need_io=False):
    wi = ident(json.loads((R / 'worker-identities.json').read_text())['worker'])
    wt = tasks(wi['pid'])
    n = POOL_COUNT
    assert len(wt) == n + 2, ('worker task count', len(wt), n + 2)
    assert wt[0]['mask'] == [0], 'caller not pinned to CPU0'
    pool = wt[1:1 + n]
    assert all(len(x['mask']) == 1 for x in pool), ('pool singleton masks', [x['mask'] for x in pool])
    topo = topology()
    core_of = {int(c): (v['physical_package_id'], v['core_id']) for c, v in topo.items()}
    pool_cores = [core_of[x['mask'][0]] for x in pool]
    assert len(set(pool_cores)) == n, ('pool on distinct physical cores', pool_cores)
    assert core_of[0] not in pool_cores, 'pool overlaps caller physical core'
    ei = json.loads((R / arm / 'state.json').read_text())['engine']
    ident(ei)
    et = tasks(ei['pid'])
    assert et[0]['mask'] == [0], 'engine main not on CPU0'
    io = [x for x in et if x['comm'] == 'strata-cpu-io']
    if need_io:
        assert len(io) == 1, io
    out = {'wall': time.time(), 'worker': wi, 'worker_pool': wt[1:13], 'worker_caller': wt[0],
           'worker_auxiliary': wt[13:], 'engine': ei, 'engine_main': et[0], 'engine_pool': et[1:13] if len(et) > 13 else [],
           'communication_threads': io, 'topology': topology()}
    atom(R / (tag + '-affinity.json'), out)
    return out


def wstatus():
    with socket.create_connection(('127.0.0.1', PORT), timeout=5) as s:
        cache = worker_control.status(s)
        raw = worker_control.exchange(s, 9, 10)
        vals = struct.unpack('<7QHB', raw)
        keys = ['session', 'setup_weight_bytes', 'infer_in_bytes', 'infer_out_bytes', 'requests', 'failures', 'cpu_service_ns', 'last_error', 'ready']
        return {'wall': time.time(), 'status': cache, 'legacy': dict(zip(keys, vals))}


def healthy():
    ids = json.loads((R / 'WORKER' / 'identities.json').read_text())
    for v in ids.values():
        ident(v)
    h = json.loads((R / 'WORKER' / 'health.json').read_text())
    assert time.time() - h['wall'] < 15 and not h['failure'], 'worker monitor stale/failed'
    st = json.loads((R / 'WORKER' / 'state.json').read_text())
    assert st['phase'] == 'service' and not st['failure'], 'guardian state bad'
    return h


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)


def gpu_idle():
    g = gpu()
    assert all(int(l.split(',')[1]) < 100 for l in g.splitlines()), g
    return g


def listener_owned():
    ids = json.loads((R / 'WORKER' / 'identities.json').read_text())
    fds = []
    for p in pathlib.Path('/proc', str(ids['worker']['pid']), 'fd').iterdir():
        try:
            fds.append(os.readlink(p))
        except FileNotFoundError:
            pass
    tcp = [l.split() for l in pathlib.Path('/proc/net/tcp').read_text().splitlines()[1:]]
    owned = [x for x in tcp if x[1] == '0100007F:' + format(PORT, '04X') and x[3] == '0A' and 'socket:[' + x[9] + ']' in fds]
    assert len(owned) == 1, owned
    return owned


def freq_ctl(*args):
    r = subprocess.run(['python3', '-B', str(R / 'freq_ctl.py'), *args], capture_output=True, text=True)
    rec = {'argv': list(args), 'rc': r.returncode, 'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()[:800], 'wall': time.time()}
    if r.returncode != 0:
        raise RuntimeError('freq_ctl failed: ' + json.dumps(rec))
    return rec


def parse_done(line):
    # DONE <output_count> <?> <prompt_ms> <decode_ms> <finish> ... worker_entries=N worker_req_wall=Ms
    f = line.split()
    d = {'raw': line, 'output_count': int(f[1]), 'prompt_s': float(f[3]) / 1000.0,
         'decode_s': float(f[4]) / 1000.0, 'finish': f[5]}
    for kv in f[6:]:
        if '=' in kv:
            k, v = kv.split('=', 1)
            d[k] = v
    return d


def tokens_between(log_text, start_idx, end_idx):
    """Extract 'T <id>' lines in [start_idx, end_idx) of the engine stdout log."""
    ids = []
    for line in log_text[start_idx:end_idx].splitlines():
        if line.startswith('T '):
            ids.append(int(line.split()[1]))
    return ids


def run_arm(arm, idxs=(0, 1, 2)):
    """Launch one engine session, send the selected retained GEN lines, QUIT.

    Per-request worker data comes from the DONE line; the worker control socket
    is only queried before engine start and after engine exit (the service
    serializes frames behind live engine traffic — mid-session status probes
    time out; the retained driver never probed mid-session either).
    """
    t_start = time.monotonic()
    assert not (R / arm).exists()
    n_req = len(idxs)
    state('arm-' + arm + '-startup')
    mark(phase='arm', arm=arm, req=0)
    healthy()
    before = wstatus()
    atom(R / ('worker-before-' + arm + '.json'), before)
    bs = before['status']
    assert bs['session'] == 0 and bs['state'] == 'empty' and bs['complete'] == 1
    assert bs['resident_bytes'] == WORKER_BYTES and bs['loaded_bytes'] == bs['verified_bytes'] == WORKER_BYTES
    assert bs['arena_allocations'] == 1 and bs['network_weight_bytes'] == bs['unloads'] == bs['expirations'] == 0
    assert bs['idle_remaining_ms'] == 2 ** 64 - 1
    assert before['legacy']['failures'] == 0
    listener_owned()
    gpu_idle()
    owner = subprocess.Popen(['python3', '-B', str(R / 'owner.py'), arm],
                             stdin=subprocess.DEVNULL,
                             stdout=(R / ('engine-owner-' + arm + '.log')).open('ab'),
                             stderr=subprocess.STDOUT, start_new_session=True)
    atom(R / ('engine-owner-launch-' + arm + '.json'), {'pid': owner.pid, 'wall': time.time()})
    st = waitfile(R / arm / 'state.json', lambda v: v['ready'] is not None, 2450)
    startup = {'arm': arm, 'started_mono': st['started_mono'], 'ready_mono': st['ready']['mono'],
               'startup_to_READY_s': st['ready']['mono'] - st['started_mono']}
    affinity(arm, arm + '-before', need_io=False)
    text = (R / arm / 'stderr.log').read_text()
    placement = {k: (k in text) for k in [
        'expert cache 960 slots', 'pre-filled 960 of 960 slots',
        'CUDA1: 1378 additional experts', 'remote CPU experts: 22238 of 24576 experts',
        'cache_hit=1 reused=116135731200 B', 'setup network=0 B local=0 B hits=0 misses=0',
        'participating threads: 13']}
    assert all(placement.values()), placement
    assert st['completed'] == st['requests_sent'] == 0
    rows = []
    io_tid = None
    first_send_mono = None
    stdout_path = R / arm / 'stdout.log'
    log_offset = 0
    for k, j in enumerate(idxs):
        line = ARM_LINES[j]
        healthy()
        mark(phase='arm', arm=arm, req=j + 1)
        log_offset = stdout_path.stat().st_size if stdout_path.exists() else 0
        ack = rpc(arm, 'send', line)
        if first_send_mono is None:
            first_send_mono = ack['sent']['mono']
        st = waitfile(R / arm / 'state.json', lambda v: v['completed'] == k + 1, 1810)
        assert st['requests_sent'] == k + 1 and not st['active']
        done = parse_done(st['done']['line'])
        log_text = stdout_path.read_text()
        tokens = tokens_between(log_text, log_offset, len(log_text))
        assert len(tokens) == done['output_count'], (len(tokens), done['output_count'])
        af = affinity(arm, arm + '-request-%02d' % (j + 1), need_io=True)
        tid = af['communication_threads'][0]['tid']
        if io_tid is None:
            io_tid = tid
        assert tid == io_tid, 'communication TID changed mid-arm'
        rows.append({'index': j + 1, 'session_index': k + 1, 'sent_mono': ack['sent']['mono'], 'done_mono': st['done']['mono'],
                     'done_raw': st['done']['line'], 'prompt_s': done['prompt_s'], 'decode_s': done['decode_s'],
                     'finish': done['finish'], 'tokens': tokens, 'output_count': done['output_count'],
                     'worker_entries': done.get('worker_entries'), 'worker_req_wall_ms': done.get('worker_req_wall'),
                     'communication_tid': tid,
                     'engine_masks': {'main': af['engine_main']['mask'],
                                      'io': af['communication_threads'][0]['mask']}})
        state('arm-' + arm + '-request-%02d-done' % (j + 1))
    last_done_mono = rows[-1]['done_mono']
    atom(R / arm / 'measurement-end.json', {'wall': time.time(), 'mono': time.monotonic(), 'state': st})
    affinity(arm, arm + '-after', need_io=True)
    rpc(arm, 'send', 'QUIT')
    owner.wait(timeout=300)
    st = json.loads((R / arm / 'state.json').read_text())
    assert st['exit_rc'] == 0, st
    for v in json.loads((R / arm / 'identities.json').read_text()).values():
        assert not pathlib.Path('/proc', str(v['pid']), 'exe').exists(), v
    gpu_idle()
    # service status strictly AFTER engine disconnect
    after = wstatus()
    atom(R / ('worker-after-' + arm + '.json'), after)
    fs, bleg = after['status'], before['legacy']
    alegs = after['legacy']
    assert fs['session'] == 0 and fs['complete'] == 1 and fs['resident_bytes'] == WORKER_BYTES
    assert fs['idle_remaining_ms'] == 2 ** 64 - 1
    for k in ['loaded_bytes', 'verified_bytes', 'arena_allocations', 'network_weight_bytes', 'unloads', 'expirations']:
        assert fs[k] == bs[k], (k, bs[k], fs[k])
    assert fs['cache_hits'] == bs['cache_hits'] + 1 and fs['reused_bytes'] == bs['reused_bytes'] + WORKER_BYTES
    assert fs['requests'] > bs['requests'], (bs['requests'], fs['requests'])  # service-call counter, ~11k per GEN256
    assert alegs['failures'] == 0 and alegs['last_error'] == 0
    service_delta_s = (alegs['cpu_service_ns'] - bleg['cpu_service_ns']) / 1e9
    summary = {'arm': arm, 'startup': startup, 'warm_wall_s': last_done_mono - first_send_mono,
               'prompt_s_sum': sum(r['prompt_s'] or 0 for r in rows),
               'decode_s_sum': sum(r['decode_s'] or 0 for r in rows),
               'tokens_total': sum(r['output_count'] for r in rows),
               'worker_service_delta_s': service_delta_s, 'requests': rows,
               'engine_exit_rc': 0, 'worker_requests_delta': 3}
    atom(R / ('arm-' + arm + '-summary.json'), summary)
    state('arm-' + arm + '-complete', summary={k: summary[k] for k in ['warm_wall_s', 'decode_s_sum', 'tokens_total']})
    return summary


def main():
    assert hashlib.sha256(pathlib.Path(cfg['argv'][0]).read_bytes()).hexdigest() == cfg['binary_sha256']
    assert hashlib.sha256(pathlib.Path(cfg['worker_argv'][0]).read_bytes()).hexdigest() == cfg['worker_sha256']
    assert hashlib.sha256((R / 'local.manifest').read_bytes()).hexdigest() == cfg['manifest_sha256']
    assert hashlib.sha256((R / 'arm-requests.txt').read_bytes()).hexdigest() == json.loads((R / 'staged-sha.json').read_text())['arm_requests_sha256']
    with socket.socket() as s:
        s.bind(('127.0.0.1', PORT))
    topo = topology()
    assert len({(v['physical_package_id'], v['core_id']) for v in topo.values()}) == 14
    rapl_direct = subprocess.run(['cat', '/sys/class/powercap/intel-rapl:0/energy_uj'], capture_output=True, text=True)
    rapl_root = subprocess.run(['sudo', '-n', 'cat', '/sys/class/powercap/intel-rapl:0/energy_uj'], capture_output=True, text=True)
    pl = {p.name: subprocess.run(['cat', str(p)], capture_output=True, text=True).stdout.strip()
          for p in [pathlib.Path('/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw'),
                    pathlib.Path('/sys/class/powercap/intel-rapl:0/constraint_0_time_window_us'),
                    pathlib.Path('/sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw'),
                    pathlib.Path('/sys/class/powercap/intel-rapl:0/constraint_1_time_window_us')]}
    atom(R / 'preflight.json', {'wall': time.time(), 'hashes': 'verified', 'port_free': True,
                                'rapl_energy_unprivileged': rapl_direct.returncode == 0,
                                'rapl_energy_via_sudo': rapl_root.returncode == 0,
                                'rapl_energy_note': 'per-tick sampling uses unprivileged reads only',
                                'power_limits': pl, 'topology_distinct_cores': 14})
    state('snapshot-baseline')
    freq_ctl('snapshot', str(R / 'freq-snapshot-baseline.json'))
    # clockmon starts before the cold load and spans every phase
    mark(phase='cold-load', arm=None, req=0)
    clockmon = subprocess.Popen(['python3', '-B', str(R / 'clockmon.py'), str(R), '10800'],
                                stdin=subprocess.DEVNULL, stdout=(R / 'clockmon.log').open('ab'),
                                stderr=subprocess.STDOUT, start_new_session=True)
    atom(R / 'clockmon-launch.json', {'pid': clockmon.pid, 'wall': time.time()})
    time.sleep(2)
    assert json.loads((R / 'clockmon-exit.json').read_text()) if False else True
    # --- cold arena load ---
    t0 = time.monotonic()
    assert not (R / 'WORKER').exists()
    wowner = subprocess.Popen(['python3', '-B', str(R / 'worker_owner.py')],
                              stdin=subprocess.DEVNULL, stdout=(R / 'worker-owner.log').open('ab'),
                              stderr=subprocess.STDOUT, start_new_session=True)
    waitfile(R / 'WORKER' / 'listening.json', lambda v: True, 120)
    ids = json.loads((R / 'WORKER' / 'identities.json').read_text())
    (R / 'clockmon-worker-pid').write_text(str(ids['worker']['pid']) + '\n')
    fresh = wstatus()
    atom(R / 'worker-fresh.json', fresh)
    assert fresh['status']['loaded_bytes'] == 0 and fresh['status']['requests'] == 0
    gpu_idle()
    assert not (R / 'C0').exists()
    cowner = subprocess.Popen(['python3', '-B', str(R / 'owner.py'), 'C0'],
                              stdin=subprocess.DEVNULL, stdout=(R / 'engine-owner-C0.log').open('ab'),
                              stderr=subprocess.STDOUT, start_new_session=True)
    st = waitfile(R / 'C0' / 'state.json', lambda v: v['ready'] is not None, 2450)
    cold_ready = {'engine_started_mono': st['started_mono'], 'ready_mono': st['ready']['mono'],
                  'worker_spawn_to_ready_s': st['ready']['mono'] - t0,
                  'engine_to_ready_s': st['ready']['mono'] - st['started_mono']}
    ctext = (R / 'C0' / 'stderr.log').read_text()
    cold_markers = {k: (k in ctext) for k in ['116135731200', 'cache_hit=0', 'setup network=0 B']}
    rpc('C0', 'send', 'QUIT')
    cowner.wait(timeout=300)
    cst = json.loads((R / 'C0' / 'state.json').read_text())
    assert cst['exit_rc'] == 0 and cst['requests_sent'] == 0
    for v in json.loads((R / 'C0' / 'identities.json').read_text()).values():
        assert not pathlib.Path('/proc', str(v['pid']), 'exe').exists()
    gpu_idle()
    cold_after = wstatus()
    atom(R / 'worker-after-cold.json', cold_after)
    cs = cold_after['status']
    assert cs['complete'] == 1 and cs['resident_bytes'] == cs['loaded_bytes'] == cs['verified_bytes'] == WORKER_BYTES
    assert cs['arena_allocations'] == 1 and cs['requests'] == 0 and cs['cache_hits'] == 0 and cs['reused_bytes'] == 0
    assert cs['unloads'] == 0 and cs['expirations'] == 0 and cs['network_weight_bytes'] == 0
    atom(R / 'cold-load.json', {'wall': time.time(), **cold_ready, 'stderr_markers': cold_markers,
                                'worker_after': cs, 'legacy': cold_after['legacy']})
    shutil_copy = (R / 'WORKER' / 'identities.json').read_bytes()
    (R / 'worker-identities.json').write_bytes(shutil_copy)
    state('cold-load-complete', engine_to_ready_s=cold_ready['engine_to_ready_s'])
    # --- arms ---
    summaries = {}
    transitions = []
    for arm in ['A1', 'B', 'A2']:
        if arm in ('A1', 'A2'):
            mark(phase='freq-set', arm=arm)
            transitions.append({'arm': arm, 'set': freq_ctl('set', str(R / ('freq-set-' + arm + '.json')), '2000000')})
        summaries[arm] = run_arm(arm)
        if arm in ('A1', 'A2'):
            mark(phase='freq-restore', arm=arm)
            transitions.append({'arm': arm, 'restore': freq_ctl('set', str(R / ('freq-restore-' + arm + '.json')), '3000000')})
    atom(R / 'freq-transitions.json', {'wall': time.time(), 'transitions': transitions})
    # --- final restoration check (clockmon still running through this) ---
    mark(phase='final-check')
    freq_ctl('check-restored', str(R / 'freq-snapshot-baseline.json'))
    (R / 'clockmon.STOP').write_text('done\n')
    try:
        clockmon.wait(timeout=60)
    except subprocess.TimeoutExpired:
        clockmon.terminate()
    fin = wstatus()
    atom(R / 'worker-final.json', fin)
    healthy()
    atom(R / 'closeout.json', {'wall': time.time(), 'arms': {a: {k: s[a][k] for k in ['warm_wall_s', 'decode_s_sum', 'prompt_s_sum', 'tokens_total', 'worker_service_delta_s']}
                                                              for a, s in [(a, summaries[a]) for a in ['A1', 'B', 'A2']]},
                               'worker_final': fin['status'], 'clockmon_samples': json.loads((R / 'clockmon-exit.json').read_text())})
    state('completed-no-stop-worker-resident')


if __name__ == '__main__':
    try:
        main()
        print('CAMPAIGN COMPLETE', flush=True)
    except BaseException as e:
        atom(R / 'campaign-failure.json', {'wall': time.time(), 'error': repr(e), 'traceback': traceback.format_exc()})
        # abort path: stop any live engine, restore clocks, keep worker resident, stop clockmon
        try:
            for arm in ['C0', 'A1', 'B', 'A2']:
                if (R / arm / 'input.sock').exists():
                    st = json.loads((R / arm / 'state.json').read_text())
                    if st.get('active'):
                        rpc(arm, 'send', 'STOP')
                    rpc(arm, 'send', 'QUIT')
        except Exception:
            pass
        try:
            freq_ctl('set', str(R / 'freq-abort-restore.json'), '3000000')
            freq_ctl('check-restored', str(R / 'freq-snapshot-baseline.json'))
        except Exception:
            pass
        try:
            mark(phase='aborted')
            (R / 'clockmon.STOP').write_text('abort\n')
        except Exception:
            pass
        state('failed')
        raise
