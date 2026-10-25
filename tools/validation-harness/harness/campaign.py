"""Shared retained campaign operations; imported without reading config or starting processes.

Extracted from sanitized source 27392d03. Sequencing and machine bindings live in run.py/config.
"""
import os, sys, json, time, socket, struct, subprocess, pathlib
import worker_control

R = pathlib.Path(__file__).resolve().parent
cfg = {}
SESSIONS = {}  # retained Popen objects, removed only after reap and absence checks
GUARDIAN = None

def configure(root, configuration):
    global R, W, cfg, PORT, WORKER_BYTES, POOL_COUNT, ARM_LINES
    R = pathlib.Path(root); W = pathlib.Path(configuration['worker_root']); cfg = configuration
    PORT = cfg['port']; WORKER_BYTES = cfg['worker_bytes']; POOL_COUNT = cfg['worker_count']
    ARM_LINES = cfg['request_lines']

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
            if len(b) > 65536: raise RuntimeError('bounded broker reply exceeded 64 KiB')
        v = json.loads(b)
        assert 'error' not in v, v
        return v


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
    for c in range(os.cpu_count() or 1):
        p = pathlib.Path('/sys/devices/system/cpu', 'cpu' + str(c), 'topology')
        t[str(c)] = {k: (p / k).read_text().strip() for k in ['physical_package_id', 'core_id', 'thread_siblings_list']}
    return t


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, env=cfg['monitor_env'], timeout=10)


def gpu_idle():
    g = gpu()
    assert all(int(l.split(',')[1]) < 100 for l in g.splitlines()), g
    return g


def listener_owned():
    ids = json.loads((W / 'identities.json').read_text())
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


def assert_worker_idle():
    # Even an idle READY engine owns the worker connection. An exited broker remains here until
    # close_session has reaped it AND checked its child/monitor identities, including failure paths.
    if SESSIONS:
        raise RuntimeError('worker controls forbidden until engine sessions are closed: ' + ','.join(SESSIONS))
    # Attached legacy roots also have identity receipts. Inspect files only, never enqueue a
    # STATUS probe to discover whether the single-connection worker is occupied.
    for path in W.parent.glob('*/identities.json'):
        row = json.loads(path.read_text()).get('engine')
        if row:
            try: ident(row)
            except (AssertionError, FileNotFoundError, ProcessLookupError): continue
            raise RuntimeError('worker control blocked by live recorded engine: ' + str(path.parent))


def wstatus():
    assert_worker_idle()
    listener_owned()
    with socket.create_connection((cfg['worker_host'], PORT), timeout=5) as s:
        cache = worker_control.status(s)
        vals = struct.unpack('<7QHB', worker_control.exchange(s, 9, 10))
        keys = ['session', 'setup_weight_bytes', 'infer_in_bytes', 'infer_out_bytes', 'requests',
                'failures', 'cpu_service_ns', 'last_error', 'ready']
        return {'wall': time.time(), 'status': cache, 'legacy': dict(zip(keys, vals))}


def wait_session(directory, predicate, seconds, owner):
    path = pathlib.Path(directory) / 'state.json'
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rc = owner.poll()
        if rc is not None:
            rc = owner.wait()  # retain/reap the actual child, including death before state.json
            raise RuntimeError('owner exited before completion: rc=' + str(rc))
        if path.exists():
            row = json.loads(path.read_text())
            if row.get('failure'): raise RuntimeError('owner failure: ' + str(row['failure']))
            if predicate(row): return row
            engine = row.get('engine')
            if engine and not pathlib.Path('/proc', str(engine['pid']), 'exe').exists():
                raise RuntimeError('engine vanished without completed owner shutdown')
        time.sleep(0.1)
    raise TimeoutError('owner deadline: ' + str(path))


def wait_absent(identities, seconds):
    end = time.monotonic() + seconds
    while True:
        alive = []
        for row in identities.values():
            try: ident(row); alive.append(row['pid'])
            except (FileNotFoundError, ProcessLookupError): pass
        if not alive: return
        if time.monotonic() >= end: raise TimeoutError('owned processes remain after owner reap: ' + str(alive))
        time.sleep(0.1)


def finish_session(arm, abort=False):
    owner = SESSIONS[arm]
    if owner.poll() is None:
        if abort:
            owner.terminate()  # broker's existing finally path owns engine/monitor cleanup
        else:
            row = json.loads((R / arm / 'state.json').read_text())
            if row.get('active'): raise RuntimeError('cannot QUIT an active request')
            rpc(arm, 'send', 'QUIT')
    rc = owner.wait(timeout=cfg['shutdown_seconds'])
    atom(R / (arm + '-owner-exit.json'), {'pid': owner.pid, 'returncode': rc})
    ids = json.loads((R / arm / 'identities.json').read_text())
    wait_absent(ids, cfg['cleanup_seconds'])
    del SESSIONS[arm]
    final = json.loads((R / arm / 'state.json').read_text())
    if not abort and (rc or final.get('failure') or final.get('exit_rc') != 0):
        raise RuntimeError('engine owner failed: ' + str(rc) + '; child=' + str(final.get('exit_rc')))


def port_free(host, port, seconds):
    # Same SO_REUSEADDR policy as the real listener; TIME_WAIT is not a live listener.
    end = time.monotonic() + seconds
    while True:
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
            return
        except OSError:
            if time.monotonic() >= end: raise
            time.sleep(0.1)


def stop_owned_worker():
    global GUARDIAN
    assert_worker_idle()
    if GUARDIAN is None: raise RuntimeError('cannot stop an attached guardian without its retained Popen owner')
    already_exited = GUARDIAN.poll()
    if already_exited is not None: GUARDIAN.wait()  # reap even if identity collection/control is no longer possible
    ids = json.loads((W / 'identities.json').read_text())
    if ids['guardian']['pid'] != GUARDIAN.pid: raise RuntimeError('guardian receipt differs from retained child')
    path = W / 'stop.json'
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt['identities'] != ids: raise RuntimeError('STOP receipt identifies another worker')
    elif already_exited is not None:
        receipt = {'operation': 'stop', 'outcome': 'observed-exited', 'control_outcome': 'not-sent', 'identities': ids}
    else:
        for row in ids.values(): ident(row)
        listener_owned()
        receipt = {'operation': 'stop', 'outcome': 'pending', 'identities': ids}
        atom(path, receipt)
        try:
            with socket.create_connection((cfg['worker_host'], PORT), timeout=5) as s:
                if worker_control.exchange(s, 11, 10): raise RuntimeError('unexpected stop payload')
            receipt.update(outcome='acknowledged', control_outcome='acknowledged')
        except (OSError, RuntimeError, TimeoutError) as error:
            # A sent STOP can queue and execute later. Reconcile via this retained child; never send
            # another STOP after a missing ACK, even when this function is called again.
            receipt.update(outcome='uncertain', control_outcome='uncertain', error=str(error))
    atom(W / 'stop.json', receipt)
    try:
        rc = GUARDIAN.wait(timeout=cfg['shutdown_seconds'])
    except subprocess.TimeoutExpired:
        receipt['outcome'] = 'uncertain'; atom(W / 'stop.json', receipt)
        raise
    receipt.update(guardian_returncode=rc, reaped=True)
    atom(W / 'stop.json', receipt)
    wait_absent(ids, cfg['cleanup_seconds'])
    if rc: raise RuntimeError('guardian exited unsuccessfully: ' + str(rc))
    final = json.loads((W / 'state.json').read_text())
    if final.get('failure') or final.get('phase') != 'exited' or final.get('exit_rc') != 0:
        raise RuntimeError('worker did not exit cleanly: ' + str(final))
    receipt['outcome'] = 'completed'; atom(W / 'stop.json', receipt)
    GUARDIAN = None
    port_free(cfg['worker_host'], PORT, cfg['cleanup_seconds'])  # only AFTER owned stop/reap
    return receipt


def abort_owned_worker_startup(reason):
    """Abort only a just-launched guardian, before startup has established a control listener."""
    global GUARDIAN
    assert_worker_idle()
    if GUARDIAN is None or cfg.get('attach_root'):
        raise RuntimeError('cannot abort an attached worker without its retained launch handle')
    if (W / 'stop.json').exists():
        raise RuntimeError('cannot startup-abort a worker with an existing STOP intent')
    path = R / 'worker-startup-abort.json'
    receipt = {'operation': 'startup-abort', 'outcome': 'pending', 'error': repr(reason),
               'guardian_pid': GUARDIAN.pid, 'control_outcome': 'not-sent'}
    atom(path, receipt)
    try:
        if GUARDIAN.poll() is None:
            GUARDIAN.terminate()  # existing guardian finally owns worker/monitor cleanup and reaping
        rc = GUARDIAN.wait(timeout=cfg['shutdown_seconds'])
        receipt.update(guardian_returncode=rc, reaped=True)
        atom(path, receipt)
        ids = json.loads((W / 'identities.json').read_text())
        if set(ids) != {'guardian', 'worker', 'monitor'}:
            raise RuntimeError('incomplete startup identity receipt; child absence is uncertain')
        if ids['guardian']['pid'] != GUARDIAN.pid:
            raise RuntimeError('startup guardian receipt differs from retained child')
        receipt['identities'] = ids
        wait_absent(ids, cfg['cleanup_seconds'])  # only after the actual guardian was reaped
        receipt['outcome'] = 'completed'
        atom(path, receipt)
        GUARDIAN = None
        return receipt
    except BaseException as cleanup:
        receipt.update(outcome='uncertain', cleanup_error=repr(cleanup))
        atom(path, receipt)
        raise


def healthy():
    ids = json.loads((W / 'identities.json').read_text())
    for row in ids.values(): ident(row)
    health = json.loads((W / 'health.json').read_text())
    if time.time() - health['wall'] >= 15 or health.get('failure'):
        raise RuntimeError('worker health stale or failed')
    return health


def launch_session(arm, session_config):
    if arm in SESSIONS or (R / arm).exists(): raise RuntimeError('session output already exists: ' + arm)
    path = R / (arm + '.config.json')
    with path.open('x') as out: json.dump(session_config, out, indent=2)
    owner = subprocess.Popen([sys.executable, '-B', str(R / 'owner.py'), arm, str(path)],
        stdin=subprocess.DEVNULL, stdout=(R / ('engine-owner-' + arm + '.log')).open('xb'),
        stderr=subprocess.STDOUT, start_new_session=True)
    SESSIONS[arm] = owner
    return wait_session(R / arm, lambda row: row['ready'] is not None, cfg['startup_seconds'], owner)


def run_arm(arm, session_config, indices, affinity):
    """Retained READY/GEN/DONE/QUIT sequence, with config-selected requests and no worker control
    between engine launch and owner reap. Cold load uses indices=[]; its timing stays separate."""
    healthy(); before = wstatus(); gpu_idle()
    atom(R / ('worker-before-' + arm + '.json'), before)
    st = launch_session(arm, session_config)
    affinity(arm, False, phase='ready')  # loader helpers have joined; final compute masks precede every GEN
    startup = st['ready']['mono'] - st['started_mono']
    text = (R / arm / 'stderr.log').read_text()
    for marker in session_config['placement_markers']:
        if marker not in text: raise RuntimeError('missing placement marker: ' + marker)
    rows = []; io_tid = None
    for k, index in enumerate(indices):
        healthy(); affinity(arm, False)
        path = R / arm / 'stdout.log'; offset = path.stat().st_size
        ack = rpc(arm, 'send', ARM_LINES[index])
        st = wait_session(R / arm, lambda row: row['completed'] == k + 1,
                          cfg['request_seconds'], SESSIONS[arm])
        if st['active'] or st['requests_sent'] != k + 1: raise RuntimeError('request ordering mismatch')
        done = parse_done(st['done']['line']); log = path.read_text()
        tokens = tokens_between(log, offset, len(log))
        if len(tokens) != done['output_count']: raise RuntimeError('token count mismatch')
        report = affinity(arm, True)
        tids = report['communication_threads']
        if len(tids) != 1: raise RuntimeError('missing unique communication thread')
        tid = tids[0]['tid']
        if io_tid is not None and tid != io_tid: raise RuntimeError('communication TID changed')
        io_tid = tid
        rows.append({'index': index + 1, 'tokens': tokens, 'output_count': done['output_count'],
            'worker_entries': done.get('worker_entries'), 'prompt_s': done['prompt_s'],
            'decode_s': done['decode_s'], 'sent_mono': ack['sent']['mono'], 'done_mono': st['done']['mono']})
    finish_session(arm)
    gpu_idle(); after = wstatus(); atom(R / ('worker-after-' + arm + '.json'), after)
    status = after['status']
    if not status['complete'] or status['resident_bytes'] != WORKER_BYTES or status['session']:
        raise RuntimeError('incomplete resident arena after closed engine session')
    if status['loaded_bytes'] != WORKER_BYTES or status['verified_bytes'] != WORKER_BYTES or status['arena_allocations'] != 1:
        raise RuntimeError('unexpected arena/load/verification accounting')
    if status['network_weight_bytes'] or status['unloads'] or status['expirations'] or after['legacy']['failures']:
        raise RuntimeError('unexpected upload/unload/expiry/failure')
    if status['idle_remaining_ms'] != 2**64 - 1: raise RuntimeError('service-owned residency not retained')
    if indices:
        for key in ['loaded_bytes','verified_bytes','arena_allocations','unloads','expirations']:
            if status[key] != before['status'][key]: raise RuntimeError('resident work changed: ' + key)
    summary = {'arm': arm, 'startup_to_READY_s': startup, 'requests': rows,
        'tokens_total': sum(row['output_count'] for row in rows),
        'decode_s_sum': sum(row['decode_s'] for row in rows), 'worker_status': status}
    atom(R / ('arm-' + arm + '-summary.json'), summary)
    return summary
