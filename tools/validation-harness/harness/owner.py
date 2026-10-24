import os, sys, json, time, socket, subprocess, threading, signal, hashlib, pathlib, selectors
R = pathlib.Path(__file__).resolve().parent
ARM = sys.argv[1]
D = R / ARM
D.mkdir(exist_ok=False)
CFG = json.loads(pathlib.Path(sys.argv[2]).read_text())
LOCK = threading.RLock()
STATE = {'arm': ARM, 'phase': 'startup', 'started_wall': time.time(), 'started_mono': time.monotonic(), 'ready': None, 'done': None, 'requests_sent': 0, 'completed': 0, 'active': False, 'failure': None}
def identity(pid):
    p = pathlib.Path('/proc') / str(pid)
    st = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    e = (p / 'exe').stat()
    return {'pid': pid, 'start_ticks': int(st[19]), 'exe_dev': e.st_dev, 'exe_ino': e.st_ino, 'exe': os.readlink(p/'exe'), 'argv': (p/'cmdline').read_bytes().decode().split('\0')[:-1], 'boot_id': pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
def atomic(path, obj):
    tmp = str(path)+'.tmp'
    pathlib.Path(tmp).write_text(json.dumps(obj, indent=2)+'\n')
    os.replace(tmp, path)
def event(kind, **fields):
    row = {'kind':kind, 'wall':time.time(), 'mono':time.monotonic(), **fields}
    with LOCK:
        with (D/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    return row
def save():
    with LOCK: atomic(D/'state.json', STATE)
def kill_engine(reason):
    with LOCK:
        if STATE['failure'] is None:
            STATE['failure'] = reason
            event('FAILURE', reason=reason)
        save()
    if ENGINE.poll() is None:
        try: os.killpg(ENGINE.pid, signal.SIGTERM)
        except ProcessLookupError: pass
ENGINE = MONITOR = server = SOCK = None
DRAINERS = []
def terminate_owner(sig, frame):
    raise SystemExit(128 + sig)
signal.signal(signal.SIGTERM, terminate_owner)
signal.signal(signal.SIGINT, terminate_owner)
try:
    BINARY = pathlib.Path(CFG['argv'][0])
    assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == CFG['binary_sha256']
    ARGV = CFG['argv']
    ENV = dict(CFG['env'])
    ENV['CUDA_VISIBLE_DEVICES'] = ','.join(CFG['gpu_uuids'])
    assert 'STRATA_ARENA_MMAP' not in ENV and 'STRATA_PRIMARY_DEVICE' not in ENV
    assert not any(a.startswith('--remote-gpu-') or a in ['--gpu'] for a in ARGV)
    atomic(D/'launch.json', {'argv':ARGV,'env':ENV,'limits':pathlib.Path('/proc/self/limits').read_text(),'wall':time.time(),'mono':time.monotonic()})
    ENGINE = subprocess.Popen(ARGV, env=ENV, cwd=str(R), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
    STATE['engine'] = identity(ENGINE.pid)
    STATE['broker'] = identity(os.getpid())
    atomic(D/'identities.json', {key: STATE[key] for key in ('engine', 'broker')})
    # No experiment-controller pipe owns the engine input. This independent broker retains the writer.
    def drain(pipe, name):
        size = 0
        with (D/name).open('wb', buffering=0) as f:
            for line in iter(lambda: pipe.readline(65536), b''):
                if len(line) == 65536 and not line.endswith(b'\n'):
                    kill_engine('bounded output line exceeded 64 KiB'); continue
                size += len(line)
                if size > 32*1024*1024:
                    kill_engine('bounded output exceeded 32 MiB: '+name)
                    continue
                f.write(line)
                text = line.decode(errors='replace').rstrip()
                if name == 'stdout.log' and (text.startswith('READY ') or text.startswith('DONE ') or text.startswith('ERR ')):
                    with LOCK:
                        row = event(text.split()[0], line=text)
                        if text.startswith('READY '):
                            try: os.sched_setaffinity(ENGINE.pid, set(CFG['engine_main_mask']))
                            except OSError as error:
                                kill_engine('configured engine-main mask failed: ' + str(error)); return
                            STATE['ready'] = row
                            STATE['phase'] = 'idle'
                        elif text.startswith('DONE '):
                            STATE['done'] = row
                            STATE['completed'] += 1
                            STATE['active'] = False
                            STATE['phase'] = 'idle'
                        else:
                            STATE['failure'] = 'engine protocol error: '+text
                        save()
                if name == 'stderr.log' and any(t in text.lower() for t in ['out of memory','cuda error','illegal memory access','device-side assert']):
                    kill_engine('GPU/memory fault: '+text)
    for name,pipe in [('stdout.log',ENGINE.stdout),('stderr.log',ENGINE.stderr)]:
        thread = threading.Thread(target=drain,args=(pipe,name),daemon=True)
        DRAINERS.append(thread); thread.start()
    MONITOR = subprocess.Popen([sys.executable,'-B',str(R/'monitor.py'),str(D),str(os.getpid()),str(ENGINE.pid), str(pathlib.Path(sys.argv[2]))], stdin=subprocess.DEVNULL, stdout=(D/'monitor.log').open('ab'), stderr=subprocess.STDOUT, env=CFG['monitor_env'], start_new_session=True)
    STATE['monitor'] = identity(MONITOR.pid)
    atomic(D/'identities.json', {k:STATE[k] for k in ['engine','broker','monitor']})
    SOCK = D/'input.sock'
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCK)); os.chmod(SOCK,0o600);server.listen(4);server.settimeout(1)
    save()
    quitting = False
    try:
        while ENGINE.poll() is None:
            with LOCK:
                STATE['heartbeat_wall'] = time.time(); save()
            if MONITOR.poll() is not None:
                kill_engine('independent monitor exited')
            hp = D/'health.json'
            if hp.exists() and time.time()-hp.stat().st_mtime > 30:
                kill_engine('independent monitor stale >30s')
            if STATE['phase'] == 'startup' and time.monotonic()-STATE['started_mono'] > CFG['startup_seconds']:
                kill_engine('startup exceeded 2400s')
            if STATE['active'] and time.monotonic()-STATE['sent']['mono'] > CFG['request_seconds']:
                kill_engine('request exceeded 1800s')
            if STATE['failure']:
                try: ENGINE.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(ENGINE.pid, signal.SIGKILL)
                break
            try: c,_ = server.accept()
            except socket.timeout: continue
            with c:
                c.settimeout(3)
                try:
                    data=b''
                    while b'\n' not in data and len(data)<16384:
                        chunk=c.recv(16384)
                        if not chunk:break
                        data += chunk
                    req=json.loads(data)
                    with LOCK:
                        if req['op']=='inspect': reply=dict(STATE)
                        elif req['op']=='send':
                            line=req['line']
                            assert '\n' not in line
                            assert line in ['STOP','QUIT'] or line in CFG['request_lines']
                            if line.startswith('GEN '):
                                assert STATE['ready'] and not STATE['active'] and not STATE['failure']
                                assert STATE['requests_sent'] < CFG['max_requests']
                                STATE['requests_sent'] += 1
                                STATE['active']=True; STATE['phase']='request'
                            STATE['sent']=event('SEND',line=line)
                            ENGINE.stdin.write((line+'\n').encode()); ENGINE.stdin.flush()
                            if line=='QUIT': quitting=True; STATE['phase']='quitting'
                            save();reply={'accepted':True,'sent':STATE['sent'],'request':STATE['requests_sent']}
                        else:raise ValueError('unknown broker operation')
                    c.sendall((json.dumps(reply)+'\n').encode())
                except Exception as e:
                    c.sendall((json.dumps({'error':repr(e)})+'\n').encode())
        rc=ENGINE.wait()
        STATE['exit_rc']=rc; STATE['phase']='exited'; STATE['exit_wall']=time.time()
        event('EXIT',rc=rc);save()
    finally:
        if ENGINE.poll() is None:
            kill_engine('broker cleanup')
            try: ENGINE.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(ENGINE.pid,signal.SIGKILL);ENGINE.wait()
        server.close()
        try:SOCK.unlink()
        except FileNotFoundError:pass
        # The monitor sees engine exit and writes a final resource receipt before exiting.
        try:MONITOR.wait(timeout=15)
        except subprocess.TimeoutExpired:MONITOR.terminate();MONITOR.wait(timeout=10)
except BaseException as error:
    STATE['failure'] = STATE.get('failure') or 'owner setup/runtime failure: ' + repr(error)
    raise
finally:
    # Covers identity/monitor/socket setup too, before the retained service loop's own finally.
    if ENGINE is not None:
        if ENGINE.poll() is None:
            kill_engine('owner cleanup')
            try: ENGINE.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(ENGINE.pid, signal.SIGKILL); ENGINE.wait()
        STATE.update(phase='exited', exit_rc=ENGINE.returncode, exit_wall=time.time())
    if MONITOR is not None and MONITOR.poll() is None:
        try: MONITOR.wait(timeout=15)
        except subprocess.TimeoutExpired: MONITOR.terminate(); MONITOR.wait(timeout=10)
    if MONITOR is not None:
        STATE['monitor_exit_rc'] = MONITOR.returncode
        if MONITOR.returncode != 0: STATE['failure'] = STATE.get('failure') or 'independent monitor exited unsuccessfully'
    if server is not None: server.close()
    if SOCK is not None:
        try: SOCK.unlink()
        except FileNotFoundError: pass
    for thread in DRAINERS:
        thread.join(timeout=5)
        if thread.is_alive(): STATE['failure'] = 'output drainer did not exit'
    if (D/'monitor-failure.json').exists(): STATE['failure'] = 'independent monitor reported failure'
    save()
if STATE.get('failure') or STATE.get('exit_rc') != 0:
    raise SystemExit(1)
