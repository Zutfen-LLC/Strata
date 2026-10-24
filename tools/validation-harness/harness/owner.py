import os, sys, json, time, socket, subprocess, threading, signal, hashlib, pathlib, selectors
R = pathlib.Path(__file__).resolve().parent
ARM = sys.argv[1]
D = R / ARM
D.mkdir(exist_ok=False)
CFG = json.loads((R / 'config.json').read_text())
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
BINARY = pathlib.Path(CFG['argv'][0])
assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == CFG['binary_sha256']
ARGV = CFG['argv']
ENV = dict(CFG['env'])
ENV['CUDA_VISIBLE_DEVICES'] = ','.join(CFG['gpu_uuids'][:1 if ARM == 'L1' else 2])
assert 'STRATA_ARENA_MMAP' not in ENV and 'STRATA_PRIMARY_DEVICE' not in ENV
assert not any(a.startswith('--remote-gpu-') or a in ['--gpu'] for a in ARGV)
atomic(D/'launch.json', {'argv':ARGV,'env':ENV,'limits':pathlib.Path('/proc/self/limits').read_text(),'wall':time.time(),'mono':time.monotonic()})
ENGINE = subprocess.Popen(ARGV, env=ENV, cwd=str(R), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
STATE['engine'] = identity(ENGINE.pid)
STATE['broker'] = identity(os.getpid())
atomic(D/'identities.json', STATE.copy())
# No experiment-controller pipe owns the engine input. This independent broker retains the writer.
def drain(pipe, name):
    size = 0
    with (D/name).open('wb', buffering=0) as f:
        for line in iter(pipe.readline, b''):
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
    threading.Thread(target=drain,args=(pipe,name),daemon=True).start()
MONITOR = subprocess.Popen(['/usr/bin/python3','-B',str(R/'monitor.py'),str(D),str(os.getpid()),str(ENGINE.pid)], stdin=subprocess.DEVNULL, stdout=(D/'monitor.log').open('ab'), stderr=subprocess.STDOUT, start_new_session=True)
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
        if STATE['phase'] == 'startup' and time.monotonic()-STATE['started_mono'] > 2400:
            kill_engine('startup exceeded 2400s')
        if STATE['active'] and time.monotonic()-STATE['sent']['mono'] > 1800:
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
                        assert line in ['STOP','QUIT'] or line.startswith('GEN 256 ')
                        if line.startswith('GEN '):
                            assert STATE['ready'] and not STATE['active'] and not STATE['failure']
                            assert STATE['requests_sent'] < 6
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
