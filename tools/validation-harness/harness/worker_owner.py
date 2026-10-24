import os,sys,json,time,signal,pathlib,subprocess,threading,hashlib
R=pathlib.Path(__file__).resolve().parent;D=R/'WORKER';D.mkdir(exist_ok=False)
cfg=json.loads((R/'config.json').read_text());lock=threading.RLock()
def atom(p,v):
 t=str(p)+'.tmp';pathlib.Path(t).write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def identity(pid):
 p=pathlib.Path('/proc',str(pid));s=(p/'stat').read_text().rsplit(')',1)[1].split();e=(p/'exe').stat()
 return dict(pid=pid,start_ticks=int(s[19]),exe_dev=e.st_dev,exe_ino=e.st_ino,exe=os.readlink(p/'exe'),argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip())
assert hashlib.sha256(pathlib.Path(cfg['worker_argv'][0]).read_bytes()).hexdigest()==cfg['worker_sha256']
assert hashlib.sha256((R/'local.manifest').read_bytes()).hexdigest()==cfg['manifest_sha256']
state={'phase':'service','wall':time.time(),'failure':None,'no_expiry':True}
atom(D/'launch.json',{'argv':cfg['worker_argv'],'env':{'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','OMP_NUM_THREADS':'2'},'wall':time.time(),'mono':time.monotonic()})
w=subprocess.Popen(cfg['worker_argv'],env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','OMP_NUM_THREADS':'2'},stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,bufsize=0)
ids={'worker':identity(w.pid),'guardian':identity(os.getpid())};atom(D/'identities.json',ids)
def fail(reason):
 with lock:
  state['failure']=reason;atom(D/'state.json',state)
 if w.poll() is None:
  try:os.killpg(w.pid,signal.SIGTERM)
  except ProcessLookupError:pass
def drain(pipe,name):
 n=0
 with (D/name).open('wb',buffering=0) as f:
  for line in iter(pipe.readline,b''):
   n+=len(line)
   if n>32*1024*1024:fail('bounded worker log overflow: '+name);continue
   f.write(line)
   if b'listening on 127.0.0.1:' in line:atom(D/'listening.json',{'wall':time.time(),'line':line.decode().strip(),'worker':ids['worker']})
   if any(t in line.lower() for t in [b'out of memory',b'illegal memory access']):fail('worker memory fault: '+line.decode())
for p,n in [(w.stdout,'stdout.log'),(w.stderr,'stderr.log')]:threading.Thread(target=drain,args=(p,n),daemon=True).start()
m=subprocess.Popen(['/usr/bin/python3','-B',str(R/'monitor.py'),str(D),str(os.getpid()),str(w.pid)],stdin=subprocess.DEVNULL,stdout=(D/'monitor.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
ids['monitor']=identity(m.pid);atom(D/'identities.json',ids)
try:
 while w.poll() is None:
  state['heartbeat_wall']=time.time();atom(D/'state.json',state)
  if m.poll() is not None:fail('independent worker monitor exited')
  hp=D/'health.json'
  if hp.exists() and time.time()-hp.stat().st_mtime>30:fail('worker monitor stale >30s')
  if state['failure']:
   try:w.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL)
   break
  time.sleep(1)
 rc=w.wait();state.update(phase='exited',exit_rc=rc,exit_wall=time.time());atom(D/'state.json',state)
finally:
 if w.poll() is None:
  fail('guardian cleanup')
  try:w.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
 try:m.wait(timeout=15)
 except subprocess.TimeoutExpired:m.terminate();m.wait(timeout=10)
