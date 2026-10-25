import os,sys,json,time,signal,pathlib,subprocess,threading,hashlib,select
R=pathlib.Path(__file__).resolve().parent
cfg=json.loads(pathlib.Path(sys.argv[1]).read_text());D=pathlib.Path(cfg['worker_root']);D.mkdir(exist_ok=False);lock=threading.RLock()
def atom(p,v):
 t=str(p)+'.tmp';pathlib.Path(t).write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def identity(pid):
 p=pathlib.Path('/proc',str(pid));s=(p/'stat').read_text().rsplit(')',1)[1].split();e=(p/'exe').stat()
 return dict(pid=pid,start_ticks=int(s[19]),exe_dev=e.st_dev,exe_ino=e.st_ino,exe=os.readlink(p/'exe'),argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip())
assert hashlib.sha256(pathlib.Path(cfg['worker_argv'][0]).read_bytes()).hexdigest()==cfg['worker_sha256']
assert hashlib.sha256(pathlib.Path(cfg['manifest_path']).read_bytes()).hexdigest()==cfg['manifest_sha256']
state={'phase':'service','wall':time.time(),'failure':None,'no_expiry':True}
atom(D/'launch.json',{'argv':cfg['worker_argv'],'env':cfg['worker_env'],'wall':time.time(),'mono':time.monotonic()})
w=m=None
rpc='--rpc' in sys.argv[2:]
rpc_ready=False
rpc_stop=False
rpc_abort=False
rpc_outcome=None
if rpc:
 import bundle, campaign as C, run as driver
 verified=bundle.verify(R)
 if verified.get('role')!='worker' or verified!=cfg:raise RuntimeError('RPC requires the exact staged worker role')
 C.configure(R,cfg)
 binding=json.loads((R/'pins.json').read_text())['config.json']
 arm=dict(name='worker',workers=cfg['workers'],loader_threads=cfg['loader_threads'],
          caller_mask=cfg['caller_mask'],pool_masks=cfg['pool_masks'],engine_main_mask=cfg['caller_mask'])
 def answer(row):
  row=dict(row,binding=binding)
  data=json.dumps(row)+'\n'
  if len(data.encode())>65536:raise RuntimeError('guardian reply exceeds 64 KiB')
  sys.stdout.write(data);sys.stdout.flush()
 def command():
  line=sys.stdin.buffer.readline(65537)
  if not line or len(line)>65536 or not line.endswith(b'\n'):raise RuntimeError('guardian command EOF/overflow')
  return json.loads(line)
 pins=json.loads((R/'pins.json').read_text())
 answer({'helpers':{name:pins[name] for name in bundle.HELPERS},'config':cfg,
         'assets':{key:pins['assets/'+key] for key in ('worker','manifest')}})
 handshake=cfg['worker_start_seconds']+cfg['shutdown_seconds']+cfg['cleanup_seconds']
 if not select.select([sys.stdin],[],[],handshake)[0]:raise RuntimeError('guardian start handshake timeout')
 first=command()
 if first=={'op':'abort'}:
  answer({'aborted':True,'children_reaped':True,'identities':{}});raise SystemExit(0)
 if first!={'op':'start'}:raise RuntimeError('guardian start handshake refused')
 C.port_free(cfg['worker_host'],cfg['port'],cfg['cleanup_seconds'])
 rpc_started=time.monotonic()
DRAINERS=[]
def terminate_guardian(sig,frame):raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,terminate_guardian)
signal.signal(signal.SIGINT,terminate_guardian)
try:
 w=subprocess.Popen(cfg['worker_argv'],env=cfg['worker_env'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,bufsize=0)
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
   for line in iter(lambda:pipe.readline(65536),b''):
    if len(line)==65536 and not line.endswith(b'\n'):fail('bounded worker line overflow');continue
    n+=len(line)
    if n>32*1024*1024:fail('bounded worker log overflow: '+name);continue
    f.write(line)
    if ('listening on '+cfg['worker_host']+':'+str(cfg['port'])+' ').encode() in line:
     atom(D/'listening.json',{'wall':time.time(),'line':line.decode().strip(),'worker':ids['worker']})
    if any(t in line.lower() for t in [b'out of memory',b'illegal memory access']):fail('worker memory fault: '+line.decode())
 for p,n in [(w.stdout,'stdout.log'),(w.stderr,'stderr.log')]:
  thread=threading.Thread(target=drain,args=(p,n),daemon=True);DRAINERS.append(thread);thread.start()
 m=subprocess.Popen([sys.executable,'-B',str(R/'monitor.py'),str(D),str(os.getpid()),str(w.pid),str(pathlib.Path(sys.argv[1]))],stdin=subprocess.DEVNULL,stdout=(D/'monitor.log').open('ab'),stderr=subprocess.STDOUT,env=cfg['monitor_env'],start_new_session=True)
 ids['monitor']=identity(m.pid);atom(D/'identities.json',ids)
 try:
  while True:
   if w.poll() is not None:
    if rpc and rpc_ready:
     # STOP may execute before its controller can send wait-stop. Reconcile this same child;
     # never infer from EOF/timeout that STOP did not execute, and never send another STOP.
     if not select.select([sys.stdin],[],[],cfg['shutdown_seconds'])[0]:raise TimeoutError('peer exit reconciliation deadline')
     request=command()
     if request.get('op')=='abort':rpc_abort=True
     elif request.get('op')=='wait-stop':rpc_stop=True;rpc_outcome=request['control_outcome']
     else:raise RuntimeError('worker exited before guardian command')
    break
   state['heartbeat_wall']=time.time();atom(D/'state.json',state)
   if w.poll() is None and m.poll() is not None:fail('independent worker monitor exited')
   hp=D/'health.json'
   if hp.exists() and time.time()-hp.stat().st_mtime>30:fail('worker monitor stale >30s')
   if state['failure']:
    try:w.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL)
    break
   if rpc:
    if not rpc_ready:
     if select.select([sys.stdin],[],[],0)[0]:
      if command()!={'op':'abort'}:raise RuntimeError('only abort permitted during startup')
      rpc_abort=True;break
     if time.monotonic()-rpc_started>cfg['worker_start_seconds']:raise TimeoutError('peer worker startup deadline')
     if (D/'listening.json').exists() and (D/'health.json').exists():
      report=driver.masks(arm,phase='setup');C.listener_owned();rpc_ready=True
      answer({'ready':True,'identities':ids,'affinity':report})
     else:time.sleep(.05)
    elif select.select([sys.stdin],[],[],.1)[0]:
     request=command();op=request['op']
     if op=='compute':
      answer(driver.masks(arm))  # caller has proved its engine is idle READY; owned TIDs only
     elif op=='inspect':
      C.healthy();C.listener_owned()
      if request.get('compute',True):report=driver.masks(arm,phase='inspect')
      else:report={'worker_tasks':C.tasks(w.pid)}
      answer({'identities':{key:C.ident(row) for key,row in ids.items()},'affinity':report})
     elif op=='wait-stop':
      rpc_stop=True;rpc_outcome=request['control_outcome']
      atom(D/'stop.json',{'operation':'stop','control_outcome':rpc_outcome,'outcome':'pending','identities':ids})
      w.wait(timeout=cfg['shutdown_seconds']);break  # controller sent STOP once through approved data-plane source
     elif op=='abort':rpc_abort=True;break  # owned process cleanup below; no control exchange
     else:raise ValueError('unknown guardian operation')
    else:time.sleep(.05)
   else:time.sleep(1)
  if not rpc_abort:
   rc=w.wait();state.update(phase='exited',exit_rc=rc,exit_wall=time.time());atom(D/'state.json',state)
 finally:
  if w.poll() is None:
   fail('guardian cleanup')
   try:w.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
  try:m.wait(timeout=15)
  except subprocess.TimeoutExpired:m.terminate();m.wait(timeout=10)
except BaseException as error:
 state['failure']=state.get('failure') or 'guardian setup/runtime failure: '+repr(error)
 raise
finally:
 if w is not None:
  if w.poll() is None:
   try:os.killpg(w.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:w.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
  state.update(phase='exited',exit_rc=w.returncode,exit_wall=time.time())
 if m is not None and m.poll() is None:
  try:m.wait(timeout=15)
  except subprocess.TimeoutExpired:m.terminate();m.wait(timeout=10)
 if m is not None:
  state['monitor_exit_rc']=m.returncode
  if m.returncode!=0:state['failure']=state.get('failure') or 'independent monitor exited unsuccessfully'
 for thread in DRAINERS:
  thread.join(timeout=5)
  if thread.is_alive():state['failure']='output drainer did not exit'
 if (D/'monitor-failure.json').exists():state['failure']='independent monitor reported failure'
 atom(D/'state.json',state)
if rpc and (rpc_stop or rpc_abort):
 C.wait_absent({key:row for key,row in ids.items() if key!='guardian'},cfg['cleanup_seconds'])
 receipt={'outcome':'completed','aborted':rpc_abort,'control_outcome':rpc_outcome,'worker_returncode':w.returncode,
          'monitor_returncode':m.returncode,'children_reaped':True,'identities':ids}
 atom(D/('abort.json' if rpc_abort else 'stop.json'),receipt);answer(receipt)
if state.get('failure') or state.get('exit_rc')!=0:raise SystemExit(1)
