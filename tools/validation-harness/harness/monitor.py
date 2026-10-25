import os,sys,json,time,signal,pathlib,subprocess,select
D=pathlib.Path(sys.argv[1]);broker=int(sys.argv[2]);engine=int(sys.argv[3])
bfd=os.pidfd_open(broker);efd=os.pidfd_open(engine)
cfg=json.loads(pathlib.Path(sys.argv[4]).read_text())
start=time.monotonic(); bad_since=None
peaks={'rss_kib':0,'hwm_kib':0,'swap_kib':0,'min_available_kib':None,'gpu_used_mib':{},'gpu_min_free_mib':{}}
def atomic(p,v):
 t=str(p)+'.monitor-tmp';pathlib.Path(t).write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def exited(fd):return bool(select.select([fd],[],[],0)[0])
def readstatus(pid):
 d={}
 for l in pathlib.Path('/proc',str(pid),'status').read_text().splitlines():
  k,_,v=l.partition(':');d[k]=v.strip()
 return d
def cgroups():
 p=pathlib.Path('/sys/fs/cgroup'+pathlib.Path('/proc',str(engine),'cgroup').read_text().strip().split('::')[1]);rows={}
 while str(p).startswith('/sys/fs/cgroup'):
  rows[str(p)]={n:(p/n).read_text().strip() for n in ['memory.max','memory.high','memory.current','memory.events','cpu.max'] if (p/n).exists()}
  if p==pathlib.Path('/sys/fs/cgroup'):break
  p=p.parent
 return rows
baseline=None
while True:
 if exited(efd):
  atomic(D/'monitor-final.json',{'wall':time.time(),'reason':'engine exited','peaks':peaks});break
 reason=None
 try:
  st=readstatus(engine)
  mem={l.split(':')[0]:int(l.split()[1]) for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if len(l.split())>=2 and l.split()[1].isdigit()}
  rss=int(st.get('VmRSS','0 kB').split()[0]);sw=int(st.get('VmSwap','0 kB').split()[0]);hwm=int(st.get('VmHWM','0 kB').split()[0]);avail=mem['MemAvailable']
  peaks['rss_kib']=max(peaks['rss_kib'],rss);peaks['hwm_kib']=max(peaks['hwm_kib'],hwm);peaks['swap_kib']=max(peaks['swap_kib'],sw)
  peaks['min_available_kib']=avail if peaks['min_available_kib'] is None else min(peaks['min_available_kib'],avail)
  gpus=[]
  if cfg.get('monitor_gpu',True):
   q=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
   if q.returncode:raise RuntimeError('nvidia-smi failure: '+q.stderr)
   for l in q.stdout.splitlines():
    u,used,free,temp,util=[s.strip() for s in l.split(',')];gpus.append({'uuid':u,'used_mib':int(used),'free_mib':int(free),'temp':int(temp),'util_pct':int(util)})
    peaks['gpu_used_mib'][u]=max(peaks['gpu_used_mib'].get(u,0),int(used));peaks['gpu_min_free_mib'][u]=min(peaks['gpu_min_free_mib'].get(u,2**60),int(free))
  worker=None
  wp=pathlib.Path(cfg['worker_root'])/'identities.json'
  if D != pathlib.Path(cfg['worker_root']) and wp.exists():
   wi=json.loads(wp.read_text())['worker'];wst=readstatus(wi['pid'])
   worker={'pid':wi['pid'],'rss_kib':int(wst.get('VmRSS','0 kB').split()[0]),'hwm_kib':int(wst.get('VmHWM','0 kB').split()[0]),'swap_kib':int(wst.get('VmSwap','0 kB').split()[0])}
  cg=cgroups()
  now_oom={p:{k:int(v) for k,v in (l.split() for l in row.get('memory.events','').splitlines()) if k in ['oom','oom_kill','oom_group_kill']} for p,row in cg.items()}
  if baseline is None:baseline=now_oom
  if any(v>baseline.get(p,{}).get(k,v) for p,d in now_oom.items() for k,v in d.items()):reason='new cgroup OOM event'
  if avail<2*1024**2:reason='host MemAvailable below 2 GiB'
  pressure=avail<4*1024**2 or sw>1024**2 or (worker is not None and worker['swap_kib']>1024**2)
  if pressure:
   if bad_since is None:bad_since=time.monotonic()
   if time.monotonic()-bad_since>30:reason='sustained low RAM or owned VmSwap >1 GiB for 30s'
  else:bad_since=None
  if exited(bfd):reason='input broker exited'
  sp=D/'state.json'
  if sp.exists() and time.time()-sp.stat().st_mtime>30:reason='input broker heartbeat stale >30s'
  tasks=[]
  for p in pathlib.Path('/proc',str(engine),'task').iterdir():
   try:tasks.append({'tid':int(p.name),'allowed':sorted(os.sched_getaffinity(int(p.name))),'comm':(p/'comm').read_text().strip()})
   except ProcessLookupError:pass
   except FileNotFoundError:pass
  row={'wall':time.time(),'mono':time.monotonic(),'engine':engine,'broker':broker,'worker':worker,'combined_rss_kib':rss+(worker['rss_kib'] if worker else 0),'rss_kib':rss,'hwm_kib':hwm,'swap_kib':sw,'available_kib':avail,'host_swap_free_kib':mem['SwapFree'],'memory_pressure':pathlib.Path('/proc/pressure/memory').read_text(),'gpus':gpus,'cgroups':cg,'tasks':tasks,'peaks':peaks,'failure':reason}
  logfile=D/'resources.jsonl'
  if logfile.exists() and logfile.stat().st_size>32*1024*1024:os.replace(logfile,D/'resources.previous.jsonl')
  with logfile.open('a') as f:f.write(json.dumps(row)+'\n')
  atomic(D/'health.json',row)
 except Exception as e:reason='monitor coverage failure: '+repr(e)
 if reason:
  atomic(D/'monitor-failure.json',{'wall':time.time(),'reason':reason,'peaks':peaks})
  try:signal.pidfd_send_signal(efd,signal.SIGTERM)
  except ProcessLookupError:pass
  if not select.select([efd],[],[],30)[0]:
   try:signal.pidfd_send_signal(efd,signal.SIGKILL)
   except ProcessLookupError:pass
  break
 time.sleep(2)
