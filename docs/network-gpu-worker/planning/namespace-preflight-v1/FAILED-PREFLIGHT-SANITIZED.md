# Namespace preflight — sanitized failed source (behavior preserved)

Origin: controller-generated admission launcher from the blocked a7f54f3
no-model validation run (evidence root SHA-256 of final REPORT.md:
f20ce2ed7218de78303ffea491705277ce58d9a1ccb0a6e3146611a79d3818d3).
Sanitization: ROLE_* identifiers and generic paths only; sanitized lines are
marked `# SANITIZED`. Behavior (operations attempted, order, failure mode) is
preserved exactly. This file documents the failure; it is not the corrected
preflight (see CORRECTION-SPEC.md).

```python
import os,json,pathlib,ctypes,fcntl,sys,time
p=pathlib.Path('/proc'); out={'pid':os.getpid(),'uid':os.getuid(),'host':os.uname().nodename,'utc':time.time(),'mono':time.monotonic(),'python':sys.version,'python_exe':os.readlink('/proc/self/exe')}
def read(pq):
 try:return pathlib.Path(pq).read_text()
 except Exception as e:return repr(e)
# --- failing statement (behavior preserved; terminated the run) ---
out['pid1_comm']=read('/proc/1/comm');out['pid1_exe']=os.readlink('/proc/1/exe');out['pid1_status']=read('/proc/1/status');out['self_status']=read('/proc/self/status');out['mountinfo']=read('/proc/self/mountinfo');out['cgroup']=read('/proc/self/cgroup');out['meminfo']=read('/proc/meminfo');out['vmstat']=read('/proc/vmstat');out['boot_id']=read('/proc/sys/kernel/random/boot_id')
# SANITIZED: original single line read PID 1 of the remote host; shown as /proc/1.
out['ns']={}
for n in ('pid','mnt','cgroup','user'):
 fd=os.open('/proc/self/ns/'+n,os.O_RDONLY);s=os.fstat(fd);v={'dev':s.st_dev,'ino':s.st_ino,'link':os.readlink('/proc/self/ns/'+n)}
 for label,op in [('parent',0xb702),('owner_user',0xb701)]:
  try:
   x=fcntl.ioctl(fd,op);t=os.fstat(x);v[label]={'dev':t.st_dev,'ino':t.st_ino};os.close(x)
  except OSError as e:v[label]={'errno':e.errno,'error':str(e)}
 os.close(fd);out['ns'][n]=v
out['pid1_ns']={n:{'dev':os.stat('/proc/1/ns/'+n).st_dev,'ino':os.stat('/proc/1/ns/'+n).st_ino} for n in ('pid','mnt','cgroup','user')}   # SANITIZED: PID 1 of remote host
fd=os.pidfd_open(os.getpid());os.close(fd);out['pidfd']=True
libc=ctypes.CDLL(None,use_errno=True);value=ctypes.c_int();out['prctl_get_subreaper']={'rc':libc.prctl(37,ctypes.byref(value),0,0,0),'value':value.value,'errno':ctypes.get_errno()}
out['cgroup_root']={n:read('/sys/fs/cgroup/'+n) for n in ('cgroup.controllers','cgroup.type','memory.current','memory.max','memory.events','memory.swap.current','memory.swap.max','memory.swap.events')}
out['conflicts']=[]
for d in p.iterdir():
 if d.name.isdigit():
  try:
   if d.stat().st_uid==os.getuid():
    cmd=(d/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
    if any(x in cmd for x in ('live_probe.py','test_integration.py','HARMLESS_CHILD_STARTED_V1')) and int(d.name)!=os.getpid():out['conflicts'].append({'pid':int(d.name),'cmd':cmd})
  except (OSError,PermissionError):pass
print(json.dumps(out))
```

The pidfd line above is the original verbatim (nothing private in it). It is
unreachable in the failed run: the statement at the failing line raised
PermissionError (errno 13) before any later line executed. Do not execute this
file — use CORRECTION-SPEC.md instead.

Observed failure (traceback preserved verbatim, paths generic):

```
Traceback (most recent call last):
  File "<string>", line 6, in <module>
    out['pid1_comm']=read('/proc/1/comm');out['pid1_exe']=os.readlink('/proc/1/exe');...
                                                          ~~~~~~~~~~~^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: '/proc/1/exe'
```

Return code 1; JSON snapshot never emitted; no later stage invoked.
