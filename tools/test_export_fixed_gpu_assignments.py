import importlib.util,sys,tempfile,struct,hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'tools'))
import export_fixed_gpu_assignments as x
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d)/'profile';pairs=[(l,e) for e in range(512) for l in range(48)]
 p.write_bytes(b'STRP'+struct.pack('<5I',1,48,512,len(pairs),len(pairs))+b''.join(struct.pack('<HH',*v) for v in pairs))
 launch={'argv':['strata','--expert-profile',str(p),'--expert-cache','960','--expert-cache-device1','1378']}
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 r=x.export(launch,p,digest,pathlib.Path(d)/'out')
 assert r['cpu_bytes']==116135731200
 sets=[]
 for peer in r['peers']:
  rows=(pathlib.Path(d)/'out'/(peer['peer']+'.txt')).read_text().splitlines();sets.append(set(rows))
  assert len(rows)==peer['experts'] and peer['assigned_bytes']<=peer['payload_cap_bytes']
 assert sum(map(len,sets))==len(set.union(*sets))
 assert not set.union(*sets)&set((pathlib.Path(d)/'out/local.txt').read_text().splitlines())
 for bad in [dict(launch,argv=launch['argv']+['--expert-cache-per-layer']),dict(launch,argv=launch['argv'][:-1]+['auto'])]:
  try:x.export(bad,p,digest,pathlib.Path(d)/'bad')
  except ValueError:pass
  else:raise AssertionError('invalid placement accepted')
 try:x.export(launch,p,'0'*64,pathlib.Path(d)/'bad')
 except ValueError:pass
 else:raise AssertionError('bad hash accepted')
 print(json.dumps(r));print('FIXED_EXPORT_PASS synthetic=1 model_reads=0')
