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
 for bad in [dict(launch,argv=launch['argv']+['--expert-cache-device2','10']),dict(launch,argv=launch['argv'][:-1]+['auto'])]:
  try:x.export(bad,p,digest,pathlib.Path(d)/'bad')
  except ValueError:pass
  else:raise AssertionError('invalid placement accepted')
 try:x.export(launch,p,'0'*64,pathlib.Path(d)/'bad')
 except ValueError:pass
 else:raise AssertionError('bad hash accepted')
 print(json.dumps(r));print('FIXED_EXPORT_PASS synthetic=1 model_reads=0')

# A skewed ranking distinguishes real per-layer admission from a global prefix.
# The exact retained flag must preserve 20 primary experts in every one of 48 layers.
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d)/'profile';pairs=[(l,e) for l in range(48) for e in range(512)]
 p.write_bytes(b'STRP'+struct.pack('<5I',1,48,512,len(pairs),len(pairs))+b''.join(struct.pack('<HH',*v) for v in pairs))
 launch={'argv':['strata','--expert-profile',str(p),'--expert-cache','960',
                 '--expert-cache-device1','1378','--expert-cache-per-layer']}
 digest=hashlib.sha256(p.read_bytes()).hexdigest();out=pathlib.Path(d)/'out'
 r=x.export(launch,p,digest,out)
 def rows(name):return {tuple(map(int,line.split())) for line in (out/name).read_text().splitlines()}
 primary={(l,e) for l in range(48) for e in range(20)}
 helper=set([pair for pair in pairs if pair not in primary][:1378])
 cpu=set(pairs)-primary-helper
 assert rows('primary.txt')==primary and rows('helper.txt')==helper
 assert primary!=set(pairs[:960]),'regression must distinguish the old incorrect prefix'
 assert rows('local.txt')==primary|helper and rows('cpu.txt')==cpu and len(cpu)==22238
 assert r['primary_per_layer'] and r['primary_layer_counts']==[20]*48
 assert r['cpu_bytes']==116135731200
 bitmap=(out/'cpu.bitmap').read_bytes()
 assert {(l,e) for l in range(48) for e in range(512) if bitmap[l*64+e//8]&(1<<(e%8))}==cpu
 assigned=set()
 for peer in r['peers']:
  share=rows(peer['peer']+'.txt')
  assert len(share)==peer['experts'] and share<=cpu and not assigned&share
  assigned|=share
 assert len(assigned)==9250
 print('PER_LAYER_FIXED_EXPORT_PASS primary=960 helper=1378 per_layer=20 cpu=22238 cpu_bytes=116135731200 model_reads=0')
