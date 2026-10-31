"""T1 native final materialization: exact local FMA, bounded own gate.
Uses the existing normal-domain exp/div v2 policy, never a final-output epsilon.
"""
import argparse,hashlib,json,struct
from pathlib import Path
import native_hc_numeric as n
import native_hc_mix_numeric as m
SIZE=51224

def parse(raw):
 n.need(len(raw)==SIZE and raw[:8]==b'HCWRITE1','write fixture size/version')
 v=list(struct.unpack_from('<12804I',raw,8));r,b,i=v[:10240],v[10240:12800],v[12800:]
 for u in v:n.normal(u)
 for u in i:n.need(m.in_activation_domain(n.fma(0x3e800000,u,0)),'write normal activation domain')
 return r,b,i

def gate_bounds(inject,backend):
 scaled=n.fma(0x3e800000,inject,0)
 if scaled&0x7fffffff==0:return (0x3f800000,0x3f800000)
 lo,hi=m.exp_bounds(scaled^0x80000000,backend);values=[]
 for e in (lo,hi):
  for q in m.quotient_bounds(0x3f800000,n.add(0x3f800000,e)):values.append(n.fma(0x40000000,q,0))
 return min(values,key=n.value),max(values,key=n.value)

def controls(r,b,i,gate,result,backend,exp=None):
 n.need(backend in ['native','vulkan'],'write backend')
 n.need(len(r)==len(result)==10240 and len(b)==2560 and len(i)==len(gate)==4,'write tensor spans')
 for u in r+b+i+gate+result:n.normal(u)
 if exp is not None:n.need(len(exp)==4,'write exp span')
 for c in range(4):
  n.need(m.inside(gate[c],gate_bounds(i[c],backend)),'write gate intrinsic interval')
  if exp is not None:
   scaled=n.fma(0x3e800000,i[c],0);n.need(m.inside(exp[c],m.exp_bounds(scaled^0x80000000,backend)),'write own exp interval')
   q=n.mul(gate[c],0x3f000000)
   n.need(n.fma(0x40000000,q,0)==gate[c] and m.inside(q,m.quotient_bounds(0x3f800000,n.add(0x3f800000,exp[c]))),'write own-exp gate association')
  for d in range(2560):n.need(result[c*2560+d]==n.fma(b[d],gate[c],r[c*2560+d]),'write exact local FMA')
 return True

def check(fixture,directory,backend):
 raw=Path(fixture).read_bytes();r,b,i=parse(raw);directory=Path(directory)
 n.need((directory/'fixture.bin').read_bytes()==raw,'write fixture binding')
 def words(name,count):
  v=(directory/(name+'.f32le')).read_bytes();n.need(len(v)==count*4,'write output size');return list(struct.unpack('<'+str(count)+'I',v))
 controls(r,b,i,words('gate',4),words('residual',10240),backend,words('exp',4) if backend=='vulkan' else None)
 return {'contract':'native-hc-final-write-t1-v1','backend':backend,'fixture_sha256':hashlib.sha256(raw).hexdigest(),'exact_own_gate_FMA':True,'gate_intrinsic_interval':True,'qualified':False,'model_integration':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--directory',required=True);p.add_argument('--backend',choices=['native','vulkan'],required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.directory,a.backend),indent=2))
