"""Standalone T1 pending-read contract; exact local FMA, own-input v1/v2.
Cross-backend equality is diagnostic. No graph-state or full-model acceptance.
"""
import argparse,hashlib,json,struct,tempfile
from pathlib import Path
import native_hc_numeric as n
import native_hc_mix_numeric as m
SIZE=8+40960+40960+81920+10240+16+2*6553600

def parse(raw):
 n.need(len(raw)==SIZE and raw[:8]==b'HCPEND1\0','pending fixture size/version')
 at=8
 def take(count,code='I'):
  nonlocal at
  v=list(struct.unpack_from('<'+str(count)+code,raw,at));at+=count*(4 if code=='I' else 2);return v
 x=take(10240);w=take(10240);wi=[u<<16 for u in take(40960,'H')];bo=take(2560);inj=take(4);down=take(3276800,'H');up=take(3276800,'H')
 for u in x+w+wi+bo+inj:n.normal(u)
 for v in down+up:n.normal(v<<16)
 for v in inj:n.need(m.in_activation_domain(n.mul(v,0x3e800000)),'pending sigmoid domain')
 return x,w,wi,bo,inj,down,up

def pending_checks(x,bo,inj,gates,exp,residual,backend):
 n.need(len(x)==len(residual)==10240 and len(bo)==2560 and len(inj)==len(gates)==len(exp)==4,'pending output shape')
 for u in x+bo+inj+gates+exp+residual:n.normal(u)
 for c in range(4):
  scaled=n.mul(inj[c],0x3e800000)
  n.need(m.inside(exp[c],m.exp_bounds(scaled^0x80000000,backend)),'pending exp intrinsic bound')
  sigma=n.mul(gates[c],0x3f000000)
  n.need(n.mul(sigma,0x40000000)==gates[c] and m.inside(sigma,m.quotient_bounds(0x3f800000,n.add(0x3f800000,exp[c]))),'pending own-exp gate association')
  for d in range(2560):n.need(residual[c*2560+d]==n.fma(bo[d],gates[c],x[c*2560+d]),'pending exact FMA-before-normalization')
 return True

def check(fixture,directory,backend):
 fixture=Path(fixture);directory=Path(directory);raw=fixture.read_bytes();x,w,wi,bo,inj,down,up=parse(raw)
 n.need((directory/'fixture.bin').read_bytes()==raw,'pending fixture binding')
 def words(name,count):
  b=(directory/(name+'.f32le')).read_bytes();n.need(len(b)==count*4,'pending output size '+name);v=list(struct.unpack('<'+str(count)+'I',b))
  for u in v:n.normal(u)
  return v
 rp=words('residual',10240);gate=words('pending-gate',4);exp=words('pending-exp',4)
 pending_checks(x,bo,inj,gate,exp,rp,backend)
 norm=(1,rp,w,wi);totals,weighted=n.source_inputs(norm);rs=words('rs',4)
 n.need(all(n.rs_allowed(v,t,backend) for v,t in zip(rs,totals)),'pending own-residual v1 RS')
 xn,inject=n.downstream(norm,weighted,rs)
 n.need(words('normalized',10240)==xn and words('inject',4)==inject,'pending exact own-RS normalized/injection')
 # Existing v2 checker gets each backend's ACTUAL normalized operands, never
 # the other backend's or a host reconstruction substituted for actual values.
 with tempfile.TemporaryDirectory(prefix='hc-pending-own-mix-') as tmp:
  path=Path(tmp)/'mix.bin'
  path.write_bytes(b'HCMIX01\0'+(directory/'normalized.f32le').read_bytes()+struct.pack('<3276800H',*down)+struct.pack('<3276800H',*up))
  mix=m.check(path,directory,backend)
 return {'contract':'native-hc-pending-t1-v1','backend':backend,'fixture_sha256':hashlib.sha256(raw).hexdigest(),'pending_exact_local_FMA':True,'own_input_normal_v1':True,'own_input_mix_v2':mix,'qualified':False,'full_hc_parity':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--directory',required=True);p.add_argument('--backend',choices=['native','vulkan'],required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.directory,a.backend),indent=2))
