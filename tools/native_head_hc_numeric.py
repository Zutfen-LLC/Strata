"""Bounded composed head check using existing HC source-order and primitive controls."""
import argparse,hashlib,json,struct,tempfile
from pathlib import Path
import native_hc_numeric as n
import native_hc_mix_numeric as m
PINS={'cancel':'5eeae00de7c6355601b6e828f411b43fcf74be41f9c561a9deee82c35c281a78','dense':'08a5b4e6dee7595b1afa1906271867dd57010140b7d5717c384a8f58aaa3d831','sparse':'1d29fa8480c7f6cd216afec76ab2d769dfcccecb8a6c0f2f6db533c450b6bbf2','tails':'0e0d14cc41b365504a5e64166126aed3671f264d1b6ef88fafa777aa67ebd636','zero':'3c09535b013a678dc1cd4b22bc879d155b1366668d85cde243de3c1ba61d826b'}
D=10240;P=D*320

def words(path,count):
 b=path.read_bytes();n.need(len(b)==count*4,'output shape '+str(path));v=struct.unpack('<'+str(count)+'I',b)
 for x in v:n.normal(x)
 return v

def load(case,reference,candidate):
 n.need(case in PINS,'unknown fixture case');reference=Path(reference);candidate=Path(candidate)
 raw=(reference/'fixture.bin').read_bytes();n.need(len(raw)==8+2*D*4+4*P and raw[:8]==b'HCHD001\0','fixture framing')
 n.need(hashlib.sha256(raw).hexdigest()==PINS[case] and (candidate/'fixture.bin').read_bytes()==raw,'frozen input mismatch')
 a=json.loads((reference/'report.json').read_text());b=json.loads((candidate/'report.json').read_text())
 n.need(a.get('native_launcher') is True and a.get('variant')==3 and b.get('production_owner') is True,'launcher/variant identity')
 for r in (a,b):n.need(r.get('apply') is False and r.get('injection') is False and r.get('repeat') is True and r.get('canaries') is True and r.get('qualified') is False,'launcher contract')
 result={}
 for backend,path,prefix in [('native',reference,''),('vulkan',candidate/'candidate','first-')]:
  result[backend]={k:words(path/(prefix+k+'.f32le'),z) for k,z in [('normalized',D),('rs',4),('low',320),('mixed',2560)]}
 return raw,result

def evaluate(case,reference,candidate):
 raw,outputs=load(case,reference,candidate)
 x=struct.unpack_from('<10240I',raw,8);w=struct.unpack_from('<10240I',raw,8+D*4);fixture=(1,x,w,[0]*(4*D))
 totals,weighted=n.source_inputs(fixture)
 with tempfile.TemporaryDirectory(prefix='head-hc-check-') as d:
  root=Path(d)
  for backend in ('native','vulkan'):
   v=outputs[backend];n.need(all(n.rs_allowed(a,b,backend) for a,b in zip(v['rs'],totals)),'rsqrt bound')
   expected=[n.mul(vv,v['rs'][i//2560]) for i,vv in enumerate(weighted)];n.need(tuple(expected)==v['normalized'],'normalized operation order')
   mix_fixture=root/(backend+'.bin');mix_fixture.write_bytes(b'HCMIX01\0'+struct.pack('<10240I',*v['normalized'])+raw[8+2*D*4:])
   source=Path(reference) if backend=='native' else Path(candidate)/'candidate';dest=root/backend;dest.mkdir()
   for name in (('low','mixed') if backend=='native' else m.NAMES):
    p=source/(('' if backend=='native' else 'first-')+name+'.f32le');(dest/(name+'.f32le')).write_bytes(p.read_bytes())
   m.check(mix_fixture,dest,backend)
 exact={k:outputs['native'][k]==outputs['vulkan'][k] for k in ('rs','normalized','low','mixed')}
 return {'fixture_sha256':PINS[case],'own_source_and_primitive_controls':True,'exact':exact,'accepted':all(exact.values()),'qualified':False,'full_model_parity':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('case',choices=sorted(PINS));p.add_argument('native');p.add_argument('vulkan');p.add_argument('output');a=p.parse_args()
 try:
  out=Path(a.output);n.need(not out.exists() and not out.is_symlink(),'fresh output required')
  r=evaluate(a.case,a.native,a.vulkan)
  with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
  raise SystemExit(0 if r['accepted'] else 1)
 except (OSError,ValueError,KeyError,struct.error) as e:print('STOP: '+str(e),file=__import__('sys').stderr);raise SystemExit(2)
