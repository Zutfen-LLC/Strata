"""Strict comparison for the frozen synthetic Q/K/V component fixture."""
from pathlib import Path
import argparse,hashlib,json,math,struct,subprocess,tempfile

PINS={'input.bin':'3161efdb6f01a2a3a1fa6f90bc16c425f45e8f7a16c1cc5170bb12eaa4194c12',
      'q.bin':'cf69caa8c0addc9c7b900451b20d85c797555c58070795a69d38e111fbfc3be6',
      'k.bin':'2bc6380d6cf4a6998462defeb591db1ac5a1180e4d0455378f594ae396b00074',
      'v.bin':'fc670581fbefef9cf741bcdd241ad3c02978cf953446ab699fab171f6fced491'}
SIZES={'qx.bin':2880,'q.f32le':49152,'k.f32le':2048,'v.f32le':2048}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def packed_literals(data):
 controls=[(0,0,[]),(0,0x8000,[]),(0x3c00,0x57f0,[127,1,-1,2,-2]),
           (0x3c00,0x57f0,[127,0]),(0x7bff,0x4000,[127,0,-127,0]),
           (0x7bff,0x7bff,[127]),(0x3c00,0xd7f0,[-127,0]),(0x3c00,0x57f1,[127])]
 for block,(d,s,codes) in enumerate(controls):
  expected=struct.pack('<HH',d,s)+bytes((x&255 for x in codes))+bytes(32-len(codes))
  if data[block*36:(block+1)*36]!=expected:raise ValueError('independent packed-byte literal differs')
def compare(fixture,validator,validator_sha,native,vulkan):
 fixture=Path(fixture).resolve(strict=True);validator=Path(validator).resolve(strict=True)
 if not validator.is_file() or validator.is_symlink() or digest(validator)!=validator_sha:raise ValueError('compiled fixture validator differs')
 for name,pin in PINS.items():
  p=fixture/name
  if p.is_symlink() or not p.is_file() or digest(p)!=pin:raise ValueError('frozen fixture differs: '+name)
 with tempfile.TemporaryDirectory(prefix='qsa-qkv-contract-') as tmp:
  output=Path(tmp)/'contract.json';r=subprocess.run([str(validator),'--contract',str(fixture),str(fixture/'input.bin'),str(output)],cwd='/',capture_output=True,text=True,timeout=120)
  if r.returncode or r.stdout or r.stderr:raise ValueError('compiled fixture validation failed: '+r.stderr)
  if json.loads(output.read_text())!={'fields':SIZES,'qualified':False}:raise ValueError('fixture contract differs')
 images={};hashes={}
 for label,root in [('native',Path(native)),('vulkan',Path(vulkan))]:
  report=json.loads((root/'report.json').read_text());key='actual_native_quantizer_and_mmvq' if label=='native' else 'actual_vulkan_quantizer_and_mmvq'
  if any(report.get(k) is not True for k in (key,'one_packed_input','stale_changes_output','invalid_noncommit_retry','immutable_canary_repeat')) or report.get('qualified') is not False:raise ValueError(label+' report differs')
  files={}
  for name,size in SIZES.items():
   p=root/name;repeat=root/(name.replace('.bin','-repeat.bin') if name.endswith('.bin') else name.replace('.f32le','-repeat.f32le'))
   if p.is_symlink() or repeat.is_symlink():raise ValueError('output symlink refused')
   data=p.read_bytes()
   if len(data)!=size or repeat.read_bytes()!=data:raise ValueError(label+' output/repeat differs: '+name)
   files[name]=data
  packed_literals(files['qx.bin'])
  for name in ('q.f32le','k.f32le','v.f32le'):
   if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',files[name])):raise ValueError(label+' nonfinite output')
  q=struct.unpack('<12288f',files['q.f32le']);k=struct.unpack('<512f',files['k.f32le']);v=struct.unpack('<512f',files['v.f32le'])
  if any(q[h*512]!=127.0 or q[h*512+256]!=-127.0 for h in range(24)) or q[5]!=0.0 or k[0]!=254.0 or v[0]!=-254.0:raise ValueError(label+' independent row/order literals differ')
  images[label]=files;hashes[label]={n:hashlib.sha256(data).hexdigest() for n,data in files.items()}
 differences={n:sum(a!=b for a,b in zip(images['native'][n],images['vulkan'][n])) for n in SIZES}
 return {'exact':all(v==0 for v in differences.values()),'different_bytes':differences,'sha256':hashes,'qualified':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--validator',required=True);p.add_argument('--validator-sha256',required=True);p.add_argument('--native',required=True);p.add_argument('--vulkan',required=True);a=p.parse_args()
 result=compare(a.fixture,a.validator,a.validator_sha256,a.native,a.vulkan);print(json.dumps(result,indent=2));raise SystemExit(0 if result['exact'] else 1)
