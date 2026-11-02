"""Strict comparison for the frozen GDN input projection component fixture."""
from pathlib import Path
import argparse,hashlib,json,math,struct,subprocess,tempfile

PINS={'input.bin':'aa76e6d501629f015285c703940589d5b23d91b460fb7ea7862c60067d9775b3',
      'dense.bin':'e762afab43438c6686a8471812281cebefc2872cf0918ed681c66759962ad073',
      'qkv.bin':'bf42724ae50e9a01a8e17d455dee88d2dae0add9c9422e9c3d5ed82eaa9417b1',
      'z.bin':'6fb19e7aa558c145b6a23b294a5bc3a50e4089c81d251b262e07ea7f8d1c3ec0'}
SIZES={'qx.bin':2880,'qkv.f32le':40960,'z.f32le':24576}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def packed_literals(data,case):
 if case=="dense.bin":
  expected=struct.pack("<HH",0x3c00,0)+bytes([127,129]+[(255 if j%2 else 1) for j in range(2,32)])
  if any(data[i*36:(i+1)*36]!=expected for i in range(80)):raise ValueError("independent dense packed-byte literal differs")
  return
 controls=[(0,0,[]),(0,0x8000,[]),(0x3c00,0x57f0,[127,1,-1,2,-2]),
           (0x3c00,0x57f0,[127,0]),(0x7bff,0x4000,[127,0,-127,0]),
           (0x7bff,0x7bff,[127]),(0x3c00,0xd7f0,[-127,0]),(0x3c00,0x57f1,[127])]
 for block,(d,s,codes) in enumerate(controls):
  expected=struct.pack('<HH',d,s)+bytes((x&255 for x in codes))+bytes(32-len(codes))
  if data[block*36:(block+1)*36]!=expected:raise ValueError('independent packed-byte literal differs')
def compare(fixture,validator,validator_sha,native,vulkan,case):
 if case not in ("input.bin","dense.bin"):raise ValueError("unsupported fixture case")
 fixture=Path(fixture).resolve(strict=True);validator=Path(validator)
 if validator.is_symlink():raise ValueError('compiled fixture validator is a symlink')
 validator=validator.resolve(strict=True)
 if not validator.is_file() or digest(validator)!=validator_sha:raise ValueError('compiled fixture validator differs')
 if any(p.is_symlink() or not p.is_file() for p in fixture.iterdir()) or {p.name for p in fixture.iterdir()}!=set(PINS):raise ValueError('fixture closure differs')
 for name,pin in PINS.items():
  p=fixture/name
  if p.is_symlink() or not p.is_file() or digest(p)!=pin:raise ValueError('frozen fixture differs: '+name)
 with tempfile.TemporaryDirectory(prefix='qsa-qkv-contract-') as tmp:
  output=Path(tmp)/'contract.json';r=subprocess.run([str(validator),'--contract',str(fixture),str(fixture/case),str(output)],cwd='/',capture_output=True,text=True,timeout=120)
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
  packed_literals(files['qx.bin'],case)
  for name in ('qkv.f32le','z.f32le'):
   if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',files[name])):raise ValueError(label+' nonfinite output')
  q=struct.unpack('<10240f',files['qkv.f32le']);z=struct.unpack('<6144f',files['z.f32le'])
  if q[0]!=127.0 or q[1]!=-127.0 or q[5]!=0.0 or z[0]!=254.0 or z[1]!=-127.0:raise ValueError(label+' independent row/order literals differ')
  images[label]=files;hashes[label]={n:hashlib.sha256(data).hexdigest() for n,data in files.items()}
 differences={n:sum(a!=b for a,b in zip(images['native'][n],images['vulkan'][n])) for n in SIZES}
 return {'exact':all(v==0 for v in differences.values()),'different_bytes':differences,'sha256':hashes,'qualified':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--case',choices=['input.bin','dense.bin'],required=True);p.add_argument('--validator',required=True);p.add_argument('--validator-sha256',required=True);p.add_argument('--native',required=True);p.add_argument('--vulkan',required=True);a=p.parse_args()
 result=compare(a.fixture,a.validator,a.validator_sha256,a.native,a.vulkan,a.case);print(json.dumps(result,indent=2));raise SystemExit(0 if result['exact'] else 1)
