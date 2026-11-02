"""Exact native-first Q/K/V comparison for frozen half-scale fixtures."""
from pathlib import Path
import argparse,hashlib,json,math,struct,subprocess,tempfile
from native_qsa_qkv_scale_pins import PINS
from native_qsa_qkv_compare import packed_literals

SIZES={'qx.bin':2880,'q.f32le':49152,'k.f32le':2048,'v.f32le':2048,
       'direct-qx.bin':2880,'direct-q.f32le':49152,'direct-k.f32le':2048,'direct-v.f32le':2048}
def sha(data):return hashlib.sha256(data).hexdigest()
def scale_literals(data,case):
 blocks=[i for i in range(8,80) if i not in (10,34,42)]
 boundaries=[0,0,1,1,2,2,1022,1022,1023,1023,1024,1024]+[0]*57
 for slot,b in enumerate(blocks):
  m=case*69+slot+1 if case<15 else 0
  expected=(m if m<=1023 else 0) if case<15 else boundaries[slot]
  if struct.unpack_from('<H',data,b*36)[0]!=expected:raise ValueError('stored half-scale literal differs')
def direct_literal():
 data=bytearray(2880)
 for block,scale in ((8,1),(9,1023),(11,1024)):
  struct.pack_into('<H',data,block*36,scale);data[block*36+4]=7;data[block*36+8]=253
 return bytes(data)
def image(root,label,case):
 root=Path(root);report=json.loads((root/'report.json').read_text());key='actual_native_quantizer_and_mmvq' if label=='native' else 'actual_vulkan_quantizer_and_mmvq'
 if any(report.get(k) is not True for k in (key,'one_packed_input','stale_changes_output','invalid_noncommit_retry','immutable_canary_repeat','direct_packed_controls','all_case_scale_literals')) or report.get('qualified') is not False:raise ValueError(label+' report differs')
 files={}
 for name,size in SIZES.items():
  p=root/name;repeat=root/(name.replace('.bin','-repeat.bin') if name.endswith('.bin') else name.replace('.f32le','-repeat.f32le'))
  if p.is_symlink() or repeat.is_symlink():raise ValueError('output symlink refused')
  data=p.read_bytes()
  if len(data)!=size or repeat.read_bytes()!=data:raise ValueError(label+' output/repeat differs: '+name)
  files[name]=data
 packed_literals(files['qx.bin']);scale_literals(files['qx.bin'],case)
 if files['direct-qx.bin']!=direct_literal():raise ValueError('direct packed literal differs')
 for prefix in ('','direct-'):
  q=struct.unpack('<12288f',files[prefix+'q.f32le']);k=struct.unpack('<512f',files[prefix+'k.f32le']);v=struct.unpack('<512f',files[prefix+'v.f32le'])
  if any(not math.isfinite(x) for row in (q,k,v) for x in row):raise ValueError('nonfinite projection output')
  if prefix:
   for row in (q,k,v):
    if row[1]!=math.ldexp(23.,-24) or row[3]!=-math.ldexp(23.*1023.,-24) or row[4]!=-math.ldexp(23.,-38):raise ValueError('independent direct consumer literal differs')
  elif any(q[h*512]!=127. or q[h*512+256]!=-127. for h in range(24)) or q[5]!=0. or k[0]!=254. or v[0]!=-254.:raise ValueError('independent row/order literals differ')
 return files
def compare(fixture,case,validator,validator_sha,native,vulkan):
 fixture=Path(fixture).resolve(strict=True);validator=Path(validator).resolve(strict=True)
 if type(case) is not int or not 0<=case<16:raise ValueError('case index differs')
 if validator.is_symlink() or not validator.is_file() or sha(validator.read_bytes())!=validator_sha:raise ValueError('compiled fixture validator differs')
 if {p.name for p in fixture.iterdir()}!=set(PINS):raise ValueError('fixture closure differs')
 for name,pin in PINS.items():
  p=fixture/name
  if p.is_symlink() or not p.is_file() or sha(p.read_bytes())!=pin:raise ValueError('frozen fixture differs: '+name)
 with tempfile.TemporaryDirectory(prefix='qkv-scale-contract-') as tmp:
  output=Path(tmp)/'contract.json';r=subprocess.run([str(validator),'--contract',str(fixture),str(fixture/f'case-{case:02}.bin'),str(output)],cwd='/',capture_output=True,text=True,timeout=120)
  if r.returncode or r.stdout or r.stderr or json.loads(output.read_text())!={'fields':SIZES,'qualified':False}:raise ValueError('compiled fixture contract differs')
 images={label:image(root,label,case) for label,root in (('native',native),('vulkan',vulkan))}
 differences={n:sum(a!=b for a,b in zip(images['native'][n],images['vulkan'][n])) for n in SIZES}
 return {'exact':all(v==0 for v in differences.values()),'different_bytes':differences,'case':case,
         'sha256':{label:{n:sha(data) for n,data in files.items()} for label,files in images.items()},'qualified':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--case',type=int,required=True);p.add_argument('--validator',required=True);p.add_argument('--validator-sha256',required=True);p.add_argument('--native',required=True);p.add_argument('--vulkan',required=True);a=p.parse_args()
 result=compare(a.fixture,a.case,a.validator,a.validator_sha256,a.native,a.vulkan);print(json.dumps(result,indent=2));raise SystemExit(0 if result['exact'] else 1)
