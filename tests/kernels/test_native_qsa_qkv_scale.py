"""Freeze and independently inspect the constructed Q/K/V scale fixtures."""
from pathlib import Path
import hashlib,importlib.util,math,struct,subprocess,sys,tempfile

binary=Path(sys.argv[1]).resolve(strict=True)
spec=importlib.util.spec_from_file_location('pins',Path(__file__).resolve().parents[2]/'tools/native_qsa_qkv_scale_pins.py');pins=importlib.util.module_from_spec(spec);spec.loader.exec_module(pins);PINS=pins.PINS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
with tempfile.TemporaryDirectory(prefix='qkv-scale-fixture-') as tmp:
 root=Path(tmp);fixture=root/'fixture'
 subprocess.run([str(binary),'--self-test'],cwd='/',check=True,timeout=120)
 subprocess.run([str(binary),'--scale-suite',str(fixture)],cwd='/',check=True,timeout=120)
 assert {p.name for p in fixture.iterdir()}=={f'case-{i:02}.bin' for i in range(16)}|{'q.bin','k.bin','v.bin'}
 assert {p.name:sha(p) for p in fixture.iterdir()}==PINS
 blocks=[i for i in range(8,80) if i not in (10,34,42)];assert len(blocks)==69
 seen=set();reciprocal=struct.unpack('<f',bytes.fromhex('0402013c'))[0]
 for case in range(16):
  path=fixture/f'case-{case:02}.bin';data=path.read_bytes();assert len(data)==10252 and struct.unpack_from('<III',data)==(0x31513850,1,2560)
  values=struct.unpack_from('<2560f',data,12)
  for slot,block in enumerate(blocks):
   expected=case*69+slot+1 if case<15 and case*69+slot+1<=1023 else 0
   if case==15:expected=([0,0,1,1,2,2,1022,1022,1023,1023,1024,1024]+[0]*57)[slot]
   a=max(abs(v) for v in values[block*32:block*32+32]);d=f32(a*reciprocal)
   actual=struct.unpack('<H',struct.pack('<e',d))[0]
   assert actual==expected,(case,slot,actual,expected)
   if case<15 and expected:seen.add(expected)
  out=root/f'contract-{case}.json';r=subprocess.run([str(binary),'--contract',str(fixture),str(path),str(out)],cwd='/',capture_output=True,text=True,timeout=120)
  assert r.returncode==0 and not r.stdout and not r.stderr,(case,r.stderr)
  assert out.is_file()
 assert seen==set(range(1,1024))
 path=fixture/'case-00.bin';data=path.read_bytes();path.write_bytes(data[:-1]);r=subprocess.run([str(binary),'--contract',str(fixture),str(path),str(root/'bad.json')],cwd='/',capture_output=True,text=True)
 assert r.returncode and not (root/'bad.json').exists()
 print('16 frozen cases cover 1,023 positive half-subnormal words, zero and RNE boundaries; fixture refusals pass')
