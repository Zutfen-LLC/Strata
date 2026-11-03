"""Exercise synthetic actual-head mixer inputs; no device execution."""
import hashlib,struct,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
import native_hc_mix_numeric as m
import native_hc_numeric as n
component=Path(sys.argv[1]).resolve(strict=True)
p=subprocess.run([str(component),'--contract'],cwd='/',capture_output=True,text=True);assert p.returncode==0 and p.stdout.strip()=='native_hc_mix_component_v1 fixture_or_head_fixture_or_sigmoid'
PINS={'tail':'949f4bae43e06cad23ad01b5a7347ff11b625a4250cb3c106412181350f8c506','transition':'ccf2817a6afade3973d659309b63a20120f668a66f42e93745993c97a8543f5e','positive':'8637c6928dcda672a1df54432161eb2f5748ed83679928891e415c0b188c7d9e','near':'6e04d88e91364e31aeeb543af36065c707720856d4958535ddf058c16f952dc0','neg88':'180bf2df22c8d43055027302dbc533dcef72b2c5d9c069c90810d5b3162a017f','neg8872':'ec841e85c6235b816ba60c6e2b35f4af58ec73ccebb072fea3a8b1fae7abbc14','beyond':'2fa29bf80239a7f3ca63401a03c0cb605304677b1f5a988945ba5938fa353142'}
EXPECTED={'tail':[0xc2c803e5,0xc2b20000,0xc2b20001,0xc2c80000,0xc3000000,0xfe7fffff],
          'transition':[0xc2b1ffff,0xc2b00000,0xc2b170a4,0xc2aeac4d], 'positive':[0x42c80000], 'near':[0xc2b1ffff], 'neg88':[0xc2b00000], 'neg8872':[0xc2b170a4], 'beyond':[0xc2aeac4d]}
with tempfile.TemporaryDirectory(prefix='head-tail-fixture-') as temp:
 for kind,pin in PINS.items():
  path=Path(temp)/kind
  cmd=[sys.executable,'-I','-S','-B',str(root/'tools/native_head_hc_tail_fixture.py'),kind,str(path)]
  p=subprocess.run(cmd,cwd='/',capture_output=True,text=True);assert p.returncode==0 and p.stdout.strip()==pin,p.stderr
  assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
  x,down,up=m.read_fixture(path)
  assert [n.mul(m.dot(x,down,row,10240),0x3e800000) for row in range(len(EXPECTED[kind]))]==EXPECTED[kind]
  assert not any(up)
  assert subprocess.run(cmd,cwd='/',capture_output=True).returncode!=0
 bad=Path(temp)/'nonfinite';data=bytearray(13148168);data[:8]=b'HCMIX01\0';struct.pack_into('<I',data,8,0x7f800000);bad.write_bytes(data)
 out=Path(temp)/'forbidden';p=subprocess.run([str(component),'--device','0','--expect-device','unavailable','--head-fixture',str(bad),'--output',str(out)],cwd='/',capture_output=True,text=True);assert p.returncode!=0 and not out.exists()
 assert subprocess.run([sys.executable,'-I','-S','-B',str(root/'tools/native_head_hc_tail_fixture.py'),'bad',str(Path(temp)/'bad')],cwd='/',capture_output=True).returncode!=0
print('Frozen synthetic inputs reproduce captured/cutoff/extreme preactivations and refuse overwrite; no GPU')
