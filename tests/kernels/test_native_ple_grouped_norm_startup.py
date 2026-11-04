"""Exercise the deployed producer contract from an unrelated directory; no GPU."""
import json,struct,subprocess,sys,tempfile
from pathlib import Path
if len(sys.argv) not in (3,4):raise SystemExit('usage: COMPONENT SCRATCH')
diagnostic_bytes=int(sys.argv[3]) if len(sys.argv)==4 else 48
exe=Path(sys.argv[1]).resolve(strict=True);scratch=Path(sys.argv[2]).resolve(strict=True)
with tempfile.TemporaryDirectory(dir=scratch) as td:
 root=Path(td);fixture=root/'fixture';fixture.mkdir();result=root/'contract.json'
 for n in ('input.f32le','gamma.f32le'):(fixture/n).write_bytes(struct.pack('<f',1.)*10240)
 def call(out):return subprocess.run([str(exe),'--contract',str(fixture),str(out)],cwd='/',capture_output=True,text=True)
 p=call(result);assert p.returncode==0,p.stderr
 assert json.loads(result.read_text())=={'input_bytes':40960,'gamma_bytes':40960,'output_bytes':40960,'diagnostic_bytes':diagnostic_bytes,'qualified':False}
 assert call(result).returncode!=0
 (fixture/'input.f32le').write_bytes(bytes(40959));assert call(root/'short.json').returncode!=0
 (fixture/'input.f32le').write_bytes(struct.pack('<I',0x7f800000)+bytes(40956));assert call(root/'nonfinite.json').returncode!=0
 assert not (root/'short.json').exists() and not (root/'nonfinite.json').exists()
 print('actual grouped norm producer contract, unrelated cwd and refusal cases passed; no GPU')
