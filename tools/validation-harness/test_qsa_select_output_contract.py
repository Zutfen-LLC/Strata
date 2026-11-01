"""Execute real component/control pre-backend CLIs from an unrelated directory."""
from pathlib import Path
import subprocess,tempfile,sys
from component_output_contract import require
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);fixture=root/'fixture.bin';subprocess.run([sys.argv[1],'--synthetic','2',str(fixture)],cwd=root,check=True)
 for i,program in enumerate(sys.argv[1:]):
  output=root/('contract-'+str(i)+'.json');r=subprocess.run([program,'--contract',str(fixture),str(output)],cwd=root,capture_output=True,text=True);assert r.returncode==0,r.stderr;assert not r.stdout and not r.stderr,'contract path initialized backend or emitted unexpected startup text'
 assert (root/'contract-0.json').read_bytes()==(root/'contract-1.json').read_bytes()
 import json
 for name,size in json.loads((root/'contract-0.json').read_text())['fields'].items():(root/name).write_bytes(bytes(size))
 require(root,root/'contract-0.json')
 (root/'selected.i32le').rename(root/'y.f32le')
 try:require(root,root/'contract-0.json')
 except ValueError:pass
 else:raise AssertionError('wrong QSA selection filename accepted')
 print('actual producer/control CLI contract agrees; no backend or GPU initialization; wrong output name refuses')
