"""Exercise Q/K/V role shapes and finite scale classes in six GGUF shards."""
from pathlib import Path
import importlib.util,struct,sys,tempfile
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
spec=importlib.util.spec_from_file_location('qkv_roles',root/'tools/native_qsa_qkv_role_preflight.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert len(m.NAMES)==36
names={f'blk.3.{name}':rows for name,rows in m.ROWS.items()};m.NAMES=names
with tempfile.TemporaryDirectory(prefix='qkv-role-test-') as tmp:
 paths=[];starts=[];items=list(names.items())
 for index in range(6):
  path=Path(tmp)/(str(index)+'.gguf');paths.append(path);text=lambda s:struct.pack('<Q',len(s))+s.encode()
  count=1 if index<3 else 0;header=bytearray(struct.pack('<IIQQ',0x46554747,3,count,4 if index==0 else 3))
  if index==0:header+=text('general.architecture')+struct.pack('<I',8)+text('qwen4exp')
  for key,typ,value in [('split.count',2,6),('split.no',2,index),('split.tensors.count',5,3)]:header+=text(key)+struct.pack('<I',typ)+struct.pack('<H' if typ==2 else '<i',value)
  if count:
   name,rows=items[index];header+=text(name)+struct.pack('<IQQIQ',2,2560,rows,8,0)
  header+=bytes((-len(header))%32);starts.append(len(header))
  with path.open('wb') as f:f.write(header);f.truncate(len(header)+(items[index][1]*80*34 if count else 0))
 report=m.scan(paths);assert report['supported'] and len(report['roles'])==3
 assert all(row['scale_classes']['zero']==names[name]*80 for name,row in report['roles'].items())
 with paths[1].open('r+b') as f:f.seek(starts[1]);f.write(b'\x01\x00')
 report=m.scan(paths);assert report['supported'] and report['roles'][items[1][0]]['scale_classes']['subnormal']==1
 with paths[1].open('r+b') as f:f.seek(starts[1]);f.write(b'\x00\x7c')
 assert not m.scan(paths)['supported']
 try:m.scan(paths[::-1])
 except ValueError:pass
 else:raise AssertionError('unordered shards accepted')
print('QKV role framing, byte hashes, finite scale classes and malformed split refusal passed')
