"""Check vocabulary and embedding roles in an ordered synthetic GGUF split."""
from pathlib import Path
import json,struct,subprocess,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import native_head_vocab_role_preflight as m
with tempfile.TemporaryDirectory(prefix='head-vocab-roles-') as tmp:
 paths=[];offsets=[]
 for index in range(6):
  path=Path(tmp)/(str(index)+'.gguf');paths.append(path);text=lambda s:struct.pack('<Q',len(s))+s.encode();count=1 if index<2 else 0
  header=bytearray(struct.pack('<IIQQ',0x46554747,3,count,4 if index==0 else 3))
  if index==0:header+=text('general.architecture')+struct.pack('<I',8)+text('qwen4exp')
  for key,typ,value in [('split.count',2,6),('split.no',2,index),('split.tensors.count',5,2)]:header+=text(key)+struct.pack('<I',typ)+struct.pack('<H' if typ==2 else '<i',value)
  if count:header+=text('output.weight' if index==0 else 'token_embd.weight')+struct.pack('<IQQIQ',2,2560,2,8,0)
  header+=bytes((-len(header))%32);offsets.append(len(header))
  with path.open('wb') as f:f.write(header);f.truncate(len(header)+(2*80*34 if count else 0))
 result=m.scan(paths);assert result['supported'] and result['shape']==[2560,2] and result['scale_classes']['zero']==160
 if len(sys.argv)>1:
  packed=Path(tmp)/'packed';packed.write_bytes(bytes(2880));logits=Path(tmp)/'logits';logits.write_bytes(bytes(8))
  command=[sys.argv[1],*(str(p) for p in paths),str(packed),str(logits)]
  r=subprocess.run(command,capture_output=True,text=True);assert r.returncode==0 and json.loads(r.stdout)['exact'],r.stderr
  logits.write_bytes(b'\x01'+bytes(7));r=subprocess.run(command,capture_output=True,text=True);assert r.returncode and 'projected byte mismatch' in r.stderr
 with paths[0].open('r+b') as f:f.seek(offsets[0]);f.write(b'\x01\x00')
 assert m.scan(paths)['scale_classes']['subnormal']==1
 with paths[0].open('r+b') as f:f.seek(offsets[0]);f.write(b'\x00\x7c')
 assert not m.scan(paths)['supported']
 try:m.scan(paths[::-1])
 except ValueError:pass
 else:raise AssertionError('unordered shards accepted')
print('vocabulary role framing, hash, finite scales and split refusals pass')
