"""Read the output Q8 role and scale classes from the existing ordered GGUF shards."""
from pathlib import Path
import json,sys
from gguf_reader import GGUFFile
from native_qsa_wo_role_preflight import inspect_role

def identity(p):
 s=p.stat();return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}

def scan(paths):
 if len(paths)!=6:raise ValueError('exactly six GGUF shards are required')
 paths=[Path(p).resolve(strict=True) for p in paths]
 if len(set(paths))!=6:raise ValueError('duplicate GGUF shard path')
 before=[identity(p) for p in paths];files=[GGUFFile(p) for p in paths];total=None;roles={}
 for index,f in enumerate(files):
  count=f.metadata.get('split.count');number=f.metadata.get('split.no');n=f.metadata.get('split.tensors.count')
  if count!=6 or number!=index or not isinstance(n,int) or (total is not None and total!=n):raise ValueError('invalid or unordered GGUF split')
  total=n
  for tensor in f.tensors:
   if tensor.name not in ('output.weight','token_embd.weight'):continue
   if tensor.name in roles:raise ValueError('duplicate head role: '+tensor.name)
   if tensor.type_id!=8 or len(tensor.shape)!=2 or tensor.shape[0]!=2560 or not 0<tensor.shape[1]<=1048576:raise ValueError('unsupported Q8 head role: '+tensor.name)
   size=tensor.shape[1]*80*34;start=f.data_start+tensor.offset
   if tensor.expected_bytes()!=size or tensor.offset<0 or start+size>f.path.stat().st_size:raise ValueError('head role outside GGUF')
   roles[tensor.name]=(f,tensor,start,size)
 if files[0].metadata.get('general.architecture')!='qwen4exp' or sum(len(f.tensors) for f in files)!=total:raise ValueError('incomplete or unsupported model')
 if set(roles)!={'output.weight','token_embd.weight'} or roles['output.weight'][1].shape!=roles['token_embd.weight'][1].shape:raise ValueError('head and embedding geometry differs')
 f,t,start,size=roles['output.weight'];classes,digest=inspect_role(f.path,start,t.shape[1]*80)
 if [identity(p) for p in paths]!=before:raise ValueError('GGUF shard changed during role inspection')
 return {'role':'output.weight','type':'Q8_0','shape':t.shape,'bytes':size,'sha256':digest,'scale_classes':classes,'supported':classes['nonfinite']==0,'shards':[{'path':str(p),'stat':s} for p,s in zip(paths,before)],'qualified':False}

if __name__=='__main__':
 try:
  result=scan(sys.argv[1:]);print(json.dumps(result,indent=2));raise SystemExit(0 if result['supported'] else 1)
 except (OSError,ValueError,KeyError) as e:
  print('native head vocabulary role preflight: '+str(e),file=sys.stderr);raise SystemExit(1)
