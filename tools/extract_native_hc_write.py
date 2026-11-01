"""Read-only recovery of an already captured native final-write tuple.
No numerical inverse, payload recomputation, model execution or new role access.
"""
import argparse,hashlib,json,os,stat,struct
from pathlib import Path
from compare_layer0_observer import read
from native_hc_write_numeric import parse

def extract(trace,expected_sha256,binding,output):
 trace=Path(trace);output=Path(output)
 for value in [expected_sha256,binding]:
  if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('expected SHA256/binding')
 def identity(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 st=trace.stat()
 if not stat.S_ISREG(st.st_mode) or trace.is_symlink():raise ValueError('regular unlinked capture required')
 actual,fields,digest=read(trace)
 if digest!=expected_sha256 or actual!=binding or trace.read_bytes()[:8]!=b'STRL004\0':raise ValueError('sealed STRL004 identity mismatch')
 if identity(st)!=identity(trace.stat()):raise ValueError('capture changed during read')
 # Rt after applying attention, actual combined FFN block after helper combine,
 # and actual FFN injection. All taps precede buffer reuse on the original stream.
 selected=[(3,'residual',40960),(10,'block',10240),(11,'inject',16)]
 for key,name,size in selected:
  if fields.get(key) is None or len(fields[key])!=size:raise ValueError('write boundary unavailable '+name)
 raw=b'HCWRITE1'+b''.join(fields[key] for key,_,_ in selected);parse(raw)
 if identity(st)!=identity(trace.stat()):raise ValueError('capture changed before publication')
 output.mkdir(mode=0o700)
 receipt={'trace_sha256':digest,'canonical_binding':actual,'trace_version':'STRL004','position':0,'layer':0,'token':struct.unpack('<i',fields[15])[0],'fixture_sha256':hashlib.sha256(raw).hexdigest(),'tuple_bytes':51216,'fields':{name:{'field':key,'bytes':size,'sha256':hashlib.sha256(fields[key]).hexdigest()} for key,name,size in selected},'capture_source':'existing graph taps; completed original native invocation','new_model_invocation':False,'reconstructed_values':False,'qualified':False}
 with (output/'fixture.bin').open('xb') as f:f.write(raw)
 with (output/'provenance.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
 with (output/'SHA256SUMS').open('x') as f:
  for p in sorted(output.iterdir()):
   if p.name!='SHA256SUMS':f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
 return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trace',required=True);p.add_argument('--trace-sha256',required=True);p.add_argument('--binding',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(extract(a.trace,a.trace_sha256,a.binding,a.output),indent=2))
