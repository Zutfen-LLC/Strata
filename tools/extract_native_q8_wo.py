"""Extract only the selected sm86 Q8_0 T1 consumer for manual SASS review."""
from pathlib import Path
import hashlib,json,re,subprocess,sys

def select(lines,output,demangle):
 arch='';selected=[];keep=False
 for line in lines:
  if 'code for sm_' in line:arch=line.strip();keep=False
  if 'Function :' in line:
   name=line.split('Function :',1)[1].strip();keep=False
   if 'native_small_mmvq_kernel' in name and arch=='code for sm_86':
    human=demangle(name)
    if re.search(r'native_small_mmvq_kernel<[^>]*Q80Block\s*,\s*8\s*,\s*false\s*>',human):
     keep=True;record={'name':name,'demangled':human,'architecture':arch,'instructions':0};selected.append(record);output.write(arch+'\n')
  if keep:
   record['instructions']+=bool(re.search(r'/\*[0-9a-fA-F]+\*/',line));output.write(line)
 if len(selected)!=1 or not selected[0]['instructions']:raise ValueError('expected exactly one nonempty sm86 Q8_0 Qi=8 SmallK=false kernel')
 return selected

def extract(obj,cuobjdump,cxxfilt,dest):
 obj,cuobjdump,cxxfilt=map(lambda p:Path(p).resolve(strict=True),(obj,cuobjdump,cxxfilt));dest=Path(dest).absolute()
 if dest.exists():raise ValueError('fresh inspection directory required')
 dest.mkdir(mode=0o700)
 def demangle(s):return subprocess.check_output([str(cxxfilt),s],text=True,timeout=20).strip()
 version=subprocess.check_output([str(cuobjdump),'--version'],text=True,timeout=20);(dest/'cuobjdump-version.txt').write_text(version)
 with (dest/'consumer.sass').open('x') as out,(dest/'cuobjdump.stderr').open('x') as err:
  proc=subprocess.Popen([str(cuobjdump),'--dump-sass',str(obj)],stdout=subprocess.PIPE,stderr=err,text=True)
  try:
   records=select(proc.stdout,out,demangle)
   if proc.wait(timeout=30):raise ValueError('cuobjdump failed')
  finally:
   if proc.poll() is None:proc.kill();proc.wait()
   proc.stdout.close()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 receipt={'native_object_sha256':sha(obj),'cuobjdump_sha256':sha(cuobjdump),'cxxfilt_sha256':sha(cxxfilt),'consumer_sass_sha256':sha(dest/'consumer.sass'),'functions':records,'stream_complete':True,'manual_fma_order_ftz_review_required':True,'qualified':False}
 (dest/'inspection.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
 if len(sys.argv)!=5:raise SystemExit('usage: extract_native_q8_wo.py OBJECT CUOBJDUMP CXXFILT NEW_DIR')
 print(json.dumps(extract(*sys.argv[1:]),indent=2))
