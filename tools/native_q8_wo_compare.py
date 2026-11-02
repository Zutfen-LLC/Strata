"""Exact native/Vulkan WO output gate; lifecycle and module gates are external."""
from pathlib import Path
import argparse,hashlib,json,subprocess

def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):
 p=Path(path)
 if p.is_symlink() or not p.is_file():raise ValueError('regular file required')
 return p.read_bytes()
def compare(weights,weight_sha,packed,packed_sha,rows,native,vulkan,control,control_sha):
 if not 8<=rows<=2560:raise ValueError('unsupported rows')
 for p,h in ((weights,weight_sha),(packed,packed_sha),(control,control_sha)):
  if len(h)!=64 or digest(read(p))!=h:raise ValueError('bound input or control differs')
 result=subprocess.run([str(control),'--validate',str(weights),str(packed),str(rows)],capture_output=True,timeout=60)
 if result.returncode or result.stdout or result.stderr:raise ValueError('compiled domain validator refused')
 arrays=[]
 for kind,d in (('native',Path(native)),('vulkan',Path(vulkan))):
  report=json.loads(read(d/'report.json'));q=read(d/'out.f32le')
  if report.get('qualified') is not False or len(q)!=rows*4 or read(d/'out-repeat.f32le')!=q:raise ValueError('output size, repeat or scope differs')
  if kind=='native' and (report.get('actual_native_mmvq_type8') is not True or report.get('immutable_canary_repeat') is not True):raise ValueError('native report differs')
  if kind=='vulkan' and (report.get('actual_vulkan_wo_candidate') is not True or report.get('immutable_canary_retry') is not True):raise ValueError('Vulkan report differs')
  arrays.append(q)
 diff=sum(a!=b for a,b in zip(*arrays))
 return {'rows':rows,'bytes':rows*4,'native_sha256':digest(arrays[0]),'vulkan_sha256':digest(arrays[1]),'different_bytes':diff,'exact':diff==0,'qualified':False,'model_integration':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--weight-sha',required=True);p.add_argument('--packed',required=True);p.add_argument('--packed-sha',required=True);p.add_argument('--rows',type=int,required=True);p.add_argument('--native',required=True);p.add_argument('--vulkan',required=True);p.add_argument('--control',required=True);p.add_argument('--control-sha',required=True);a=p.parse_args();result=compare(a.weights,a.weight_sha,a.packed,a.packed_sha,a.rows,a.native,a.vulkan,a.control,a.control_sha);print(json.dumps(result,indent=2));raise SystemExit(0 if result['exact'] else 1)
