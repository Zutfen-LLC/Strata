"""Compare one independently pinned replay pair. Lifecycle gates remain external."""
from pathlib import Path
import argparse,hashlib,json,subprocess

def sha(data):return hashlib.sha256(data).hexdigest()
def file(path):
 path=Path(path)
 if path.is_symlink() or not path.is_file():raise ValueError('regular file required: '+str(path))
 return path.read_bytes()
def compare(fixture,raw_sha,native,vulkan,validator,validator_sha):
 if len(raw_sha)!=64 or any(c not in '0123456789abcdef' for c in raw_sha):raise ValueError('invalid raw digest')
 fixture=Path(fixture);image=file(fixture)
 if len(image)!=24588 or image[:12]!=bytes.fromhex('503851310200000000180000') or sha(image[12:])!=raw_sha:raise ValueError('replay fixture framing or raw input digest differs')
 if sha(file(validator))!=validator_sha:raise ValueError('compiled validator identity differs')
 result=subprocess.run([str(validator),'--validate',str(fixture)],capture_output=True,timeout=20)
 if result.returncode or result.stdout or result.stderr:raise ValueError('compiled replay validator refused fixture')
 rows=[]
 for kind,d in (('native',Path(native)),('vulkan',Path(vulkan))):
  if d.is_symlink() or not d.is_dir():raise ValueError('producer output directory differs')
  if file(d/'fixture.bin')!=image:raise ValueError('producer input echo differs')
  q=file(d/'qx.bin');repeat=file(d/'qx-repeat.bin')
  if len(q)!=6912 or repeat!=q:raise ValueError('packed size or repeat differs')
  report=json.loads(file(d/'report.json'))
  if report.get('qualified') is not False or report.get('replay_fixture') is not True:raise ValueError('replay report differs')
  if kind=='native' and (report.get('actual_unchanged_native_launcher') is not True or report.get('immutable_canary_repeat') is not True):raise ValueError('native producer report differs')
  if kind=='vulkan' and (report.get('actual_vulkan') is not True or report.get('mode')!=0 or report.get('rcp_observation_candidate') is not True or 'rn_candidate' in report or report.get('immutable_canary_repeat_refusal') is not True):raise ValueError('Vulkan producer report differs')
  rows.append(q)
 differences=sum(a!=b for a,b in zip(*rows))
 return {'input_sha256':raw_sha,'bytes':6912,'native_packed_sha256':sha(rows[0]),'vulkan_packed_sha256':sha(rows[1]),'different_bytes':differences,'exact':differences==0,'qualified':False,'model_integration':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--raw-sha256',required=True);p.add_argument('--native',required=True);p.add_argument('--vulkan',required=True);p.add_argument('--validator',required=True);p.add_argument('--validator-sha256',required=True);a=p.parse_args();r=compare(a.fixture,a.raw_sha256,a.native,a.vulkan,a.validator,a.validator_sha256);print(json.dumps(r,indent=2));raise SystemExit(0 if r['exact'] else 1)
