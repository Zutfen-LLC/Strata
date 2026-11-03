#!/usr/bin/env python3
"""Compare native scalar observations with the actual head mixer capture.

Numerical gate only: verified source/build/compiled instructions, device,
module, admission, ownership and cleanup remain independent prerequisites.
"""
import argparse,hashlib,json,re,struct
from pathlib import Path
import native_hc_mix_numeric as m

FIXTURE_SHA='949f4bae43e06cad23ad01b5a7347ff11b625a4250cb3c106412181350f8c506'
ROWS=('c2c803e5','c2b20000','c2b20001','c2c80000','c3000000','fe7fffff')
EXTRA='ff7fffff'

def file(root,name):
 p=Path(root)/name
 if p.is_symlink() or not p.is_file():raise ValueError('missing or symlinked '+name)
 return p.read_bytes()

def sha(data):return hashlib.sha256(data).hexdigest()

def check(fixture,vulkan,native,module_sha):
 fixture=Path(fixture)
 if fixture.is_symlink() or not fixture.is_file() or sha(fixture.read_bytes())!=FIXTURE_SHA:raise ValueError('tail fixture differs')
 if not re.fullmatch('[0-9a-f]{64}',module_sha) or sha(file(vulkan,'effective.spv'))!=module_sha:raise ValueError('head module differs')
 raw=file(vulkan,'fixture.bin')
 if raw!=fixture.read_bytes():raise ValueError('Vulkan input echo differs')
 report=json.loads(file(vulkan,'report.json'))
 if any(report.get(k) is not True for k in ('head_fixture','guards','repeat_exact','alias_refused')) or report.get('qualified') is not False:raise ValueError('Vulkan runner report differs')
 native_results={}
 for operand in (*ROWS,EXTRA):
  path=Path(native)/operand;data=json.loads(file(path,'report.json'));raw_words=file(path,'words.u32le')
  if len(raw_words)!=40:raise ValueError('native word span differs')
  words=[format(v,'08x') for v in struct.unpack('<10I',raw_words)]
  if (data.get('probe')!='native_head_hc_primitive_v1' or data.get('inputs')!=['bf5544aa',operand,'3e800022','3fffffff'] or data.get('words')!=words or data.get('repeat') is not True or data.get('canaries') is not True or data.get('compiled_instruction_review_required') is not True or data.get('qualified') is not False or words[4:8]!=[operand,'7f800000','7f800000','80000000']):raise ValueError('native tail observation differs: '+operand)
  native_results[operand]={'input':words[4],'exp':words[5],'denominator':words[6],'low':words[7]}
 def words(name):
  data=file(vulkan,name+'.f32le')
  if len(data)!=1280:raise ValueError('Vulkan '+name+' span differs')
  return struct.unpack('<320I',data)
 pre,exp,low=map(words,('low-pre','low-exp','low'))
 for row,operand in enumerate(ROWS):
  if (pre[row],exp[row],low[row])!=(int(operand,16),0x7f800000,0x80000000):raise ValueError('Vulkan head tail differs: '+operand)
 m.check(fixture,vulkan,'vulkan',head_silu_overflow=True)
 return {'contract':'native-head-hc-silu-overflow-v2','fixture_sha256':FIXTURE_SHA,'module_sha256':module_sha,'native':native_results,'vulkan_rows':list(ROWS),'native_extreme_only':EXTRA,'cross_backend_tail_exact':True,'numerical_acceptance':True,'physical_gates_required':True,'qualified':False,'full_model_parity':False}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture');p.add_argument('vulkan');p.add_argument('native');p.add_argument('module_sha');p.add_argument('output');a=p.parse_args()
 try:
  out=Path(a.output)
  if out.exists() or out.is_symlink():raise ValueError('fresh output required')
  result=check(a.fixture,a.vulkan,a.native,a.module_sha)
  with out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
  print(json.dumps(result,sort_keys=True))
 except (OSError,ValueError,KeyError,struct.error) as e:print('STOP: '+str(e),file=__import__('sys').stderr);raise SystemExit(1)
