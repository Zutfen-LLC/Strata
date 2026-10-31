#!/usr/bin/env python3
"""Exact standalone selected-native HC boundaries; never full model qualification."""
import argparse,hashlib,json,math,struct
from pathlib import Path
def need(ok,msg):
 if not ok:raise ValueError(msg)
def compare(reference,candidate):
 need((reference/'fixture.bin').stat().st_size<=1<<20 and (candidate/'fixture.bin').stat().st_size<=1<<20,'bounded fixture');fixture=(reference/'fixture.bin').read_bytes();need(fixture==(candidate/'fixture.bin').read_bytes(),'fixture bytes differ')
 need(len(fixture)>=32,'fixture header');magic,version,h,c,t,eps,typ,flags=struct.unpack_from('<8I',fixture)
 need((magic,version,h,c,eps,typ,flags)==(0x3143484e,1,2560,4,0x358637bd,30,0) and 1<=t<=8,'selected HC method fixture contract')
 need(len(fixture)==32+t*40960+40960+81920,'fixture payload length')
 native=json.loads((reference/'report.json').read_text());vk=json.loads((candidate/'report.json').read_text());need(native.get('actual_fused_gr_launcher') is True and native.get('tokens')==t,'actual native source receipt required');need(vk.get('guards') is True and vk.get('finite_normal_domain') is True and vk.get('repeat_exact') is True and vk.get('tokens')==t,'Vulkan guards/repeat receipt')
 stages=[]
 for name,count in [('rs',t*4),('normalized',t*10240),('inject',t*4)]:
  x=(reference/(name+'.f32le')).read_bytes();y=(candidate/(name+'.f32le')).read_bytes();need(len(x)==len(y)==count*4,'boundary shape '+name)
  a=struct.unpack('<'+str(count)+'f',x);b=struct.unpack('<'+str(count)+'f',y);need(all(math.isfinite(v) for v in a+b),'nonfinite '+name)
  wa=struct.unpack('<'+str(count)+'I',x);wb=struct.unpack('<'+str(count)+'I',y);different=[i for i,(u,v) in enumerate(zip(wa,wb)) if u!=v]
  stages.append({'name':name,'exact':not different,'different_words':len(different),'first_word':different[0] if different else None,'max_abs':max(abs(u-v) for u,v in zip(a,b))})
 return {'scope':'standalone HC norm/inject fixed geometry; actual native launcher; no model parity','qualified':False,'fixture_sha256':hashlib.sha256(fixture).hexdigest(),'exact_captured_boundaries':all(s['exact'] for s in stages),'first_difference':next((s['name'] for s in stages if not s['exact']),None),'stages':stages}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 try:
  r=compare(a.reference,a.candidate)
  with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
  raise SystemExit(0 if r['exact_captured_boundaries'] else 1)
 except (OSError,ValueError) as e:print('STOP:',e,file=__import__('sys').stderr);raise SystemExit(2)
