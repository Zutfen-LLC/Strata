"""Check fixed input framing and informational comparisons; no CUDA execution."""
from pathlib import Path
import json,struct,subprocess,sys,tempfile
script=Path(__file__).resolve().parents[2]/'tools/native_head_hc_primitive_report.py'
with tempfile.TemporaryDirectory(prefix='head-primitive-report-') as t:
 root=Path(t);words=[0xbf5544aa,0x4013399b,0x4053399b,0x3e9b2215,0x33c013a7,0x3f7ffffe,0x3fffffff,0x334013a8,0x3fffffdd,0x334013a8]
 (root/'words.u32le').write_bytes(struct.pack('<10I',*words));report={'probe':'native_head_hc_primitive_v1','inputs':['bf5544aa','33c013a7','3e800022','3fffffff'],'words':[f'{x:08x}' for x in words],'repeat':True,'canaries':True,'compiled_instruction_review_required':True,'qualified':False};(root/'report.json').write_text(json.dumps(report))
 p=subprocess.run([sys.executable,'-I','-S','-B',str(script),str(root)],cwd='/',text=True,capture_output=True);assert p.returncode==0,p.stderr
 value=json.loads(p.stdout);assert value['comparisons']['gate_matches_inferred_native'] and value['comparisons']['rsqrt_matches_retained_native'] and not value['qualified']
 report['words'][1]='4013399a';(root/'report.json').write_text(json.dumps(report));assert subprocess.run([sys.executable,'-I','-S','-B',str(script),str(root)],cwd='/',capture_output=True).returncode!=0
 report['words'][1]='4013399b';report['qualified']=True;(root/'report.json').write_text(json.dumps(report));assert subprocess.run([sys.executable,'-I','-S','-B',str(script),str(root)],cwd='/',capture_output=True).returncode!=0
print('fixed native primitive report framing and tamper/qualification refusals passed; no GPU')
