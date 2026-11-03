"""Exercise the new head contract without changing the historical exact gate."""
import json,struct,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
import native_hc_numeric as n
tool=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='head-numeric-') as tmp:
 d=Path(tmp);native=d/'native';vk=d/'vk';v=vk/'candidate';native.mkdir();v.mkdir(parents=True)
 fixture=d/'zero.bin';subprocess.run([str(tool),'--make','zero',str(fixture)],cwd='/',check=True)
 raw=fixture.read_bytes();(native/'fixture.bin').write_bytes(raw);(vk/'fixture.bin').write_bytes(raw)
 common={'apply':False,'injection':False,'repeat':True,'canaries':True,'qualified':False}
 (native/'report.json').write_text(json.dumps({**common,'native_launcher':True,'variant':3}));(vk/'report.json').write_text(json.dumps({**common,'production_owner':True}))
 rs=n.rsqrt_nearest(n.EPS)
 for target,prefix in [(native,''),(v,'first-')]:
  for name,words in [('rs',[rs]*4),('normalized',[0]*10240),('low',[0]*320),('mixed',[0]*2560)]:
   (target/(prefix+name+'.f32le')).write_bytes(struct.pack('<'+str(len(words))+'I',*words))
 for name,words in [('low-pre',[0]*320),('low-exp',[0x3f800000]*320),('up-pre',[0]*10240),('gates',[0x3f000000]*10240),('up-exp',[0x3f800000]*10240)]:
  (v/('first-'+name+'.f32le')).write_bytes(struct.pack('<'+str(len(words))+'I',*words))
 cmd=[sys.executable,'-I','-S','-B',str(root/'tools/isolated_head_entry.py'),str(root/'tools/compare_native_head_hc_contract.py'),'zero',str(native),str(vk),str(d/'check.json')]
 result=subprocess.run(cmd,cwd='/',capture_output=True,text=True,timeout=300);assert result.returncode==0,result.stderr
 data=json.loads((d/'check.json').read_text());assert data['numerical_acceptance'] and data['vulkan_rne_silu']['exact'] and data['native_gate_scope']=='not_exported_existing_propagation_bound' and data['native_dot_scope']=='not_exported_reference_dots_and_activation_bounds' and data['native_sum_scope']=='not_exported_reference_sum_and_rsqrt_bound' and data['vulkan_captured_dots_exact'] and not data['qualified']
 historical=subprocess.run(cmd[:-1]+[str(d/'history.json'),'--retained-observation'],cwd='/',capture_output=True,text=True,timeout=300);assert historical.returncode==0,historical.stderr
 history=json.loads((d/'history.json').read_text());assert not history['numerical_acceptance'] and history['original_exact_gate']=='separate_not_reclassified'
 assert subprocess.run(cmd,cwd='/',capture_output=True,timeout=300).returncode!=0
 mixed=v/'first-mixed.f32le';mixed.write_bytes(struct.pack('<I',0x3f800000)+bytes(10236))
 assert subprocess.run(cmd[:-1]+[str(d/'wrong-output.json')],cwd='/',capture_output=True,timeout=300).returncode!=0 and not (d/'wrong-output.json').exists()
print('New per-backend head contract and distinct historical mode passed independently formed zero captures; actual native gates remain unexported')
