"""Actual numerical entrypoint, synthetic inputs/providers, no CUDA or GPU."""
import hashlib,json,struct,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
import native_hc_numeric as n
import native_head_hc_tail_fixture as fixture_tool
import compare_native_head_hc_tail as tail

def write(p,values):p.write_bytes(struct.pack('<'+str(len(values))+'I',*values))
with tempfile.TemporaryDirectory(prefix='head-tail-compare-') as tmp:
 d=Path(tmp);fixture=d/'fixture.bin';assert fixture_tool.generate('tail',fixture)==tail.FIXTURE_SHA
 vk=d/'vk';vk.mkdir();(vk/'fixture.bin').write_bytes(fixture.read_bytes());(vk/'effective.spv').write_bytes(b'simulated-head-module');module=hashlib.sha256((vk/'effective.spv').read_bytes()).hexdigest()
 (vk/'report.json').write_text(json.dumps({'head_fixture':True,'guards':True,'repeat_exact':True,'alias_refused':True,'qualified':False}))
 pre=[0]*320;exp=[0x3f800000]*320;low=[0]*320
 for i,operand in enumerate(tail.ROWS):pre[i]=int(operand,16);exp[i]=0x7f800000;low[i]=0x80000000
 write(vk/'low-pre.f32le',pre);write(vk/'low-exp.f32le',exp);write(vk/'low.f32le',low)
 for name,values in [('up-pre',[0]*10240),('up-exp',[0x3f800000]*10240),('gates',[0x3f000000]*10240)]:write(vk/(name+'.f32le'),values)
 x=struct.unpack_from('<10240I',fixture.read_bytes(),8);mixed=[0]*2560
 for i in range(6):mixed[i]=n.mul(n.mul(x[i],0x3f000000),0x3e800000)
 write(vk/'mixed.f32le',mixed)
 native=d/'native';native.mkdir()
 for operand in (*tail.ROWS,tail.EXTRA):
  sub=native/operand;sub.mkdir();words=[0,0,0,0,int(operand,16),0x7f800000,0x7f800000,0x80000000,0,0];write(sub/'words.u32le',words)
  (sub/'report.json').write_text(json.dumps({'probe':'native_head_hc_primitive_v1','inputs':['bf5544aa',operand,'3e800022','3fffffff'],'words':[f'{w:08x}' for w in words],'repeat':True,'canaries':True,'compiled_instruction_review_required':True,'qualified':False}))
 out=d/'result.json';cmd=[sys.executable,'-I','-S','-B',str(root/'tools/isolated_head_entry.py'),str(root/'tools/compare_native_head_hc_tail.py'),str(fixture),str(vk),str(native),module,str(out)]
 p=subprocess.run(cmd,cwd='/',capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stderr
 result=json.loads(out.read_text());assert result['cross_backend_tail_exact'] and result['physical_gates_required'] and not result['qualified']
 assert subprocess.run(cmd,cwd='/',capture_output=True).returncode!=0
 words=native/tail.ROWS[0]/'words.u32le';data=bytearray(words.read_bytes());struct.pack_into('<I',data,28,0);words.write_bytes(data)
 p=subprocess.run(cmd[:-1]+[str(d/'bad.json')],cwd='/',capture_output=True);assert p.returncode!=0 and not (d/'bad.json').exists()
 p=subprocess.run(cmd[:-2]+['0'*64,str(d/'bad-module.json')],cwd='/',capture_output=True);assert p.returncode!=0 and not (d/'bad-module.json').exists()
print('Tail comparator actual entrypoint and tamper/refusal paths pass; native/Vulkan data simulated')
