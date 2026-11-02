"""Test replay framing and real producer entrypoints with synthetic inputs only."""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil,struct,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent
replay=Path(sys.argv[1]).resolve(); original=Path(sys.argv[2]).resolve(); vulkan=Path(sys.argv[3]).resolve() if len(sys.argv)>3 else None
spec=importlib.util.spec_from_file_location('native_provider',ROOT/'test_native_qsa_projection_q8_native_cli.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def run(args,**kw):return subprocess.run([str(x) for x in args],capture_output=True,text=True,**kw)
with tempfile.TemporaryDirectory(prefix='q8-replay-contract-') as t:
 p=Path(t);native=m.build(p,replay=True)
 for pos in range(8):
  raw=p/f'{pos}.raw'; payload=struct.pack('<6144f',*(0.25+(i%23)*0.0625+pos*0.125 for i in range(6144)));raw.write_bytes(payload)
  f=p/f'{pos}.bin';r=run([replay,'--wrap',raw,f],cwd=p);assert r.returncode==0,r.stderr
  assert f.read_bytes()==struct.pack('<III',0x31513850,2,6144)+payload
  assert run([replay,'--validate',f],cwd=p).returncode==0
  assert run([original,'--validate',f],cwd=p).returncode!=0
  paths=[]
  for n,exe in enumerate([replay,native]+([vulkan] if vulkan else [])):
   out=p/f'{pos}-{n}.json';r=run([exe,'--contract',f,out],cwd=p);assert r.returncode==0,r.stderr;paths.append(out.read_bytes())
  assert len(set(paths))==1 and json.loads(paths[0])=={'fields':{'qx.bin':6912},'qualified':False}
  trace=p/f'{pos}.trace';out=p/f'out-{pos}'
  r=run([native,'--device','0','--expect-device','signature-fixture GPU','--fixture',f,'--output',out],cwd=p,env=dict(os.environ,Q8_TEST_TRACE=str(trace)))
  assert r.returncode==0,r.stderr
  assert (out/'fixture.bin').read_bytes()==f.read_bytes() and (out/'qx.bin').read_bytes()==(out/'qx-repeat.bin').read_bytes()
  assert json.loads((out/'report.json').read_text())['replay_fixture'] is True
  assert trace.read_text().splitlines()==['create','launch','sync','launch','sync','destroy']
 # Exercise the released checker against actual native-provider output and a
 # declared synthetic Vulkan result; this is not numerical GPU parity.
 check=ROOT.parent.parent/'tools/native_qsa_projection_q8_replay_check.py'
 vk=p/'vk';shutil.copytree(out,vk)
 (vk/'report.json').write_text(json.dumps({'actual_vulkan':True,'mode':0,'rcp_observation_candidate':True,'immutable_canary_repeat_refusal':True,'replay_fixture':True,'qualified':False}))
 rawsha=hashlib.sha256(payload).hexdigest();vsha=hashlib.sha256(replay.read_bytes()).hexdigest()
 args=[sys.executable,check,'--fixture',f,'--raw-sha256',rawsha,'--native',out,'--vulkan',vk,'--validator',replay,'--validator-sha256',vsha]
 assert run(args).returncode==0
 q=bytearray((vk/'qx.bin').read_bytes());q[-1]^=1;(vk/'qx.bin').write_bytes(q);(vk/'qx-repeat.bin').write_bytes(q)
 result=run(args);assert result.returncode==1 and json.loads(result.stdout)['different_bytes']==1
 assert run(args[:-1]+['0'*64]).returncode!=0
 (vk/'fixture.bin').write_bytes((vk/'fixture.bin').read_bytes()[:-1]);assert run(args).returncode!=0
 assert run([replay,'--wrap',raw,f]).returncode!=0 # no overwrite
 for rawdata in (payload[:-1],payload+b'\x00',struct.pack('<I',0x7f800000)+payload[4:],struct.pack('<I',1)+payload[4:]):
  bad=p/'bad.raw';bad.write_bytes(rawdata);dest=p/'bad.bin';assert run([replay,'--wrap',bad,dest]).returncode!=0;assert not dest.exists()
 old=p/'old.bin';assert run([original,'--synthetic','0',old]).returncode==0
 assert run([replay,'--validate',old]).returncode!=0
 print('eight unchanged payloads, distinct framing, validator refusals and actual producer contracts pass; CUDA provider simulated')
