"""Replay preserves all 2,560 source values; native launcher is a host provider here."""
from pathlib import Path
import importlib.util,json,os,struct,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2];control=Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location('provider',root/'tests/kernels/test_native_qsa_projection_q8_native_cli.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='qkv-replay-') as tmp:
 d=Path(tmp);m.build(d,replay=True);provider=d/'provider.cpp';text=provider.read_text();assert text.count('n!=6144')==1;provider.write_text(text.replace('n!=6144','n!=2560'))
 exe=d/'reference';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-DSTRATA_PROJECTION_Q8_QKV=1','-DSTRATA_PROJECTION_Q8_REPLAY=1','-I'+str(d),'-I'+str(root/'include'),'-I'+str(root/'src/core'),'-I'+str(root/'tests/kernels'),'-x','c++',str(root/'tests/kernels/native_qsa_projection_q8_reference.cu'),str(provider),'-o',str(exe)],check=True)
 raw=d/'mixed.f32le';raw.write_bytes(b''.join(struct.pack('<f',float((i*37)%251-125)/8.0) for i in range(2560)))
 fixture=d/'fixture.bin';subprocess.run([str(control),'--wrap',str(raw),str(fixture)],cwd='/',check=True)
 assert fixture.read_bytes()==struct.pack('<III',0x31513850,2,2560)+raw.read_bytes()
 for binary in (control,exe):
  contract=d/(binary.name+'.json');r=subprocess.run([str(binary),'--contract',str(fixture),str(contract)],cwd='/',capture_output=True,text=True);assert r.returncode==0 and not r.stdout and not r.stderr,r.stderr
  assert json.loads(contract.read_text())=={'fields':{'qx.bin':2880},'qualified':False}
 trace=d/'trace';out=d/'native';r=subprocess.run([str(exe),'--device','0','--expect-device','signature-fixture GPU','--fixture',str(fixture),'--output',str(out)],cwd='/',env=dict(os.environ,Q8_TEST_TRACE=str(trace)),capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr
 assert (out/'qx.bin').read_bytes()==(out/'qx-repeat.bin').read_bytes() and len((out/'qx.bin').read_bytes())==2880
 assert json.loads((out/'report.json').read_text())['replay_fixture'] is True
 subprocess.run([str(control),'--check',str(fixture),str(out/'qx.bin')],cwd='/',check=True)
 bad=d/'short.f32le';bad.write_bytes(raw.read_bytes()[:-4]);assert subprocess.run([str(control),'--wrap',str(bad),str(d/'bad.bin')],capture_output=True).returncode
 print('actual 2560-value replay framing, native launcher contract, repeat and short-input refusal pass; host provider only')
