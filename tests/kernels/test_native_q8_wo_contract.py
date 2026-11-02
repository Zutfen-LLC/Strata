"""Actual WO producer contracts and native CLI with a declared host CUDA provider."""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2];control=Path(sys.argv[1]).resolve();vulkan=Path(sys.argv[2]).resolve()
spec=importlib.util.spec_from_file_location('cuda_provider',root/'tests/kernels/test_native_qsa_projection_q8_native_cli.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def run(args,**kw):return subprocess.run([str(a) for a in args],capture_output=True,text=True,timeout=120,**kw)
with tempfile.TemporaryDirectory(prefix='q8-wo-contract-') as t:
 d=Path(t);m.build(d);provider=d/'provider.cpp';prefix=provider.read_text().split('namespace strata::kernels {')[0]
 prefix=prefix.replace('#include "native_qsa_projection_q8_fixture.hpp"','#include "native_q8_wo_fixture.hpp"')
 provider.write_text(prefix+r'''
namespace strata::kernels {
void native_mmvq(int type,const void* weights,const void* x,float* y,int n,int rows,int cols,void* stream){
 validate_shape(n,cols,32);validate_pointer(weights);validate_pointer(x);validate_pointer(y);validate_stream(stream);
 if(type!=8||n!=6144||cols!=1||stream!=active||rows<8)throw std::runtime_error("WO provider: signature");
 log("launch");using namespace native_q8_wo_fixture;
 native_q8_wo_fixture::Fixture f{size_t(rows),Bytes(size_t(rows)*q8_wo_row_bytes),Bytes(q8_wo_input_bytes)};
 std::memcpy(f.weights.data(),weights,f.weights.size());std::memcpy(f.input.data(),x,f.input.size());
 for(int row=0;row<rows;++row)y[row]=reference(f,size_t(row)).value;
}}
''')
 native=d/'wo-native-provider';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(d),'-I'+str(root/'include'),'-I'+str(root/'src/core'),'-I'+str(root/'tests/kernels'),'-x','c++',str(root/'tests/kernels/native_q8_wo_reference.cu'),str(provider),'-o',str(native)],check=True)
 for label,rows in (('basic-8',8),('basic-full',2560),('subnormal',2560)):
  fixture=d/f'fixture-{label}';r=run([control,'--subnormal',fixture] if label=='subnormal' else [control,'--synthetic',rows,fixture],cwd=d);assert r.returncode==0,r.stderr
  weights=fixture/'weights.bin';packed=fixture/'qx.bin';assert weights.stat().st_size==rows*6528 and packed.stat().st_size==6912
  if label=='basic-full':
   assert hashlib.sha256(weights.read_bytes()).hexdigest()=='96425599c0b6af6e3cc3fefe21bd6c03b7b2257a2e8e1c6ac8f1b6aab9479e69'
   assert hashlib.sha256(packed.read_bytes()).hexdigest()=='9391f116934a0f2b0865a97e5f0f3e0749be6bd6b413c19071ca7ccb72d5cc46'
  if label=='subnormal':
   assert hashlib.sha256(weights.read_bytes()).hexdigest()=='899d690eb7db509aebd3b6503e09db4b1a26f343b3bb8ec2d85fd5933b5f3b4c'
   assert hashlib.sha256(packed.read_bytes()).hexdigest()=='bd3af54e07a9cee57137304088cbe67121a5793604ad0ba4ae2eb0a2e04c1d48'
  contracts=[]
  for idx,exe in enumerate((control,native,vulkan)):
   output=d/f'contract-{label}-{idx}.json';r=run([exe,'--contract',weights,packed,rows,output],cwd=d);assert r.returncode==0,r.stderr;contracts.append(output.read_bytes())
  assert len(set(contracts))==1 and json.loads(contracts[0])=={'fields':{'out.f32le':rows*4},'qualified':False}
  env=dict(os.environ,Q8_TEST_TRACE=str(d/f'trace-{label}'));out=d/f'native-{label}'
  args=[native,'--device','0','--expect-device','signature-fixture GPU','--weights',weights,'--input',packed,'--rows',rows,'--budget',str(32<<20),'--output',out]
  r=run(args,cwd=d,env=env);assert r.returncode==0,r.stderr
  assert (d/f'trace-{label}').read_text().splitlines()==['create','launch','sync','launch','sync','destroy']
  assert (out/'out.f32le').read_bytes()==(out/'out-repeat.f32le').read_bytes()
  fake=d/f'vulkan-{label}';shutil.copytree(out,fake);(fake/'report.json').write_text(json.dumps({'actual_vulkan_wo_candidate':True,'immutable_canary_retry':True,'qualified':False}))
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  compare=[sys.executable,root/'tools/native_q8_wo_compare.py','--weights',weights,'--weight-sha',sha(weights),'--packed',packed,'--packed-sha',sha(packed),'--rows',rows,'--native',out,'--vulkan',fake,'--control',control,'--control-sha',sha(control)]
  assert run(compare,cwd=d).returncode==0
  if label=='subnormal':
   raw=weights.read_bytes()
   for m in range(1,1024):
    for row,h in ((7+m,m),(1030+m,0x8000|m)):
     block=raw[row*6528+34:row*6528+68]
     assert int.from_bytes(block[:2],'little')==h and block[2]==1 and not any(block[3:])
     assert not any(raw[row*6528:row*6528+34]) and not any(raw[row*6528+68:(row+1)*6528])
   literal=[sys.executable,root/'tools/native_q8_wo_subnormal_compare.py','--weights',weights,'--weight-sha',sha(weights),'--packed',packed,'--packed-sha',sha(packed),'--native',out,'--vulkan',fake,'--control',control,'--control-sha',sha(control)]
   r=run(literal,cwd=d);assert r.returncode==0,r.stderr;assert json.loads(r.stdout)['signed_subnormal_encodings']==2046
   for parent in (out,fake):
    for name in ('out.f32le','out-repeat.f32le'):
     path=parent/name;data=bytearray(path.read_bytes());data[8*4]^=1;path.write_bytes(data)
   r=run(literal,cwd=d);assert r.returncode==1 and not json.loads(r.stdout)['all_literals_exact']
   for parent in (out,fake):
    for name in ('out.f32le','out-repeat.f32le'):
     path=parent/name;data=bytearray(path.read_bytes());data[8*4]^=1;path.write_bytes(data)
  q=fake/'out.f32le';data=bytearray(q.read_bytes());data[-1]^=1;q.write_bytes(data);(fake/'out-repeat.f32le').write_bytes(data)
  r=run(compare,cwd=d);assert r.returncode==1 and json.loads(r.stdout)['different_bytes']==1
  rejected=list(args);rejected[-3]='1';rejected[-1]=str(d/f'bad-budget-{label}');r=run(rejected,cwd=d,env=env);assert r.returncode and not Path(rejected[-1]).exists()
 print('WO full geometry, actual producer contracts, owned native stream and exact-byte gate pass with simulated CUDA; no GPU')
