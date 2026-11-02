"""Exercise the scale component CLIs using an explicit host CUDA provider."""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2];validator=Path(sys.argv[1]).resolve();vulkan=Path(sys.argv[2]).resolve();sys.path.insert(0,str(root/'tools'))
spec=importlib.util.spec_from_file_location('provider',root/'tests/kernels/test_native_qsa_projection_q8_native_cli.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='qsa-qkv-contract-') as tmp:
 d=Path(tmp);module.build(d,replay=True);provider=d/'provider.cpp';prefix=provider.read_text().split('namespace strata::kernels {')[0]
 provider.write_text(prefix+r'''
#include "native_q8_wo_fixture.hpp"
namespace strata::kernels {
void native_quantize_q8_1(const float*x,void*q,int n,int cols,void*stream){
 validate_shape(n,cols);validate_pointer(x);validate_pointer(q);validate_stream(stream);
 if(stream!=active||n!=2560||cols!=1)throw std::runtime_error("wrong QKV quantizer stream or shape");
 log("quant");auto v=native_q8_fixture::quantize(std::vector<float>(x,x+n));std::memcpy(q,v.data(),v.size());
}
void native_mmvq(int type,const void*weights,const void*x,float*y,int n,int rows,int cols,void*stream){
 if(type!=8)throw std::invalid_argument("unsupported native MMVQ GGML type");validate_shape(n,cols);validate_pointer(weights);validate_pointer(x);validate_pointer(y);validate_stream(stream);
 if(stream!=active||n!=2560||cols!=1||(rows!=12288&&rows!=512))throw std::runtime_error("wrong QKV stream or shape");log("mmvq");
 using namespace native_q8_wo_fixture;auto*w=static_cast<const uint8_t*>(weights);auto*q=static_cast<const uint8_t*>(x);
 for(int row=0;row<rows;++row){std::array<float,128>p{};
  for(size_t tid=0;tid<128;++tid)for(size_t b=tid/4;b<80;b+=32){size_t z=2*(tid%4),wi=size_t(row)*2720+b*34,xi=b*36;int dot=0;
   for(size_t j=0;j<8;++j)dot+=s8(w[wi+2+z*4+j])*s8(q[xi+4+z*4+j]);
   p[tid]=checked(std::fma(checked(half(get16(w,wi))*input_half(get16(q,xi),InputScaleDomain::qkv_finite)),float(dot),p[tid]));}
  for(size_t lane=0;lane<32;++lane){float v=p[lane];for(size_t warp=1;warp<4;++warp)v=checked(v+p[warp*32+lane]);p[lane]=v;}
  for(size_t off=16;off;off>>=1){auto old=p;for(size_t lane=0;lane<32;++lane)p[lane]=checked(old[lane]+old[lane^off]);}y[row]=p[0];
 }} }
''')
 native=d/'native';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-DSTRATA_QKV_Q81_SCALE_SUITE=1','-I'+str(d),'-I'+str(root/'include'),'-I'+str(root/'src/core'),'-I'+str(root/'tests/kernels'),'-x','c++',str(root/'tests/kernels/native_qsa_qkv_reference.cu'),str(provider),'-o',str(native)],check=True)
 fixture=d/'fixture';subprocess.run([str(validator),'--scale-suite',str(fixture)],check=True)
 for exe in (validator,native,vulkan):
  out=d/(exe.name+'.json');r=subprocess.run([str(exe),'--contract',str(fixture),str(fixture/'case-00.bin'),str(out)],cwd='/',capture_output=True,text=True)
  assert r.returncode==0 and not r.stdout and not r.stderr,(exe,r.stderr)
  assert json.loads(out.read_text())=={'fields':{'qx.bin':2880,'q.f32le':49152,'k.f32le':2048,'v.f32le':2048,'direct-qx.bin':2880,'direct-q.f32le':49152,'direct-k.f32le':2048,'direct-v.f32le':2048},'qualified':False}
 trace=d/'trace';out=d/'native-output';r=subprocess.run([str(native),'--device','0','--expect-device','signature-fixture GPU','--weights',str(fixture),'--fixture',str(fixture/'case-00.bin'),'--output',str(out)],cwd='/',env=dict(os.environ,Q8_TEST_TRACE=str(trace)),capture_output=True,text=True,timeout=180)
 assert r.returncode==0,r.stderr
 expected=['create']+(['quant','sync']+['mmvq','sync','mmvq','sync']*3+['mmvq','sync']*3)*2+['destroy'];assert trace.read_text().splitlines()==expected
 fake=d/'vulkan-output';shutil.copytree(out,fake);report=json.loads((fake/'report.json').read_text());report.pop('actual_native_quantizer_and_mmvq');report['actual_vulkan_quantizer_and_mmvq']=True;(fake/'report.json').write_text(json.dumps(report))
 cspec=importlib.util.spec_from_file_location('compare',root/'tools/native_qsa_qkv_scale_compare.py');compare=importlib.util.module_from_spec(cspec);cspec.loader.exec_module(compare)
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();args=(fixture,0,validator,digest(validator),out,fake);assert compare.compare(*args)['exact']
 altered=bytearray((fake/'qx.bin').read_bytes());altered[400]^=1;(fake/'qx.bin').write_bytes(altered);(fake/'qx-repeat.bin').write_bytes(altered);assert not compare.compare(*args)['exact']
 (fake/'qx.bin').write_bytes((out/'qx.bin').read_bytes());(fake/'qx-repeat.bin').write_bytes((out/'qx.bin').read_bytes())
 altered=bytearray((fake/'q.f32le').read_bytes());altered[400]^=1;(fake/'q.f32le').write_bytes(altered);(fake/'q-repeat.f32le').write_bytes(altered);assert not compare.compare(*args)['exact']
 changed=fixture/'case-00.bin';original=changed.read_bytes();changed.write_bytes(original[:-1])
 try:compare.compare(*args)
 except ValueError:pass
 else:raise AssertionError('changed input fixture accepted')
 changed.write_bytes(original)
 for case in (14,15):
  current=d/f'native-{case}';trace_case=d/f'trace-{case}';r=subprocess.run([str(native),'--device','0','--expect-device','signature-fixture GPU','--weights',str(fixture),'--fixture',str(fixture/f'case-{case:02}.bin'),'--output',str(current)],cwd='/',env=dict(os.environ,Q8_TEST_TRACE=str(trace_case)),capture_output=True,text=True,timeout=180)
  assert r.returncode==0,(case,r.stderr);assert trace_case.read_text().splitlines()==expected
  other=d/f'vulkan-{case}';shutil.copytree(current,other);report=json.loads((other/'report.json').read_text());report.pop('actual_native_quantizer_and_mmvq');report['actual_vulkan_quantizer_and_mmvq']=True;(other/'report.json').write_text(json.dumps(report))
  assert compare.compare(fixture,case,validator,digest(validator),current,other)['exact']
 print('scale native/Vulkan contracts, owned stream, direct and producer packed inputs, literals and exact bytes pass with host provider only')
