"""Exercise the actual component CLIs using an explicit host CUDA provider."""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2];validator=Path(sys.argv[1]).resolve();vulkan=Path(sys.argv[2]).resolve()
spec=importlib.util.spec_from_file_location('provider',root/'tests/kernels/test_native_qsa_projection_q8_native_cli.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='gdn-input-contract-') as tmp:
 subprocess.run([str(validator),'--layout'],check=True,cwd='/')
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
 if(stream!=active||n!=2560||cols!=1||(rows!=10240&&rows!=6144))throw std::runtime_error("wrong QKV stream or shape");log("mmvq");
 using namespace native_q8_wo_fixture;auto*w=static_cast<const uint8_t*>(weights);auto*q=static_cast<const uint8_t*>(x);
 for(int row=0;row<rows;++row){std::array<float,128>p{};
  for(size_t tid=0;tid<128;++tid)for(size_t b=tid/4;b<80;b+=32){size_t z=2*(tid%4),wi=size_t(row)*2720+b*34,xi=b*36;int dot=0;
   for(size_t j=0;j<8;++j)dot+=s8(w[wi+2+z*4+j])*s8(q[xi+4+z*4+j]);
   p[tid]=checked(std::fma(checked(half(get16(w,wi))*input_half(get16(q,xi))),float(dot),p[tid]));}
  for(size_t lane=0;lane<32;++lane){float v=p[lane];for(size_t warp=1;warp<4;++warp)v=checked(v+p[warp*32+lane]);p[lane]=v;}
  for(size_t off=16;off;off>>=1){auto old=p;for(size_t lane=0;lane<32;++lane)p[lane]=checked(old[lane]+old[lane^off]);}y[row]=p[0];
 }} }
''')
 native=d/'native';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(d),'-I'+str(root/'include'),'-I'+str(root/'src/core'),'-I'+str(root/'tests/kernels'),'-x','c++',str(root/'tests/kernels/native_gdn_input_reference.cu'),str(provider),'-o',str(native)],check=True)
 fixture=d/'fixture';subprocess.run([str(validator),'--synthetic',str(fixture)],check=True)
 for exe in (validator,native,vulkan):
  out=d/(exe.name+'.json');r=subprocess.run([str(exe),'--contract',str(fixture),str(fixture/'input.bin'),str(out)],cwd='/',capture_output=True,text=True)
  assert r.returncode==0 and not r.stdout and not r.stderr,(exe,r.stderr)
  assert json.loads(out.read_text())=={'fields':{'qx.bin':2880,'qkv.f32le':40960,'z.f32le':24576},'qualified':False}
 trace=d/'trace'
 cspec=importlib.util.spec_from_file_location('compare',root/'tools/native_gdn_input_compare.py');compare=importlib.util.module_from_spec(cspec);cspec.loader.exec_module(compare)
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for case in ('input.bin','dense.bin'):
  trace.unlink(missing_ok=True);out=d/('native-'+case);fake=d/('vulkan-'+case)
  r=subprocess.run([str(native),'--device','0','--expect-device','signature-fixture GPU','--weights',str(fixture),'--fixture',str(fixture/case),'--output',str(out)],cwd='/',env=dict(os.environ,Q8_TEST_TRACE=str(trace)),capture_output=True,text=True,timeout=180)
  assert r.returncode==0,r.stderr
  expected=['create']+(['quant','sync']+['mmvq','sync','mmvq','sync']*2)*2+['destroy'];assert trace.read_text().splitlines()==expected
  shutil.copytree(out,fake);report=json.loads((fake/'report.json').read_text());report.pop('actual_native_quantizer_and_mmvq');report['actual_vulkan_quantizer_and_mmvq']=True;(fake/'report.json').write_text(json.dumps(report))
  args=(fixture,validator,digest(validator),out,fake,case);assert compare.compare(*args)['exact']
  for name,repeat in (('qx.bin','qx-repeat.bin'),('qkv.f32le','qkv-repeat.f32le'),('z.f32le','z-repeat.f32le')):
   altered=bytearray((fake/name).read_bytes());altered[400]^=1;(fake/name).write_bytes(altered);(fake/repeat).write_bytes(altered)
   try: result=compare.compare(*args)
   except ValueError: pass  # An independent literal may reject the corruption first.
   else: assert not result['exact']
   (fake/name).write_bytes((out/name).read_bytes());(fake/repeat).write_bytes((out/name).read_bytes())
  changed=fixture/case;original=changed.read_bytes();changed.write_bytes(original[:-1])
  try:compare.compare(*args)
  except ValueError:pass
  else:raise AssertionError('changed input fixture accepted')
  changed.write_bytes(original)
 print('Both actual startup contracts and native wrapper, packed literals, row controls, and exact byte gates passed with a host provider; no native/Vulkan parity is claimed.')
