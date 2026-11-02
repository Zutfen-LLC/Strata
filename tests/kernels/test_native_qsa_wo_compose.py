"""Exercise actual native composition CLI with a signature-faithful host provider.

The provider computes host controls; it does not claim CUDA or Vulkan parity.
"""
from pathlib import Path
import hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile

root=Path(__file__).resolve().parents[2]
validator=Path(sys.argv[1]).resolve();wo_input=Path(sys.argv[2]).resolve();vulkan=Path(sys.argv[3]).resolve()
spec=importlib.util.spec_from_file_location('native_provider',root/'tests/kernels/test_native_qsa_projection_q8_native_cli.py')
provider_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider_module)

with tempfile.TemporaryDirectory(prefix='qsa-wo-compose-') as tmp:
    d=Path(tmp);provider_module.build(d,replay=True)
    path=d/'provider.cpp';prefix=path.read_text().split('namespace strata::kernels {')[0]
    path.write_text(prefix+r'''
#include "native_q8_wo_fixture.hpp"
namespace strata::kernels {
void native_quantize_q8_1(const float* x,void* q,int n,int cols,void* stream){
 validate_shape(n,cols);validate_pointer(x);validate_pointer(q);validate_stream(stream);
 if(stream!=active || n!=6144 || cols!=1)throw std::runtime_error("wrong quantizer stream or shape");
 log("quant");std::vector<float> input(x,x+n);auto bytes=native_q8_fixture::quantize(input);std::memcpy(q,bytes.data(),bytes.size());
}
void native_mmvq(int type,const void* weights,const void* x,float* y,int n,int rows,int cols,void* stream){
 if(type!=8)throw std::invalid_argument("unsupported native MMVQ GGML type");
 validate_shape(n,cols);validate_pointer(weights);validate_pointer(x);validate_pointer(y);validate_stream(stream);
 if(stream!=active || n!=6144 || cols!=1 || rows!=2560)throw std::runtime_error("wrong WO stream or shape");
 log("wo");using namespace native_q8_wo_fixture;
 native_q8_wo_fixture::Fixture f{size_t(rows),Bytes(size_t(rows)*q8_wo_row_bytes),Bytes(q8_wo_input_bytes)};
 std::memcpy(f.weights.data(),weights,f.weights.size());std::memcpy(f.input.data(),x,f.input.size());
 for(int row=0;row<rows;++row)y[row]=reference(f,size_t(row)).value;
}}
''')
    native=d/'native';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off',
        '-I'+str(d),'-I'+str(root/'include'),'-I'+str(root/'src/core'),'-I'+str(root/'tests/kernels'),'-x','c++',
        str(root/'tests/kernels/native_qsa_wo_compose_reference.cu'),str(path),'-o',str(native)],check=True)
    fixture=d/'input.bin';weights=d/'weights';subprocess.run([str(validator),'--synthetic-input',str(fixture)],check=True)
    subprocess.run([str(wo_input),'--synthetic','2560',str(weights)],check=True)
    for exe in (validator,native,vulkan):
        target=d/(exe.name+'.json')
        r=subprocess.run([str(exe),'--contract',str(weights/'weights.bin'),str(fixture),str(target)],cwd='/',capture_output=True,text=True)
        assert r.returncode==0 and not r.stdout and not r.stderr,(exe,r.stderr)
        assert json.loads(target.read_text())=={'fields':{'qx.bin':6912,'out.f32le':10240},'qualified':False}
    trace=d/'trace';out=d/'out'
    r=subprocess.run([str(native),'--device','0','--expect-device','signature-fixture GPU','--weights',str(weights/'weights.bin'),
                      '--fixture',str(fixture),'--output',str(out)],cwd='/',env=dict(os.environ,Q8_TEST_TRACE=str(trace)),capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stderr
    assert trace.read_text().splitlines()==['create','quant','sync','wo','sync','wo','sync','quant','sync','wo','sync','wo','sync','destroy']
    report=json.loads((out/'report.json').read_text());assert report['producer_buffer_consumed'] and report['stale_changes_output'] and report['invalid_noncommit_retry']
    assert (out/'qx.bin').read_bytes()==(out/'qx-repeat.bin').read_bytes() and (out/'out.f32le').read_bytes()==(out/'out-repeat.f32le').read_bytes()
    subprocess.run([str(validator),'--validate-packed',str(weights/'weights.bin'),str(fixture),str(out/'qx.bin')],check=True)
    bad=bytearray((out/'qx.bin').read_bytes());bad[:2]=b'\x01\x00';(d/'subnormal.bin').write_bytes(bad)
    r=subprocess.run([str(validator),'--validate-packed',str(weights/'weights.bin'),str(fixture),str(d/'subnormal.bin')],capture_output=True)
    assert r.returncode and b'unsupported half scale' in r.stderr
    other=d/'vulkan-fake';shutil.copytree(out,other)
    report=json.loads((other/'report.json').read_text());report.pop('actual_native_quantizer_and_mmvq');report.pop('same_stream');report['actual_vulkan_quantizer_and_wo']=True
    (other/'report.json').write_text(json.dumps(report));digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('compare',root/'tools/native_qsa_wo_compose_compare.py');compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    args=(weights/'weights.bin',digest(weights/'weights.bin'),fixture,digest(fixture),out,other,validator,digest(validator))
    assert compare.compare(*args)['exact']
    altered=bytearray((other/'qx.bin').read_bytes());altered[4]^=1
    (other/'qx.bin').write_bytes(altered);(other/'qx-repeat.bin').write_bytes(altered)
    result=compare.compare(*args);assert not result['exact'] and result['packed_different_bytes']==1
    (other/'qx.bin').write_bytes((out/'qx.bin').read_bytes());(other/'qx-repeat.bin').write_bytes((out/'qx.bin').read_bytes())
    altered=bytearray((other/'out.f32le').read_bytes());altered[0]^=1
    (other/'out.f32le').write_bytes(altered);(other/'out-repeat.f32le').write_bytes(altered)
    result=compare.compare(*args);assert not result['exact'] and result['output_different_bytes']==1
    print('composition startup contracts, owned stream, direct wiring, stale refusal, half-scale guard and both exact-byte gates passed; host provider only')
